from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import base64
import io
import time
import random
import os
import logging
import asyncio
import gc
from typing import Optional, List
from PIL import Image
import tempfile
from version import __version__

# MLX memory management imports
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None

from flux import FluxModel
from schema import (
    ImageGenerationRequest, 
    ImageGenerationResponse, 
    ImageGenerationError, 
    ImageGenerationErrorResponse,
    ImageEditRequest,
    ImageData,
    ImageSize,
    ErrorCode,
    ResponseFormat,
    Priority,
    ModelType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
flux_model: Optional[FluxModel] = None

# Semaphore to limit concurrent image generation requests
semaphore: Optional[asyncio.Semaphore] = None

def cleanup_inference_memory():
    """Clean up memory after inference operations to prevent memory leaks"""
    try:
        if MLX_AVAILABLE and mx is not None:
            # Clear MLX cache to free intermediate tensors
            mx.clear_cache()
            
            # Log memory usage for debugging
            if hasattr(mx, 'get_active_memory') and hasattr(mx, 'get_cache_memory'):
                active_memory = mx.get_active_memory()
                cache_memory = mx.get_cache_memory()
                logger.info(f"Memory after cleanup - Active: {active_memory / 1024 / 1024:.2f} MB, Cache: {cache_memory / 1024 / 1024:.2f} MB")
            
            # Force synchronization to ensure cleanup is complete
            mx.synchronize()
            
        # Python garbage collection
        gc.collect()
        
    except Exception as e:
        logger.warning(f"Memory cleanup failed: {e}")

def log_memory_usage(stage: str = ""):
    """Log current memory usage for debugging"""
    try:
        if MLX_AVAILABLE and mx is not None:
            if hasattr(mx, 'get_active_memory') and hasattr(mx, 'get_cache_memory'):
                active_memory = mx.get_active_memory()
                cache_memory = mx.get_cache_memory()
                peak_memory = mx.get_peak_memory() if hasattr(mx, 'get_peak_memory') else 0
                logger.info(f"Memory usage {stage} - Active: {active_memory / 1024 / 1024:.2f} MB, "
                           f"Cache: {cache_memory / 1024 / 1024:.2f} MB, "
                           f"Peak: {peak_memory / 1024 / 1024:.2f} MB")
    except Exception as e:
        logger.warning(f"Memory logging failed: {e}")

def force_memory_cleanup():
    """Force aggressive memory cleanup during shutdown or critical memory situations"""
    try:
        if MLX_AVAILABLE and mx is not None:
            # Clear cache multiple times to ensure cleanup
            mx.clear_cache()
            mx.synchronize()
            
            # Reset peak memory counter
            if hasattr(mx, 'reset_peak_memory'):
                mx.reset_peak_memory()
            
        # Multiple garbage collection passes
        for _ in range(3):
            gc.collect()
            
        logger.info("Forced memory cleanup completed")
        
    except Exception as e:
        logger.warning(f"Forced memory cleanup failed: {e}")

def parse_image_size(size: ImageSize) -> tuple[int, int]:
    """Parse ImageSize enum to width, height tuple"""
    size_mapping = {
        ImageSize.SMALL: (256, 256),
        ImageSize.MEDIUM: (512, 512),
        ImageSize.LARGE: (1024, 1024),
        ImageSize.COSMOS_SIZE: (1024, 1024),
    }
    return size_mapping.get(size, (1024, 1024))

def image_to_base64(image) -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode()

def create_error_response(error_code: ErrorCode, message: str, error_type: str = "server_error") -> ImageGenerationErrorResponse:
    """Create a standardized error response"""
    return ImageGenerationErrorResponse(
        created=int(time.time()),
        error=ImageGenerationError(
            code=error_code,
            message=message,
            type=error_type
        )
    )

def initialize_flux_model(model_path: str = "flux-dev", config_name: str = "dev", quantize: int = 8, 
                         lora_paths: Optional[List[str]] = None, lora_scales: Optional[List[float]] = None) -> FluxModel:
    """Initialize the Flux model"""
    try:
        logger.info(f"Loading Flux model from {model_path} with config {config_name} and quantization {quantize}...")
        if lora_paths:
            logger.info(f"Using LoRA adapters: {', '.join(lora_paths)}")
            if lora_scales:
                logger.info(f"LoRA scales: {', '.join(str(s) for s in lora_scales)}")
        
        model = FluxModel(
            model_path=model_path,
            config_name=config_name,
            quantize=quantize,
            lora_paths=lora_paths,
            lora_scales=lora_scales
        )
        logger.info("Flux model loaded successfully!")
        return model
    except Exception as e:
        logger.error(f"Error loading Flux model: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app"""
    # Startup
    global flux_model, semaphore
    
    # Initialize semaphore for image generation concurrency control
    semaphore = asyncio.Semaphore(1)
    logger.info("Image generation semaphore initialized")
    
    model_path = os.getenv("FLUX_MODEL_PATH", "flux-dev")
    config_name = os.getenv("FLUX_CONFIG", "dev")
    quantize = int(os.getenv("FLUX_QUANTIZE", "8"))
    
    # Parse LoRA parameters from environment variables
    lora_paths = None
    lora_scales = None
    
    lora_paths_env = os.getenv("FLUX_LORA_PATHS")
    lora_scales_env = os.getenv("FLUX_LORA_SCALES")
    
    if lora_paths_env:
        lora_paths = [path.strip() for path in lora_paths_env.split(',') if path.strip()]
    
    if lora_scales_env:
        lora_scales = [float(scale.strip()) for scale in lora_scales_env.split(',') if scale.strip()]
    
    try:
        # Log memory usage before model loading
        log_memory_usage("before model loading")
        
        flux_model = initialize_flux_model(model_path, config_name, quantize, lora_paths, lora_scales)
        
        # Log memory usage after model loading
        log_memory_usage("after model loading")
        
        logger.info("FastAPI startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model during startup: {e}")
        # Continue startup even if model fails to load
        # This allows the health endpoint to report the error
    
    yield
    
    # Shutdown
    logger.info("Starting FastAPI shutdown cleanup...")
    
    # Force aggressive memory cleanup during shutdown
    force_memory_cleanup()
    
    logger.info("FastAPI shutdown completed")

# Create FastAPI app with lifespan
app = FastAPI(
    title="MLX-Flux Image Generation API",
    description="FastAPI server for Flux image generation using MLX",
    version=__version__,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    # Get current LoRA configuration
    lora_info = {}
    if flux_model:
        try:
            current_config = flux_model.get_current_config()
            if 'lora_paths' in current_config:
                lora_info['lora_paths'] = current_config['lora_paths']
            if 'lora_scales' in current_config:
                lora_info['lora_scales'] = current_config['lora_scales']
        except:
            pass
    
    # Get memory usage information
    memory_info = {}
    if MLX_AVAILABLE and mx is not None:
        try:
            if hasattr(mx, 'get_active_memory') and hasattr(mx, 'get_cache_memory'):
                memory_info = {
                    "active_memory_mb": round(mx.get_active_memory() / 1024 / 1024, 2),
                    "cache_memory_mb": round(mx.get_cache_memory() / 1024 / 1024, 2),
                }
                if hasattr(mx, 'get_peak_memory'):
                    memory_info["peak_memory_mb"] = round(mx.get_peak_memory() / 1024 / 1024, 2)
        except:
            pass
    
    return {
        "message": "MLX-Flux Image Generation API", 
        "version": __version__,
        "docs_url": "/docs",
        "health_url": "/health",
        "endpoints": {
            "image_generation": "/v1/images/generations",
            "image_editing": "/v1/images/edits",
            "memory_cleanup": "/cleanup-memory"
        },
        "available_configs": FluxModel.get_available_configs(),
        "model_loaded": flux_model is not None,
        "supported_sizes": [size.value for size in ImageSize],
        "response_formats": [format.value for format in ResponseFormat],
        "supported_models": [model.value for model in ModelType],
        "lora_configuration": lora_info,
        "memory_info": memory_info
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with memory information"""
    memory_info = {}
    if MLX_AVAILABLE and mx is not None:
        try:
            if hasattr(mx, 'get_active_memory') and hasattr(mx, 'get_cache_memory'):
                memory_info = {
                    "active_memory_mb": round(mx.get_active_memory() / 1024 / 1024, 2),
                    "cache_memory_mb": round(mx.get_cache_memory() / 1024 / 1024, 2),
                }
                if hasattr(mx, 'get_peak_memory'):
                    memory_info["peak_memory_mb"] = round(mx.get_peak_memory() / 1024 / 1024, 2)
        except:
            pass
    
    return {
        "status": "ok",
        "model_loaded": flux_model is not None,
        "mlx_available": MLX_AVAILABLE,
        "memory_info": memory_info
    }

@app.post("/cleanup-memory")
async def cleanup_memory_endpoint():
    """Manual memory cleanup endpoint for debugging"""
    try:
        # Log memory usage before cleanup
        log_memory_usage("before manual cleanup")
        
        # Perform cleanup
        cleanup_inference_memory()
        
        # Log memory usage after cleanup
        log_memory_usage("after manual cleanup")
        
        # Get final memory info
        memory_info = {}
        if MLX_AVAILABLE and mx is not None:
            try:
                if hasattr(mx, 'get_active_memory') and hasattr(mx, 'get_cache_memory'):
                    memory_info = {
                        "active_memory_mb": round(mx.get_active_memory() / 1024 / 1024, 2),
                        "cache_memory_mb": round(mx.get_cache_memory() / 1024 / 1024, 2),
                    }
                    if hasattr(mx, 'get_peak_memory'):
                        memory_info["peak_memory_mb"] = round(mx.get_peak_memory() / 1024 / 1024, 2)
            except:
                pass
        
        return {
            "status": "success",
            "message": "Memory cleanup completed",
            "memory_info": memory_info
        }
    except Exception as e:
        logger.error(f"Manual memory cleanup failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": f"Memory cleanup failed: {str(e)}"
            }
        )

@app.post("/v1/images/generations", response_model=ImageGenerationResponse)
async def generate_image(request: ImageGenerationRequest):
    """Generate image using Flux model"""
    global flux_model
    
    if flux_model is None:
        logger.error("Flux model is not loaded")
        error_response = create_error_response(
            ErrorCode.INTERNAL_ERROR,
            "Flux model is not loaded. Please check server logs and try again."
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error_response.dict()
        )
    
    # Check if semaphore is initialized
    if semaphore is None:
        logger.error("Image generation semaphore is not initialized")
        error_response = create_error_response(
            ErrorCode.INTERNAL_ERROR,
            "Server is not properly initialized. Please try again."
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error_response.dict()
        )
    
    # Use semaphore to limit concurrent requests to 1
    async with semaphore:
        logger.info("Acquired semaphore for image generation")
        
        try:
            # Parse image size using the enum
            width, height = parse_image_size(request.size)
            
            # Generate seed if not provided
            seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
            
            # Log generation request
            logger.info(f"Generating image with prompt: '{request.prompt[:50]}...' "
                       f"(size: {width}x{height}, steps: {request.steps}, seed: {seed}, image_strength: {request.image_strength})")
            
            # Log memory usage before inference
            log_memory_usage("before inference")
            
            # Generate the image using the FluxModel in a separate thread to avoid blocking
            start_time = time.time()
            loop = asyncio.get_event_loop()
            image = await loop.run_in_executor(
                None,  # Use default ThreadPoolExecutor
                lambda: flux_model(
                    prompt=request.prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_inference_steps=request.steps,
                    image_strength=request.image_strength
                )
            )
            
            generation_time = time.time() - start_time
            logger.info(f"Image generated successfully in {generation_time:.2f} seconds")
            
            # Log memory usage after inference, before cleanup
            log_memory_usage("after inference")
            
            # Clean up inference memory to prevent memory leaks
            cleanup_inference_memory()
            
            # Create response data based on response format
            if request.response_format == ResponseFormat.B64_JSON:
                b64_image = image_to_base64(image)
                image_data = ImageData(b64_json=b64_image, url=None)
            else:  # URL format - for future implementation
                # For now, we only support base64
                b64_image = image_to_base64(image)
                image_data = ImageData(b64_json=b64_image, url=None)
            
            # Create response
            response = ImageGenerationResponse(
                created=int(time.time()),
                data=[image_data]
            )
            
            logger.info("Image generation completed successfully, releasing semaphore")
            return response
            
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            # Clean up memory even in error cases
            cleanup_inference_memory()
            error_response = create_error_response(
                ErrorCode.VALIDATION_ERROR,
                str(e),
                "validation_error"
            )
            logger.info("Image generation failed due to validation error, releasing semaphore")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=error_response.dict()
            )
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            # Clean up memory even in error cases
            cleanup_inference_memory()
            error_response = create_error_response(
                ErrorCode.GENERATION_ERROR,
                f"Failed to generate image: {str(e)}"
            )
            logger.info("Image generation failed, releasing semaphore")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error_response.dict()
            )

@app.post("/v1/images/edits", response_model=ImageGenerationResponse)
async def create_image_edit(
    image: UploadFile = File(..., description="The image to edit. Must be a valid PNG file, less than 4MB, and square."),
    prompt: str = Form(..., description="A text description of the desired image edits. The maximum length is 1000 characters."),
    model: Optional[str] = Form(default=ModelType.FLUX_KONTEXT_DEV, description="The model to use for image editing."),
    n: Optional[int] = Form(default=1, ge=1, le=1, description="The number of images to generate. Currently only supports 1."),
    size: Optional[str] = Form(default="1024x1024", description="The size of the generated images."),
    response_format: Optional[str] = Form(default=ResponseFormat.B64_JSON, description="The format in which the generated images are returned."),
    guidance_scale: Optional[float] = Form(default=2.5, ge=0.0, le=20.0, description="Guidance scale for the image editing."),
    user: Optional[str] = Form(default=None, description="Unique identifier representing your end-user."),
    priority: Optional[str] = Form(default=Priority.NORMAL, description="Task priority in queue."),
    async_mode: Optional[bool] = Form(default=False, description="Whether to process asynchronously.")
):
    """Edit image using Flux model with provided image and prompt"""
    global flux_model
    
    if flux_model is None:
        logger.error("Flux model is not loaded")
        error_response = create_error_response(
            ErrorCode.INTERNAL_ERROR,
            "Flux model is not loaded. Please check server logs and try again."
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error_response.dict()
        )
    
    # Check if semaphore is initialized
    if semaphore is None:
        logger.error("Image generation semaphore is not initialized")
        error_response = create_error_response(
            ErrorCode.INTERNAL_ERROR,
            "Server is not properly initialized. Please try again."
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error_response.dict()
        )
    
    # Validate file type and size
    if image.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        error_response = create_error_response(
            ErrorCode.VALIDATION_ERROR,
            "Image must be a PNG, JPEG, or JPG file."
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=error_response.dict()
        )
    
    # Create request object for validation
    try:
        edit_request = ImageEditRequest(
            prompt=prompt,
            model=model,
            n=n,
            size=size,
            response_format=response_format,
            guidance_scale=guidance_scale,
            user=user,
            priority=priority,
            async_mode=async_mode
        )
    except Exception as e:
        logger.error(f"Request validation error: {e}")
        error_response = create_error_response(
            ErrorCode.VALIDATION_ERROR,
            f"Invalid request parameters: {str(e)}"
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=error_response.dict()
        )
    
    # Use semaphore to limit concurrent requests
    async with semaphore:
        logger.info("Acquired semaphore for image editing")
        
        try:
            
            # Read and process the uploaded image
            image_contents = await image.read()
            input_image = Image.open(io.BytesIO(image_contents))
            
            # Convert to RGB if necessary
            if input_image.mode != 'RGB':
                input_image = input_image.convert('RGB')
            
            width, height = input_image.size


            # Save to temporary file for FluxModel
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                input_image.save(temp_file.name, format='PNG')
                temp_image_path = temp_file.name
            
            # Generate seed
            seed = random.randint(0, 2**32 - 1)
            
            # Log generation request
            logger.info(f"Editing image with prompt: '{prompt[:50]}...' "
                       f"(size: {width}x{height}, guidance: {guidance_scale}, seed: {seed})")
            
            # Log memory usage before inference
            log_memory_usage("before image editing")
            
            # Generate the edited image using FluxModel
            start_time = time.time()
            loop = asyncio.get_event_loop()
            edited_image = await loop.run_in_executor(
                None,
                lambda: flux_model(
                    prompt=prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_inference_steps=28,  # Default for image editing
                    guidance=guidance_scale,
                    image_path=temp_image_path
                )
            )

            # resize to width, height
            edited_image = edited_image.resize((width, height), Image.Resampling.LANCZOS)
            
            generation_time = time.time() - start_time
            logger.info(f"Image edited successfully in {generation_time:.2f} seconds")
            
            # Log memory usage after inference, before cleanup
            log_memory_usage("after image editing")
            
            # Clean up inference memory to prevent memory leaks
            cleanup_inference_memory()
            
            # Clean up temporary file
            os.unlink(temp_image_path)
            
            # Create response data based on response format
            if response_format == ResponseFormat.B64_JSON:
                b64_image = image_to_base64(edited_image)
                image_data = ImageData(b64_json=b64_image, url=None)
            else:  # URL format - for future implementation
                b64_image = image_to_base64(edited_image)
                image_data = ImageData(b64_json=b64_image, url=None)
            
            # Create response
            response = ImageGenerationResponse(
                created=int(time.time()),
                data=[image_data]
            )
            
            logger.info("Image editing completed successfully, releasing semaphore")
            return response
            
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            # Clean up memory even in error cases
            cleanup_inference_memory()
            error_response = create_error_response(
                ErrorCode.VALIDATION_ERROR,
                str(e),
                "validation_error"
            )
            logger.info("Image editing failed due to validation error, releasing semaphore")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=error_response.dict()
            )
        except Exception as e:
            logger.error(f"Error editing image: {e}")
            # Clean up memory even in error cases
            cleanup_inference_memory()
            error_response = create_error_response(
                ErrorCode.GENERATION_ERROR,
                f"Failed to edit image: {str(e)}"
            )
            logger.info("Image editing failed, releasing semaphore")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error_response.dict()
            )
        finally:
            # Clean up temp file if it still exists
            if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
                os.unlink(temp_image_path)

def run_server(host: str = "0.0.0.0", port: int = 8000, model_path: str = "flux-dev", 
               config_name: str = "dev", quantize: int = 8, lora_paths: Optional[List[str]] = None,
               lora_scales: Optional[List[float]] = None):
    """Run the FastAPI server with specified configuration"""
    # Set environment variables for model configuration
    os.environ["FLUX_MODEL_PATH"] = model_path
    os.environ["FLUX_CONFIG"] = config_name
    os.environ["FLUX_QUANTIZE"] = str(quantize)
    
    # Set LoRA environment variables
    if lora_paths:
        os.environ["FLUX_LORA_PATHS"] = ",".join(lora_paths)
        if lora_scales:
            os.environ["FLUX_LORA_SCALES"] = ",".join(str(s) for s in lora_scales)
        else:
            # Default to scale 1.0 for all LoRA adapters
            os.environ["FLUX_LORA_SCALES"] = ",".join("1.0" for _ in lora_paths)
    
    logger.info(f"Starting MLX-Flux API server on {host}:{port}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Config: {config_name}")
    logger.info(f"Quantize: {quantize}")
    if lora_paths:
        logger.info(f"LoRA paths: {', '.join(lora_paths)}")
        logger.info(f"LoRA scales: {os.environ.get('FLUX_LORA_SCALES', 'default')}")
    
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_server() 