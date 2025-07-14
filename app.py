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
from typing import Optional
from PIL import Image
import tempfile

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

def initialize_flux_model(model_path: str = "flux-dev", config_name: str = "dev", quantize: int = 8) -> FluxModel:
    """Initialize the Flux model"""
    try:
        logger.info(f"Loading Flux model from {model_path} with config {config_name} and quantization {quantize}...")
        model = FluxModel(
            model_path=model_path,
            config_name=config_name,
            quantize=quantize
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
    
    try:
        flux_model = initialize_flux_model(model_path, config_name, quantize)
        logger.info("FastAPI startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model during startup: {e}")
        # Continue startup even if model fails to load
        # This allows the health endpoint to report the error
    
    yield
    
    # Shutdown
    logger.info("FastAPI shutdown completed")

# Create FastAPI app with lifespan
app = FastAPI(
    title="MLX-Flux Image Generation API",
    description="FastAPI server for Flux image generation using MLX",
    version="1.0.0",
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
    return {
        "message": "MLX-Flux Image Generation API", 
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "endpoints": {
            "image_generation": "/v1/images/generations",
            "image_editing": "/v1/images/edits"
        },
        "available_configs": FluxModel.get_available_configs(),
        "model_loaded": flux_model is not None,
        "supported_sizes": [size.value for size in ImageSize],
        "response_formats": [format.value for format in ResponseFormat],
        "supported_models": [model.value for model in ModelType]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok"
    }

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
                       f"(size: {width}x{height}, steps: {request.steps}, seed: {seed})")
            
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
                    num_inference_steps=request.steps
                )
            )
            
            generation_time = time.time() - start_time
            logger.info(f"Image generated successfully in {generation_time:.2f} seconds")
            
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
               config_name: str = "dev", quantize: int = 8):
    """Run the FastAPI server with specified configuration"""
    # Set environment variables for model configuration
    os.environ["FLUX_MODEL_PATH"] = model_path
    os.environ["FLUX_CONFIG"] = config_name
    os.environ["FLUX_QUANTIZE"] = str(quantize)
    
    logger.info(f"Starting MLX-Flux API server on {host}:{port}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Config: {config_name}")
    logger.info(f"Quantize: {quantize}")
    
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_server() 