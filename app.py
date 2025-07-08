from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
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

from flux import FluxModel
from schema import (
    ImageGenerationRequest, 
    ImageGenerationResponse, 
    ImageGenerationError, 
    ImageGenerationErrorResponse,
    ImageData
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
flux_model: Optional[FluxModel] = None

# Semaphore to limit concurrent image generation requests
image_generation_semaphore: Optional[asyncio.Semaphore] = None

def parse_image_size(size_str: str) -> tuple[int, int]:
    """Parse image size string to width, height tuple"""
    size_mapping = {
        "256x256": (256, 256),
        "512x512": (512, 512),
        "1024x1024": (1024, 1024),
    }
    return size_mapping.get(size_str, (1024, 1024))

def image_to_base64(image) -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode()

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
    global flux_model, image_generation_semaphore
    
    # Initialize semaphore for image generation concurrency control
    image_generation_semaphore = asyncio.Semaphore(1)
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
        "available_configs": FluxModel.get_available_configs(),
        "model_loaded": flux_model is not None
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
        error_response = ImageGenerationErrorResponse(
            created=int(time.time()),
            error=ImageGenerationError(
                code="model_not_loaded",
                message="Flux model is not loaded. Please check server logs and try again.",
                type="server_error"
            )
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error_response.dict()
        )
    
    # Check if semaphore is initialized
    if image_generation_semaphore is None:
        logger.error("Image generation semaphore is not initialized")
        error_response = ImageGenerationErrorResponse(
            created=int(time.time()),
            error=ImageGenerationError(
                code="server_not_ready",
                message="Server is not properly initialized. Please try again.",
                type="server_error"
            )
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error_response.dict()
        )
    
    # Use semaphore to limit concurrent requests to 1
    async with image_generation_semaphore:
        logger.info("Acquired semaphore for image generation")
        
        try:
            # Parse image size
            width, height = parse_image_size(request.size.value)
            
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
            
            # Convert to base64
            b64_image = image_to_base64(image)
            
            # Create response
            response = ImageGenerationResponse(
                created=int(time.time()),
                data=[
                    ImageData(
                        b64_json=b64_image,
                        url=None  # We're returning base64, not URL
                    )
                ]
            )
            
            logger.info("Image generation completed successfully, releasing semaphore")
            return response
            
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            error_response = ImageGenerationErrorResponse(
                created=int(time.time()),
                error=ImageGenerationError(
                    code="generation_error",
                    message=f"Failed to generate image: {str(e)}",
                    type="server_error"
                )
            )
            logger.info("Image generation failed, releasing semaphore")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error_response.dict()
            )

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