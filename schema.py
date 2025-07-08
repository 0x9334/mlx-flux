from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class ImageSize(str, Enum):
    """Available image sizes"""
    SMALL = "256x256"
    MEDIUM = "512x512"
    LARGE = "1024x1024"
    COSMOS_SIZE = "1024x1024"


class Priority(str, Enum):
    """Task priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class ImageGenerationRequest(BaseModel):
    """Request schema for OpenAI-compatible image generation API"""
    prompt: str = Field(..., description="A text description of the desired image(s). The maximum length is 1000 characters.", max_length=1000)
    model: Optional[str] = Field(default="flux-dev", description="The model to use for image generation")
    size: Optional[ImageSize] = Field(default=ImageSize.LARGE, description="The size of the generated images")
    negative_prompt: Optional[str] = Field(None, description="The negative prompt to generate the image from")
    steps: Optional[int] = Field(default=20, ge=1, le=50, description="The number of inference steps (1-50)")
    priority: Optional[Priority] = Field(default=Priority.NORMAL, description="Task priority in queue")
    async_mode: Optional[bool] = Field(default=False, description="Whether to process asynchronously")
    seed: Optional[int] = Field(None, description="Seed for reproducible generation")

class ImageData(BaseModel):
    """Individual image data in the response"""
    url: Optional[str] = Field(None, description="The URL of the generated image, if response_format is url")
    b64_json: Optional[str] = Field(None, description="The base64-encoded JSON of the generated image, if response_format is b64_json")

class ImageGenerationResponse(BaseModel):
    """Response schema for OpenAI-compatible image generation API"""
    created: int = Field(..., description="The Unix timestamp (in seconds) when the image was created")
    data: List[ImageData] = Field(..., description="List of generated images")

class ImageGenerationError(BaseModel):
    """Error response schema"""
    code: str = Field(..., description="Error code (e.g., 'contentFilter', 'generation_error', 'queue_full')")
    message: str = Field(..., description="Human-readable error message")
    type: Optional[str] = Field(None, description="Error type")

class ImageGenerationErrorResponse(BaseModel):
    """Error response wrapper"""
    created: int = Field(..., description="The Unix timestamp (in seconds) when the error occurred")
    error: ImageGenerationError = Field(..., description="Error details")