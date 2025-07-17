from pydantic import BaseModel, Field, validator
from typing import Optional, List, Union
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

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


class ResponseFormat(str, Enum):
    """Response format options"""
    URL = "url"
    B64_JSON = "b64_json"


class ModelType(str, Enum):
    """Available model types"""
    FLUX_DEV = "flux-dev"
    FLUX_KONTEXT_DEV = "flux-kontext-dev"


# ============================================================================
# BASE CLASSES
# ============================================================================

class BaseRequest(BaseModel):
    """Base class for all API requests"""
    model: Optional[str] = Field(default=ModelType.FLUX_DEV, description="The model to use for generation")
    response_format: Optional[ResponseFormat] = Field(
        default=ResponseFormat.B64_JSON, 
        description="The format in which the generated images are returned"
    )
    priority: Optional[Priority] = Field(default=Priority.NORMAL, description="Task priority in queue")
    async_mode: Optional[bool] = Field(default=False, description="Whether to process asynchronously")


class BaseResponse(BaseModel):
    """Base class for all API responses"""
    created: int = Field(..., description="The Unix timestamp (in seconds) when the response was created")


# ============================================================================
# REQUEST MODELS
# ============================================================================

class ImageGenerationRequest(BaseRequest):
    """Request schema for OpenAI-compatible image generation API"""
    prompt: str = Field(
        ..., 
        description="A text description of the desired image(s). The maximum length is 1000 characters.", 
        max_length=1000
    )
    size: Optional[ImageSize] = Field(default=ImageSize.LARGE, description="The size of the generated images")
    negative_prompt: Optional[str] = Field(None, description="The negative prompt to generate the image from")
    steps: Optional[int] = Field(default=20, ge=1, le=50, description="The number of inference steps (1-50)")
    seed: Optional[int] = Field(None, description="Seed for reproducible generation")
    image_strength: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="The strength of the image to generate")
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v or not v.strip():
            raise ValueError('Prompt cannot be empty')
        return v.strip()


class ImageEditRequest(BaseRequest):
    """Request schema for OpenAI-compatible image edit API"""
    prompt: str = Field(
        ..., 
        description="A text description of the desired edits. The maximum length is 1000 characters.", 
        max_length=1000
    )
    model: Optional[str] = Field(default=ModelType.FLUX_KONTEXT_DEV, description="The model to use for image editing")
    n: Optional[int] = Field(default=1, ge=1, le=1, description="The number of images to generate. Currently only supports 1.")
    size: Optional[str] = Field(default="1024x1024", description="The size of the generated images")
    guidance_scale: Optional[float] = Field(
        default=2.5, 
        ge=0.0, 
        le=20.0, 
        description="Guidance scale for the image editing"
    )
    user: Optional[str] = Field(None, description="Unique identifier representing your end-user (for abuse monitoring)")
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v or not v.strip():
            raise ValueError('Prompt cannot be empty')
        return v.strip()


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class ImageData(BaseModel):
    """Individual image data in the response"""
    url: Optional[str] = Field(None, description="The URL of the generated image, if response_format is url")
    b64_json: Optional[str] = Field(None, description="The base64-encoded JSON of the generated image, if response_format is b64_json")
    
    @validator('b64_json')
    def validate_b64_json(cls, v):
        if v is not None and not v.strip():
            raise ValueError('b64_json cannot be empty string')
        return v
    
    @validator('url')
    def validate_url(cls, v):
        if v is not None and not v.strip():
            raise ValueError('url cannot be empty string')
        return v


class ImageGenerationResponse(BaseResponse):
    """Response schema for OpenAI-compatible image generation API"""
    data: List[ImageData] = Field(..., description="List of generated images")
    
    @validator('data')
    def validate_data(cls, v):
        if not v:
            raise ValueError('Response must contain at least one image')
        return v


# ============================================================================
# ERROR MODELS
# ============================================================================

class ErrorCode(str, Enum):
    """Standard error codes"""
    CONTENT_FILTER = "contentFilter"
    GENERATION_ERROR = "generation_error"
    QUEUE_FULL = "queue_full"
    VALIDATION_ERROR = "validation_error"
    RATE_LIMIT = "rate_limit"
    INTERNAL_ERROR = "internal_error"


class ImageGenerationError(BaseModel):
    """Error response schema"""
    code: ErrorCode = Field(..., description="Standardized error code")
    message: str = Field(..., description="Human-readable error message")
    type: Optional[str] = Field(None, description="Error type classification")
    
    @validator('message')
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError('Error message cannot be empty')
        return v.strip()


class ImageGenerationErrorResponse(BaseResponse):
    """Error response wrapper"""
    error: ImageGenerationError = Field(..., description="Error details")


# ============================================================================
# UTILITY TYPES
# ============================================================================

# Union type for all possible responses
APIResponse = Union[ImageGenerationResponse, ImageGenerationErrorResponse]

# Union type for all possible requests
APIRequest = Union[ImageGenerationRequest, ImageEditRequest]
