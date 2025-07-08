from mflux import Flux1, Config
from mflux.config.model_config import ModelConfig
from PIL import Image


class BaseFluxModel:
    """Base class for Flux models with common functionality."""
    
    def __init__(self, model_path: str, model_config: ModelConfig, quantize: int = 8):
        self.model_path = model_path
        self.model_config = model_config
        self.quantize = quantize
        self.flux = Flux1(model_config, local_path=model_path, quantize=quantize)

    def __call__(self, prompt: str, seed: int = 42, **kwargs) -> Image.Image:
        """Generate an image from a text prompt."""
        config = Config(**kwargs)
        
        result = self.flux.generate_image(
            config=config,
            prompt=prompt,
            seed=seed,
        )
        
        return result.image


class FluxSchnell(BaseFluxModel):
    """Flux Schnell model for fast image generation."""
    
    def __init__(self, model_path: str, quantize: int = 8):
        super().__init__(model_path, ModelConfig.schnell(), quantize)

class FluxKontext(BaseFluxModel):
    """Flux Kontext model for high-quality image generation."""
    
    def __init__(self, model_path: str, quantize: int = 8):
        super().__init__(model_path, ModelConfig.dev_kontext(), quantize)


class FluxDev(BaseFluxModel):
    """Flux Dev model for high-quality image generation."""
    
    def __init__(self, model_path: str, quantize: int = 8):
        super().__init__(model_path, ModelConfig.dev(), quantize)


class FluxModel:
    """Factory class for creating Flux models."""
    
    _MODEL_CLASSES = {
        "flux-schnell": FluxSchnell,
        "flux-dev": FluxDev,
        "flux-kontext": FluxKontext,
    }
    
    def __init__(self, model_path: str, config_name: str, quantize: int = 8):
        self.config_name = config_name
        self.model_path = model_path
        self.quantize = quantize
        
        if config_name not in self._MODEL_CLASSES:
            available_configs = ", ".join(self._MODEL_CLASSES.keys())
            raise ValueError(f"Invalid config name: {config_name}. Available options: {available_configs}")
        
        model_class = self._MODEL_CLASSES[config_name]
        self.flux = model_class(model_path, quantize)
    
    def __call__(self, prompt: str, seed: int = 42, **kwargs) -> Image.Image:
        """Generate an image using the configured model."""
        return self.flux(prompt, seed, **kwargs)
    
    @classmethod
    def get_available_configs(cls) -> list[str]:
        """Get list of available model configurations."""
        return list(cls._MODEL_CLASSES.keys())