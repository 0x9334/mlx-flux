import os
import sys
from pathlib import Path
from PIL import Image
from flux import FluxModel, ModelLoadError, ModelGenerationError, InvalidConfigurationError

# Configuration
MODEL_PATH = os.getenv("FLUX_MODEL_PATH", "flux-kontext")
CONFIG_NAME = "flux-kontext"
QUANTIZE = int(os.getenv("FLUX_QUANTIZE", "8"))

# Image paths - use relative paths or environment variables
WORKSPACE_DIR = Path.cwd()
INPUT_IMAGE_PATH = os.getenv("INPUT_IMAGE_PATH", str(WORKSPACE_DIR / "image.png"))
OUTPUT_IMAGE_PATH = os.getenv("OUTPUT_IMAGE_PATH", str(WORKSPACE_DIR / "output.png"))

def main():
    print("MLX-Flux Image Generation Example")
    print("=" * 40)
    
    # Check if input image exists
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"❌ Error: Input image not found at {INPUT_IMAGE_PATH}")
        print("Please ensure the image file exists or set INPUT_IMAGE_PATH environment variable")
        print(f"Example: export INPUT_IMAGE_PATH=/path/to/your/image.png")
        return 1
    
    # Load input image to get dimensions
    try:
        input_image = Image.open(INPUT_IMAGE_PATH)
        width, height = input_image.size
        print(f"✓ Input image loaded successfully")
        print(f"  Dimensions: {width}x{height}")
        print(f"  Format: {input_image.format}")
    except Exception as e:
        print(f"❌ Error loading input image: {e}")
        return 1
    
    # Initialize Flux model
    try:
        print(f"\n🔄 Loading {CONFIG_NAME} model...")
        flux_model = FluxModel(
            model_path=MODEL_PATH,
            config_name=CONFIG_NAME,
            quantize=QUANTIZE
        )
        print("✓ Model loaded successfully!")
        
        # Print model configuration
        config_info = flux_model.get_current_config()
        print(f"\n📋 Model Configuration:")
        print(f"  Config: {config_info['config_name']}")
        print(f"  Type: {config_info['type']}")
        print(f"  Path: {config_info['model_path']}")
        print(f"  Quantization: {config_info['quantize']}")
        print(f"  Default steps: {config_info['default_steps']}")
        print(f"  Default guidance: {config_info['default_guidance']}")
        print(f"  Is loaded: {config_info['is_loaded']}")
        
    except InvalidConfigurationError as e:
        print(f"❌ Configuration Error: {e}")
        print(f"Available configurations: {', '.join(FluxModel.get_available_configs())}")
        return 1
    except ModelLoadError as e:
        print(f"❌ Model Load Error: {e}")
        print("Please check:")
        print("  - Model path is correct")
        print("  - Model files are accessible")
        print("  - Required dependencies are installed")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error during model initialization: {e}")
        return 1
    
    # Generate image
    try:
        print(f"\n🎨 Generating image...")
        print(f"  Prompt: 'Make it more realistic'")
        print(f"  Seed: 42")
        print(f"  Dimensions: {width}x{height}")
        print(f"  Input image: {INPUT_IMAGE_PATH}")
        
        # Generate image with image editing
        generated_image = flux_model(
            prompt="Make it more realistic",
            seed=42,
            width=width,
            height=height,
            num_inference_steps=28,
            guidance=2.5,
            image_path=INPUT_IMAGE_PATH
        )
        
        # Save output image
        generated_image.save(OUTPUT_IMAGE_PATH)
        print(f"\n✅ Image generated successfully!")
        print(f"   Output saved to: {OUTPUT_IMAGE_PATH}")
        
        # Display image info
        print(f"\n📊 Generated Image Info:")
        print(f"  Size: {generated_image.size}")
        print(f"  Mode: {generated_image.mode}")
        print(f"  Format: {generated_image.format}")
        
        return 0
        
    except ModelGenerationError as e:
        print(f"❌ Image Generation Error: {e}")
        print("Please check:")
        print("  - Input parameters are valid")
        print("  - Input image is accessible")
        print("  - Model is properly loaded")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error during image generation: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)