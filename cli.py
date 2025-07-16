#!/usr/bin/env python3
"""
MLX-Flux CLI Interface
"""

import click
import os
import sys
from typing import Optional, List
import uvicorn
import time
import random
from version import __version__

from flux import FluxModel
from app import run_server


@click.group()
@click.version_option(version=__version__, prog_name="mlx-flux")
def cli():
    """MLX-Flux: FastAPI server for Flux image generation using MLX"""
    pass


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, type=int, help='Port to bind to')
@click.option('--model-path', default='flux-dev', help='Path to the model')
@click.option('--config-name', default='flux-dev', type=click.Choice(['flux-dev', 'flux-schnell', 'flux-kontext']), help='Model configuration')
@click.option('--quantize', default=8, type=click.Choice([4, 8, 16]), help='Quantization level')
@click.option('--lora-paths', multiple=True, help='Paths to LoRA adapters (can be specified multiple times)')
@click.option('--lora-scales', multiple=True, type=float, help='Scales for LoRA adapters (can be specified multiple times, should match number of paths)')
@click.option('--reload', is_flag=True, help='Auto-reload on code changes (development)')
def serve(host: str, port: int, model_path: str, config_name: str, quantize: int, 
          lora_paths: tuple, lora_scales: tuple, reload: bool):
    """Start the FastAPI server"""
    
    # Validate LoRA parameters
    if lora_paths and lora_scales:
        if len(lora_paths) != len(lora_scales):
            click.echo("Error: Number of LoRA paths must match number of LoRA scales", err=True)
            sys.exit(1)
    elif lora_paths and not lora_scales:
        # Default to scale 1.0 for all LoRA adapters
        lora_scales = tuple(1.0 for _ in lora_paths)
    elif lora_scales and not lora_paths:
        click.echo("Error: LoRA scales provided but no LoRA paths specified", err=True)
        sys.exit(1)
    
    if reload:
        # For development mode with auto-reload
        os.environ["FLUX_MODEL_PATH"] = model_path
        os.environ["FLUX_CONFIG"] = config_name
        os.environ["FLUX_QUANTIZE"] = str(quantize)
        
        # Set LoRA environment variables
        if lora_paths:
            os.environ["FLUX_LORA_PATHS"] = ",".join(lora_paths)
            os.environ["FLUX_LORA_SCALES"] = ",".join(str(s) for s in lora_scales)
        
        click.echo(f"Starting MLX-Flux API server in development mode...")
        click.echo(f"Host: {host}")
        click.echo(f"Port: {port}")
        click.echo(f"Model Path: {model_path}")
        click.echo(f"Config: {config_name}")
        click.echo(f"Quantize: {quantize}")
        if lora_paths:
            click.echo(f"LoRA Paths: {', '.join(lora_paths)}")
            click.echo(f"LoRA Scales: {', '.join(str(s) for s in lora_scales)}")
        
        uvicorn.run(
            "app:app",
            host=host,
            port=port,
            reload=True,
            reload_dirs=["."],
            reload_includes=["*.py"]
        )
    else:
        run_server(
            host=host, 
            port=port, 
            model_path=model_path, 
            config_name=config_name, 
            quantize=quantize,
            lora_paths=list(lora_paths) if lora_paths else None,
            lora_scales=list(lora_scales) if lora_scales else None
        )


@cli.command()
@click.option('--prompt', required=True, help='Text prompt for image generation')
@click.option('--model-path', default='flux-dev', help='Path to the model')
@click.option('--config-name', default='flux-dev', type=click.Choice(['flux-dev', 'flux-schnell', 'flux-kontext']), help='Model configuration')
@click.option('--steps', default=20, type=int, help='Number of inference steps')
@click.option('--width', default=1024, type=int, help='Image width')
@click.option('--height', default=1024, type=int, help='Image height')
@click.option('--seed', type=int, help='Random seed for reproducibility')
@click.option('--quantize', default=8, type=click.Choice([4, 8, 16]), help='Quantization level')
@click.option('--lora-paths', multiple=True, help='Paths to LoRA adapters (can be specified multiple times)')
@click.option('--lora-scales', multiple=True, type=float, help='Scales for LoRA adapters (can be specified multiple times, should match number of paths)')
@click.option('--output', default='generated_image.png', help='Output filename')
def generate(prompt: str, model_path: str, config_name: str, steps: int, width: int, height: int, 
             seed: Optional[int], quantize: int, lora_paths: tuple, lora_scales: tuple, output: str):
    """Generate an image directly using the CLI"""
    try:
        # Generate seed if not provided
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        # Validate LoRA parameters
        if lora_paths and lora_scales:
            if len(lora_paths) != len(lora_scales):
                click.echo("Error: Number of LoRA paths must match number of LoRA scales", err=True)
                sys.exit(1)
        elif lora_paths and not lora_scales:
            # Default to scale 1.0 for all LoRA adapters
            lora_scales = tuple(1.0 for _ in lora_paths)
        elif lora_scales and not lora_paths:
            click.echo("Error: LoRA scales provided but no LoRA paths specified", err=True)
            sys.exit(1)
        
        click.echo(f"Generating image with the following parameters:")
        click.echo(f"  Prompt: {prompt}")
        click.echo(f"  Model Path: {model_path}")
        click.echo(f"  Config: {config_name}")
        click.echo(f"  Steps: {steps}")
        click.echo(f"  Size: {width}x{height}")
        click.echo(f"  Seed: {seed}")
        click.echo(f"  Quantize: {quantize}")
        if lora_paths:
            click.echo(f"  LoRA Paths: {', '.join(lora_paths)}")
            click.echo(f"  LoRA Scales: {', '.join(str(s) for s in lora_scales)}")
        
        # Load the model
        click.echo("Loading Flux model...")
        flux_model = FluxModel(
            model_path=model_path,
            config_name=config_name,
            quantize=quantize,
            lora_paths=list(lora_paths) if lora_paths else None,
            lora_scales=list(lora_scales) if lora_scales else None
        )
        
        # Generate the image
        click.echo("Generating image...")
        image = flux_model(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_inference_steps=steps
        )
        
        # Save the image
        image.save(output)
        click.echo(f"Image saved as: {output}")
        
    except Exception as e:
        click.echo(f"Error generating image: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--model-path', default='flux-dev', help='Path to the model')
@click.option('--config-name', default='flux-dev', type=click.Choice(['flux-dev', 'flux-schnell', 'flux-kontext']), help='Model configuration to test')
@click.option('--quantize', default=8, type=click.Choice([4, 8, 16]), help='Quantization level')
@click.option('--lora-paths', multiple=True, help='Paths to LoRA adapters (can be specified multiple times)')
@click.option('--lora-scales', multiple=True, type=float, help='Scales for LoRA adapters (can be specified multiple times, should match number of paths)')
def test_model(model_path: str, config_name: str, quantize: int, lora_paths: tuple, lora_scales: tuple):
    """Test model loading and basic functionality"""
    try:
        # Validate LoRA parameters
        if lora_paths and lora_scales:
            if len(lora_paths) != len(lora_scales):
                click.echo("Error: Number of LoRA paths must match number of LoRA scales", err=True)
                sys.exit(1)
        elif lora_paths and not lora_scales:
            # Default to scale 1.0 for all LoRA adapters
            lora_scales = tuple(1.0 for _ in lora_paths)
        elif lora_scales and not lora_paths:
            click.echo("Error: LoRA scales provided but no LoRA paths specified", err=True)
            sys.exit(1)
        
        click.echo(f"Testing model loading with config: {config_name}")
        click.echo(f"Model Path: {model_path}")
        click.echo(f"Quantization level: {quantize}")
        if lora_paths:
            click.echo(f"LoRA Paths: {', '.join(lora_paths)}")
            click.echo(f"LoRA Scales: {', '.join(str(s) for s in lora_scales)}")
        
        # Test model loading
        click.echo("Loading model...")
        start_time = time.time()
        
        flux_model = FluxModel(
            model_path=model_path,
            config_name=config_name,
            quantize=quantize,
            lora_paths=list(lora_paths) if lora_paths else None,
            lora_scales=list(lora_scales) if lora_scales else None
        )
        
        load_time = time.time() - start_time
        click.echo(f"Model loaded successfully in {load_time:.2f} seconds")
        
        # Test basic generation
        click.echo("Testing image generation...")
        start_time = time.time()
        
        test_image = flux_model(
            prompt="A simple test image",
            seed=42,
            height=512,
            width=512,
            num_inference_steps=4  # Use fewer steps for quick test
        )
        
        generation_time = time.time() - start_time
        click.echo(f"Test image generated successfully in {generation_time:.2f} seconds")
        
        # Save test image
        test_output = f"test_output_{config_name}_{quantize}.png"
        test_image.save(test_output)
        click.echo(f"Test image saved as: {test_output}")
        
        click.echo("✅ Model test completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Model test failed: {e}", err=True)
        sys.exit(1)


@cli.command()
def info():
    """Show information about available models and configurations"""
    click.echo("MLX-Flux Information")
    click.echo("=" * 50)
    
    # Available configurations
    try:
        available_configs = FluxModel.get_available_configs()
        click.echo(f"Available configurations: {', '.join(available_configs)}")
    except Exception as e:
        click.echo(f"Could not get available configurations: {e}")
    
    # Configuration details
    click.echo("\nConfiguration Details:")
    click.echo("  flux-dev: High-quality image generation (slower, better quality)")
    click.echo("  flux-schnell: Fast image generation (faster, good quality)")
    click.echo("  flux-kontext: Specialized configuration for high-quality generation")
    
    # Quantization levels
    click.echo("\nQuantization Levels:")
    click.echo("  4: Most aggressive quantization (smallest memory usage)")
    click.echo("  8: Balanced quantization (recommended)")
    click.echo("  16: Minimal quantization (highest quality, more memory)")
    
    # LoRA Support
    click.echo("\nLoRA Support:")
    click.echo("  --lora-paths: Specify paths to LoRA adapters")
    click.echo("  --lora-scales: Specify scales for LoRA adapters (default: 1.0)")
    click.echo("  Example: --lora-paths path1.safetensors --lora-paths path2.safetensors --lora-scales 0.8 --lora-scales 1.2")
    
    # Environment variables
    click.echo("\nEnvironment Variables:")
    click.echo("  FLUX_MODEL_PATH: Path to model (default: flux-dev)")
    click.echo("  FLUX_CONFIG: Model configuration (default: flux-dev)")
    click.echo("  FLUX_QUANTIZE: Quantization level (default: 8)")
    click.echo("  FLUX_LORA_PATHS: Comma-separated LoRA paths")
    click.echo("  FLUX_LORA_SCALES: Comma-separated LoRA scales")
    
    # Current settings
    click.echo("\nCurrent Environment:")
    click.echo(f"  FLUX_MODEL_PATH: {os.getenv('FLUX_MODEL_PATH', 'flux-dev')}")
    click.echo(f"  FLUX_CONFIG: {os.getenv('FLUX_CONFIG', 'flux-dev')}")
    click.echo(f"  FLUX_QUANTIZE: {os.getenv('FLUX_QUANTIZE', '8')}")
    click.echo(f"  FLUX_LORA_PATHS: {os.getenv('FLUX_LORA_PATHS', 'None')}")
    click.echo(f"  FLUX_LORA_SCALES: {os.getenv('FLUX_LORA_SCALES', 'None')}")
    
    # Default model path
    click.echo("\nDefault Model Path:")
    click.echo("  flux-dev (can be overridden with --model-path option)")


if __name__ == '__main__':
    cli() 