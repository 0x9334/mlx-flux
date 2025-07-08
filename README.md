# MLX-Flux

A FastAPI server for Flux image generation using MLX, providing both a REST API and CLI interface for high-quality image generation.

## Features

- 🚀 **FastAPI REST API** with OpenAI-compatible endpoints
- 🖥️ **CLI Interface** for direct image generation and server management
- 🔧 **Multiple Model Configurations** (dev, schnell)
- ⚡ **MLX Acceleration** for Apple Silicon
- 📦 **Easy Installation** with pip
- 🎛️ **Configurable Parameters** (quantization, inference steps, etc.)

## Installation

### Option 1: Install from Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/eternalai/mlx-flux.git
cd mlx-flux

# Install the package
pip install -e .
```

### Option 2: Install from PyPI (if published)

```bash
pip install mlx-flux
```

### Option 3: Development Installation

```bash
# Clone the repository
git clone https://github.com/your-username/mlx-flux.git
cd mlx-flux

# Install with development dependencies
pip install -e .[dev]
```

## Quick Start

### CLI Usage

After installation, you can use the CLI commands:

```bash
# Start the API server
mlx-flux serve

# Generate an image directly
mlx-flux generate --prompt "A beautiful sunset over mountains"

# Test model loading
mlx-flux test-model

# Get information about available models
mlx-flux info
```

### API Server

Start the server with default settings:

```bash
mlx-flux serve
```

Or with custom configuration:

```bash
mlx-flux serve --host 0.0.0.0 --port 8000 --config dev --quantize 8
```

The API will be available at `http://localhost:8000` with:
- Interactive docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### API Usage

Generate an image via REST API:

```bash
curl -X POST "http://localhost:8000/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over mountains",
    "size": "1024x1024",
    "steps": 20,
    "model": "flux-dev"
  }'
```

## CLI Commands

### Server Commands

```bash
# Start server with default settings
mlx-flux serve

# Start server with custom configuration
mlx-flux serve --host 0.0.0.0 --port 8000 --config dev --quantize 8

# Start server with auto-reload (development)
mlx-flux serve --reload
```

### Generation Commands

```bash
# Generate image with prompt
mlx-flux generate --prompt "A beautiful landscape"

# Generate with custom parameters
mlx-flux generate \
  --prompt "A futuristic city" \
  --config schnell \
  --steps 10 \
  --width 512 \
  --height 512 \
  --seed 12345 \
  --output my_image.png
```

### Utility Commands

```bash
# Test model loading
mlx-flux test-model --config dev

# Show available models and configurations
mlx-flux info

# Show version
mlx-flux --version
```

## Configuration Options

### Model Configurations

- **dev**: High-quality image generation (slower, better quality)
- **schnell**: Fast image generation (faster, good quality)

### Quantization Levels

- **4**: Most aggressive quantization (smallest memory usage)
- **8**: Balanced quantization (recommended)
- **16**: Minimal quantization (highest quality, more memory)

### Environment Variables

Set these environment variables to configure the server:

```bash
export FLUX_MODEL_PATH="flux-dev"
export FLUX_CONFIG="dev"
export FLUX_QUANTIZE="8"
```

## API Reference

### POST /v1/images/generations

Generate images using the Flux model.

**Request Body:**
```json
{
  "prompt": "A beautiful sunset over mountains",
  "model": "flux-dev",
  "size": "1024x1024",
  "steps": 20,
  "seed": 12345,
  "negative_prompt": "blurry, low quality"
}
```

**Response:**
```json
{
  "created": 1677649751,
  "data": [
    {
      "b64_json": "base64-encoded-image-data",
      "url": null
    }
  ]
}
```

### Available Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /v1/images/generations` - Generate images

## Development

### Running Tests

```bash
# Install test dependencies
pip install -e .[test]

# Run tests
pytest
```

### Code Formatting

```bash
# Install dev dependencies
pip install -e .[dev]

# Format code
black .
isort .

# Type checking
mypy .
```

## Requirements

- Python 3.11+
- Apple Silicon Mac (for MLX acceleration)
- MLX library
- FastAPI
- Click

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if needed
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation
- Review the API docs at `/docs` when running the server

## Changelog

### v1.0.0
- Initial release
- FastAPI server with OpenAI-compatible API
- CLI interface for server management and direct generation
- Support for Flux dev and schnell models
- Configurable quantization and inference parameters