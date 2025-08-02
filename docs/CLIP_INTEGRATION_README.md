# CLIP Integration with sentence-transformers

This document describes the integration of CLIP (Contrastive Language-Image Pre-training) models using the sentence-transformers library into the search AI system.

## Overview

The integration adds multimodal capabilities to the existing `SentenceTransformersModel` class, allowing it to encode both text and images using CLIP models. This enables text-image similarity computation and cross-modal search functionality.

## Features

- **Unified Interface**: Same model factory system works for both text-only and multimodal models
- **Auto-detection**: Automatically detects whether inputs are text or image paths
- **Batch Processing**: Efficient batch encoding for both text and images
- **Multiple CLIP Models**: Support for various CLIP architectures (ViT-B/32, ViT-L/14, ViT-B/16)
- **Seamless Integration**: Works with existing retrievers and search infrastructure

## Available CLIP Models

The following CLIP models are configured and ready to use:

| Model Name | Type | Dimension | Description |
|------------|------|-----------|-------------|
| `sentence-transformers-clip-ViT-B-32` | sentence_transformers | 512 | Standard CLIP model, good balance of speed and quality |
| `sentence-transformers-clip-ViT-L-14` | sentence_transformers | 768 | Large CLIP model, higher quality embeddings |
| `sentence-transformers-clip-ViT-B-16` | sentence_transformers | 512 | CLIP with ViT-B/16 architecture |

## Usage Examples

### Basic Text Encoding

```python
from src.models.model_factory import ModelFactory
from sentence_transformers import util

# Load CLIP model
model = ModelFactory.create_model(
    model_type="sentence_transformers",
    model_name="sentence-transformers-clip-ViT-B-32"
)

# Encode text
texts = ['Two dogs in the snow', 'A cat on a table']
text_embeddings = model.encode(texts, input_type="text")
print(f"Text embeddings shape: {text_embeddings.shape}")
```

### Image Encoding

```python
from PIL import Image

# Encode images (requires actual image files)
image_paths = ['path/to/image1.jpg', 'path/to/image2.png']
image_embeddings = model.encode(image_paths, input_type="image")
print(f"Image embeddings shape: {image_embeddings.shape}")
```

### Text-Image Similarity

```python
# Compute cross-modal similarity
cos_scores = util.cos_sim(image_embeddings, text_embeddings)
print(f"Similarity scores: {cos_scores}")
```

### Auto-Detection

```python
# Auto-detect input type (text vs image paths)
auto_text = model.encode("A beautiful landscape")  # Detected as text
auto_image = model.encode("image.jpg")  # Detected as image (if file exists)
```

### Integration with Existing Code

The CLIP models work seamlessly with your existing model factory system:

```python
# List all available models including CLIP
all_models = ModelFactory.list_configured_models()
clip_models = {k: v for k, v in all_models.items() 
              if 'clip' in k.lower() and v['type'] == 'sentence_transformers'}

for model_key, info in clip_models.items():
    print(f"{model_key}: {info['description']}")
```

## Implementation Details

### Enhanced SentenceTransformersModel

The `SentenceTransformersModel` class has been enhanced with:

- **Multimodal Support**: Added `supports_text` and `supports_images` flags
- **Unified Encode Method**: Single `encode()` method handles both text and images
- **Input Type Detection**: Automatic detection based on file extensions and path existence
- **Image Loading**: PIL-based image loading with RGB conversion
- **Error Handling**: Graceful handling of missing or corrupted images

### Configuration

CLIP models are configured in `config/models/embedding_models.yaml`:

```yaml
sentence_transformers:
  sentence-transformers-clip-ViT-B-32:
    type: "sentence_transformers"
    model_name: "sentence-transformers/clip-ViT-B-32"
    dimension: 512
    description: "CLIP model for joint text-image embeddings"
    max_seq_length: 77
    supports_text: true
    supports_images: true
```

### Method Signatures

```python
# Unified encoding interface
def encode(self, inputs: Union[str, List[str]], input_type: str = "auto", **kwargs) -> np.ndarray

# Specific encoding methods
def encode_text(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray
def encode_images(self, image_paths: Union[str, List[str]], **kwargs) -> np.ndarray

# Auto-detection
def _detect_input_type(self, inputs: Union[str, List[str]]) -> str
```

## Dependencies

The CLIP integration requires:

- `sentence-transformers` - For CLIP model loading and inference
- `PIL` (Pillow) - For image loading and preprocessing
- `torch` - PyTorch backend (automatically installed with sentence-transformers)

Install with:
```bash
pip install sentence-transformers Pillow
```

## Performance Considerations

- **Model Loading**: CLIP models are larger (~600MB) and take longer to load initially
- **GPU Acceleration**: Automatically uses GPU if available for faster inference
- **Batch Processing**: Use batch encoding for better performance with multiple inputs
- **Memory Usage**: CLIP models require more memory than text-only models

## Comparison with Existing CLIP Implementation

The repository already had a CLIP implementation using Hugging Face transformers (`CLIPModel` in `image_models.py`). This new integration provides:

| Feature | HuggingFace CLIP | sentence-transformers CLIP |
|---------|------------------|----------------------------|
| **API Simplicity** | More complex setup | Simple, unified API |
| **Model Loading** | Manual tokenizer/processor | Automatic handling |
| **Preprocessing** | Manual image preprocessing | Built-in preprocessing |
| **Batch Processing** | Manual batching logic | Automatic batching |
| **Integration** | Separate model type | Extends existing SentenceTransformers |

Both implementations are available and can be used based on your specific needs.

## Testing

Run the provided test scripts to verify the integration:

```bash
# Basic functionality test
python examples/test_clip_integration.py

# Comprehensive example
python examples/clip_example.py
```

## Future Enhancements

Potential improvements for the CLIP integration:

1. **Caching**: Add embedding caching for frequently used images
2. **Image Preprocessing**: Advanced image augmentation and preprocessing options
3. **Multi-GPU**: Support for distributed inference across multiple GPUs
4. **Streaming**: Support for streaming large batches of images
5. **Custom Models**: Easy integration of fine-tuned CLIP models

## Troubleshooting

### Common Issues

1. **Model Download**: First-time usage downloads ~600MB model files
2. **PIL Import Error**: Install Pillow with `pip install Pillow`
3. **CUDA Memory**: Reduce batch size if running out of GPU memory
4. **Image Loading**: Ensure image files exist and are in supported formats

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- WebP (.webp)
- BMP (.bmp)
- TIFF (.tiff)
- GIF (.gif)

## Contributing

When adding new CLIP models:

1. Add model configuration to `embedding_models.yaml`
2. Set `supports_text: true` and `supports_images: true`
3. Specify correct dimension and max_seq_length
4. Test with both text and image inputs
5. Update documentation

---

This integration brings powerful multimodal capabilities to your search AI system while maintaining compatibility with existing code and infrastructure.
