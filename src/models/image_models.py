"""Image embedding model implementations."""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict, Any
import logging
from pathlib import Path

from src.models.embedding_models import BaseEmbeddingModel

logger = logging.getLogger(__name__)


class BaseImageEmbeddingModel(BaseEmbeddingModel):
    """Abstract base class for image embedding models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.supports_text = config.get("supports_text", False)
        self.supports_images = config.get("supports_images", True)
    
    @abstractmethod
    def encode_images(self, image_paths: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode images into embeddings.
        
        Args:
            image_paths: Single image path or list of image paths
            **kwargs: Additional arguments for encoding
            
        Returns:
            Numpy array of embeddings
        """
        pass
    
    def encode(self, inputs: Union[str, List[str]], input_type: str = "auto", **kwargs) -> np.ndarray:
        """Unified encoding interface.
        
        Args:
            inputs: Text strings or image paths
            input_type: "text", "image", or "auto" for auto-detection
            **kwargs: Additional arguments
            
        Returns:
            Numpy array of embeddings
        """
        if input_type == "auto":
            input_type = self._detect_input_type(inputs)
        
        if input_type == "text":
            if not self.supports_text:
                raise ValueError(f"Model {self.model_name} does not support text encoding")
            # Apply query translation if available
            inputs = self._apply_query_translation(inputs)
            return self.encode_text(inputs, **kwargs)
        elif input_type == "image":
            if not self.supports_images:
                raise ValueError(f"Model {self.model_name} does not support image encoding")
            return self.encode_images(inputs, **kwargs)
        else:
            raise ValueError(f"Unsupported input_type: {input_type}")
    
    def encode_text(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode text into embeddings. Override in subclasses that support text.
        
        Args:
            texts: Single text or list of texts
            **kwargs: Additional arguments
            
        Returns:
            Numpy array of embeddings
        """
        raise NotImplementedError("Text encoding not supported by this model")
    
    def _detect_input_type(self, inputs: Union[str, List[str]]) -> str:
        """Auto-detect whether inputs are text or image paths.
        
        Args:
            inputs: Input data
            
        Returns:
            "text" or "image"
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        
        # Check if inputs look like file paths
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.gif'}
        
        for inp in inputs[:5]:  # Check first 5 inputs
            path = Path(inp)
            if path.suffix.lower() in image_extensions or path.exists():
                return "image"
        
        return "text"
    
    def _load_image(self, image_path: str):
        """Load and preprocess an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image
        """
        try:
            from PIL import Image
            
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except ImportError:
            raise ImportError(
                "PIL is required for image processing. "
                "Install it with: pip install Pillow"
            )


class CLIPModel(BaseImageEmbeddingModel):
    """CLIP model for joint text-image embeddings."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.processor = None
        self.supports_text = True
        self.supports_images = True
        self.load_model()
    
    def load_model(self):
        """Load the CLIP model and processor."""
        try:
            from transformers import CLIPModel as HFCLIPModel, CLIPProcessor
            import torch
            
            logger.info(f"Loading CLIP model: {self.model_name}")
            
            # Try loading with from_tf=True if PyTorch weights not available
            try:
                self.model = HFCLIPModel.from_pretrained(self.model_name)
            except OSError as e:
                if "pytorch_model.bin" in str(e) and "TensorFlow" in str(e):
                    logger.info(f"Loading CLIP model from TensorFlow weights: {self.model_name}")
                    self.model = HFCLIPModel.from_pretrained(self.model_name, from_tf=True)
                else:
                    raise
            
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            
            # Set device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            elif self.device is None:
                self.device = "cpu"
            
            self.model.to(self.device)
            
            # Update dimension if not specified
            if not self.dimension:
                self.dimension = self.model.config.projection_dim
                
        except ImportError:
            raise ImportError(
                "transformers and torch are required for CLIPModel. "
                "Install them with: pip install transformers torch"
            )
    
    def encode_text(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode texts using CLIP text encoder.
        
        Args:
            texts: Single text or list of texts
            **kwargs: Additional arguments
            
        Returns:
            Numpy array of text embeddings
        """
        import torch
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize
            inputs = self.processor(
                text=batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                # Normalize embeddings
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                embeddings.append(text_features.cpu().numpy())
        
        embeddings = np.vstack(embeddings)
        return self._normalize(embeddings)
    
    def encode_images(self, image_paths: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode images using CLIP image encoder.
        
        Args:
            image_paths: Single image path or list of image paths
            **kwargs: Additional arguments
            
        Returns:
            Numpy array of image embeddings
        """
        import torch
        
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            
            # Load images
            images = []
            for path in batch_paths:
                try:
                    image = self._load_image(path)
                    images.append(image)
                except Exception as e:
                    logger.warning(f"Failed to load image {path}: {e}")
                    # Use a blank image as placeholder
                    from PIL import Image
                    images.append(Image.new('RGB', (224, 224), color='white'))
            
            if not images:
                continue
            
            # Process images
            inputs = self.processor(
                images=images,
                return_tensors="pt"
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Normalize embeddings
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embeddings.append(image_features.cpu().numpy())
            print(i)
        if not embeddings:
            # Return empty array with correct shape
            return np.empty((0, self.dimension))
        
        embeddings = np.vstack(embeddings)
        return self._normalize(embeddings)
