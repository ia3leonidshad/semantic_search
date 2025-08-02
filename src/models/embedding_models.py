"""Embedding model implementations."""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the embedding model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model_name = config.get("model_name")
        self.image_model_name = config.get("image_model_name")
        self.dimension = config.get("dimension")
        self.max_seq_length = config.get("max_seq_length", 512)
        self.normalize_embeddings = config.get("normalize_embeddings", True)
        self.batch_size = config.get("batch_size", 32)
        self.device = config.get("device", 'cpu')
        self.base_url = config.get('base_url', 'https://pd67dqn1bd.execute-api.eu-west-1.amazonaws.com')
        self.query_translator = config.get("query_translator", None)
        
    @abstractmethod
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode texts into embeddings.
        
        Args:
            texts: Single text or list of texts to encode
            **kwargs: Additional arguments for encoding
            
        Returns:
            Numpy array of embeddings
        """
        pass
    
    @abstractmethod
    def load_model(self):
        """Load the embedding model."""
        pass
    
    def _apply_query_translation(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        """Apply query translation if translator is available.
        
        Args:
            texts: Single text or list of texts to translate
            
        Returns:
            Translated texts (same format as input)
        """
        if self.query_translator is None:
            return texts
        
        try:
            if isinstance(texts, str):
                return self.query_translator(texts)
            else:
                return [self.query_translator(text) for text in texts]
        except Exception as e:
            logger.warning(f"Query translation failed: {e}. Using original texts.")
            return texts
    
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit vectors.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Normalized embeddings
        """
        if self.normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            return embeddings / norms
        return embeddings


class SentenceTransformersModel(BaseEmbeddingModel):
    """Sentence Transformers embedding model implementation with multimodal support."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.image_model = None
        self.text_model = None
        self.supports_text = config.get("supports_text", True)
        self.supports_images = config.get("supports_images", False)
        self.load_model()
    
    def load_model(self):
        """Load the Sentence Transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformers model: {self.model_name}, {self.image_model_name}, {self.device}")
            self.text_model = SentenceTransformer(self.model_name, device=self.device, trust_remote_code=True)
            if self.image_model_name:
                if self.image_model_name == self.model_name:
                    self.image_model = self.text_model
                else:
                    self.image_model = SentenceTransformer(self.image_model_name, device=self.device, trust_remote_code=True)
            # Update dimension if not specified in config
            if not self.dimension:
                self.dimension = self.text_model.get_sentence_embedding_dimension()
            
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformersModel. "
                "Install it with: pip install sentence-transformers"
            )
    
    def encode(self, inputs: Union[str, List[str]], input_type: str = "auto", **kwargs) -> np.ndarray:
        """Encode texts or images using Sentence Transformers.
        
        Args:
            inputs: Single text/image path or list of texts/image paths to encode
            input_type: "text", "image", or "auto" for auto-detection
            **kwargs: Additional arguments (show_progress_bar, etc.)
            
        Returns:
            Numpy array of embeddings
        """
        if input_type == "auto":
            input_type = self._detect_input_type(inputs)
        
        if input_type == "text":
            if not self.supports_text:
                raise ValueError(f"Model {self.model_name} does not support text encoding")
            return self.encode_text(inputs, **kwargs)
        elif input_type == "image":
            if not self.supports_images:
                raise ValueError(f"Model {self.image_model_name} does not support image encoding")
            return self.encode_images(inputs, **kwargs)
        else:
            raise ValueError(f"Unsupported input_type: {input_type}")
    
    def encode_text(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode texts using Sentence Transformers.
        
        Args:
            texts: Single text or list of texts to encode
            **kwargs: Additional arguments (show_progress_bar, etc.)
            
        Returns:
            Numpy array of embeddings
        """
        assert self.text_model
        if isinstance(texts, str):
            texts = [texts]
        
        # Apply query translation if available
        texts = self._apply_query_translation(texts)
        
        # Set default kwargs
        encode_kwargs = {
            "batch_size": self.batch_size,
            "show_progress_bar": len(texts) > 100,
            "convert_to_numpy": True,
            # "normalize_embeddings": True,
            **kwargs
        }
        
        embeddings = self.text_model.encode(texts, **encode_kwargs)
        return self._normalize(embeddings)
    
    def encode_images(self, image_paths: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode images using Sentence Transformers CLIP.
        
        Args:
            image_paths: Single image path or list of image paths
            **kwargs: Additional arguments
            
        Returns:
            Numpy array of image embeddings
        """
        assert self.image_model
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        
        # Load images
        images = []
        for path in image_paths:
            try:
                image = self._load_image(path)
                images.append(image)
            except Exception as e:
                logger.warning(f"Failed to load image {path}: {e}")
                # Skip failed images
                continue
        
        if not images:
            # Return empty array with correct shape
            return np.empty((0, self.dimension))
        
        # Set default kwargs
        encode_kwargs = {
            "batch_size": self.batch_size,
            "show_progress_bar": len(images) > 100,
            "convert_to_numpy": True,
            # "normalize_embeddings": True,
            **kwargs
        }
        
        embeddings = self.image_model.encode(images, **encode_kwargs)
        return self._normalize(embeddings)
    
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
            if path.suffix.lower() in image_extensions and path.exists():
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


class HuggingFaceModel(BaseEmbeddingModel):
    """HuggingFace transformers embedding model implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.prefix = config.get("prefix", "")
        self.load_model()
    
    def load_model(self):
        """Load the HuggingFace model and tokenizer."""
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            logger.info(f"Loading HuggingFace model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Set device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            
        except ImportError:
            raise ImportError(
                "transformers and torch are required for HuggingFaceModel. "
                "Install them with: pip install transformers torch"
            )
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode texts using HuggingFace transformers.
        
        Args:
            texts: Single text or list of texts to encode
            **kwargs: Additional arguments
            
        Returns:
            Numpy array of embeddings
        """
        import torch
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Apply query translation if available
        texts = self._apply_query_translation(texts)
        
        # Add prefix if specified
        if self.prefix:
            texts = [self.prefix + text for text in texts]
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling
                batch_embeddings = self._mean_pooling(outputs, inputs["attention_mask"])
                embeddings.append(batch_embeddings.cpu().numpy())
        
        embeddings = np.vstack(embeddings)
        return self._normalize(embeddings)
    
    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        import torch
        
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class OpenAIModel(BaseEmbeddingModel):
    """OpenAI embedding model implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = None
        self.load_model()
    
    def load_model(self):
        """Initialize the OpenAI client."""
        try:
            import openai
            
            logger.info(f"Initializing OpenAI model: {self.model_name}")
            self.client = openai.OpenAI(base_url=self.base_url)  # Uses OPENAI_API_KEY env var
            
        except ImportError:
            raise ImportError(
                "openai is required for OpenAIModel. "
                "Install it with: pip install openai"
            )
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode texts using OpenAI embeddings API.
        
        Args:
            texts: Single text or list of texts to encode
            **kwargs: Additional arguments
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Apply query translation if available
        texts = self._apply_query_translation(texts)
        
        embeddings = []
        
        # Process in batches (OpenAI has batch limits)
        batch_size = min(self.batch_size, 100)  # OpenAI limit
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch_texts
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        
        embeddings = np.float32(embeddings)
        return self._normalize(embeddings)
