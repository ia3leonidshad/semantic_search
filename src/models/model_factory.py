"""Model factory for creating embedding models dynamically."""

from typing import Dict, Any, Optional
import logging

from src.models.embedding_models import (
    BaseEmbeddingModel,
    SentenceTransformersModel,
    HuggingFaceModel,
    OpenAIModel
)
from src.models.image_models import (
    CLIPModel,
)
from src.models.llm_models import (
    BaseLLMModel,
    OpenAILLMModel,
    OpenAIPromptProcessor
)
from src.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory class for creating embedding and LLM models."""
    
    # Registry of available embedding model types
    _model_registry = {
        "sentence_transformers": SentenceTransformersModel,
        "huggingface": HuggingFaceModel,
        "openai": OpenAIModel,
        "clip": CLIPModel,
    }
    
    # Registry of available LLM model types
    _llm_registry = {
        "openai": OpenAILLMModel,
    }
    
    @classmethod
    def register_model(cls, model_type: str, model_class: type):
        """Register a new model type.
        
        Args:
            model_type: String identifier for the model type
            model_class: Class that implements BaseEmbeddingModel
        """
        if not issubclass(model_class, BaseEmbeddingModel):
            raise ValueError(f"Model class must inherit from BaseEmbeddingModel")
        
        cls._model_registry[model_type] = model_class
        logger.info(f"Registered model type: {model_type}")
    
    @classmethod
    def get_available_models(cls) -> Dict[str, type]:
        """Get all available model types.
        
        Returns:
            Dictionary mapping model type names to their classes
        """
        return cls._model_registry.copy()
    
    @classmethod
    def create_model(
        cls,
        model_type: str,
        model_name: str,
        config_override: Optional[Dict[str, Any]] = None
    ) -> BaseEmbeddingModel:
        """Create an embedding model instance.
        
        Args:
            model_type: Type of model (e.g., 'sentence_transformers')
            model_name: Name of the specific model
            config_override: Optional configuration overrides
            
        Returns:
            Initialized embedding model instance
            
        Raises:
            ValueError: If model type is not registered
            KeyError: If model configuration is not found
        """
        if model_type not in cls._model_registry:
            available_types = list(cls._model_registry.keys())
            raise ValueError(
                f"Unknown model type '{model_type}'. "
                f"Available types: {', '.join(available_types)}"
            )
        
        # Load configuration
        try:
            config = ConfigLoader.get_model_config(model_type, model_name)
        except KeyError as e:
            logger.error(f"Failed to load model configuration: {e}")
            raise
        
        # Apply configuration overrides
        if config_override:
            config.update(config_override)
            logger.info(f"Applied configuration overrides: {config_override}")
        
        # Create model instance
        model_class = cls._model_registry[model_type]
        logger.info(f"Creating {model_type} model: {model_name}")
        
        try:
            return model_class(config)
        except Exception as e:
            logger.error(f"Failed to create model {model_type}/{model_name}: {e}")
            raise
    
    @classmethod
    def create_default_model(
        cls,
        config_override: Optional[Dict[str, Any]] = None
    ) -> BaseEmbeddingModel:
        """Create the default embedding model.
        
        Args:
            config_override: Optional configuration overrides
            
        Returns:
            Initialized default embedding model instance
        """
        config = ConfigLoader.get_default_model_config()
        
        # Apply configuration overrides
        if config_override:
            config.update(config_override)
        
        model_type = config.get("type")
        model_name = config.get("model_name")
        
        if not model_type or not model_name:
            raise ValueError("Default model configuration is incomplete")
        
        logger.info(f"Creating default model: {model_type}/{model_name}")
        return cls.create_model(model_type, model_name, config_override)
    
    @classmethod
    def list_configured_models(cls) -> Dict[str, Dict[str, Any]]:
        """List all models configured in the configuration files.
        
        Returns:
            Dictionary of all configured models with their metadata
        """
        config = ConfigLoader.load_model_config()
        models_info = {}
        
        for model_type, models in config.get("models", {}).items():
            for model_name, model_config in models.items():
                key = f"{model_type}/{model_name}"
                models_info[key] = {
                    "type": model_type,
                    "name": model_name,
                    "dimension": model_config.get("dimension"),
                    "description": model_config.get("description", ""),
                    "max_seq_length": model_config.get("max_seq_length"),
                    "requires_api_key": model_config.get("requires_api_key", False)
                }
        
        return models_info
    
    # LLM Model Methods
    
    @classmethod
    def register_llm_model(cls, model_type: str, model_class: type):
        """Register a new LLM model type.
        
        Args:
            model_type: String identifier for the LLM model type
            model_class: Class that implements BaseLLMModel
        """
        if not issubclass(model_class, BaseLLMModel):
            raise ValueError(f"Model class must inherit from BaseLLMModel")
        
        cls._llm_registry[model_type] = model_class
        logger.info(f"Registered LLM model type: {model_type}")
    
    @classmethod
    def get_available_llm_models(cls) -> Dict[str, type]:
        """Get all available LLM model types.
        
        Returns:
            Dictionary mapping LLM model type names to their classes
        """
        return cls._llm_registry.copy()
    
    @classmethod
    def create_llm_model(
        cls,
        model_type: str,
        model_name: str,
        config_override: Optional[Dict[str, Any]] = None
    ) -> BaseLLMModel:
        """Create an LLM model instance.
        
        Args:
            model_type: Type of LLM model (e.g., 'openai')
            model_name: Name of the specific model
            config_override: Optional configuration overrides
            
        Returns:
            Initialized LLM model instance
            
        Raises:
            ValueError: If model type is not registered
            KeyError: If model configuration is not found
        """
        if model_type not in cls._llm_registry:
            available_types = list(cls._llm_registry.keys())
            raise ValueError(
                f"Unknown LLM model type '{model_type}'. "
                f"Available types: {', '.join(available_types)}"
            )
        
        # Load configuration
        try:
            config = ConfigLoader.get_llm_config(model_type, model_name)
        except KeyError as e:
            logger.error(f"Failed to load LLM model configuration: {e}")
            raise
        
        # Apply configuration overrides
        if config_override:
            config.update(config_override)
            logger.info(f"Applied LLM configuration overrides: {config_override}")
        
        # Create model instance
        model_class = cls._llm_registry[model_type]
        logger.info(f"Creating {model_type} LLM model: {model_name}")
        
        try:
            return model_class(config)
        except Exception as e:
            logger.error(f"Failed to create LLM model {model_type}/{model_name}: {e}")
            raise
    
    @classmethod
    def create_default_llm_model(
        cls,
        config_override: Optional[Dict[str, Any]] = None
    ) -> BaseLLMModel:
        """Create the default LLM model.
        
        Args:
            config_override: Optional configuration overrides
            
        Returns:
            Initialized default LLM model instance
        """
        config = ConfigLoader.get_default_llm_config()

        # Apply configuration overrides
        if config_override:
            config.update(config_override)

        model_type = config.get("type")
        model_name = config.get("model_name")
        
        if not model_type or not model_name:
            raise ValueError("Default LLM model configuration is incomplete")
        
        logger.info(f"Creating default LLM model: {model_type}/{model_name}")
        return cls.create_llm_model(model_type, model_name, config_override)
    
    @classmethod
    def create_prompt_processor(
        cls,
        model_type: str = None,
        model_name: str = None,
        config_override: Optional[Dict[str, Any]] = None
    ) -> OpenAIPromptProcessor:
        """Create a prompt processor with an LLM model.
        
        Args:
            model_type: Type of LLM model (defaults to default config)
            model_name: Name of the specific model (defaults to default config)
            config_override: Optional configuration overrides
            
        Returns:
            Initialized OpenAIPromptProcessor instance
        """
        if model_type and model_name:
            llm_model = cls.create_llm_model(model_type, model_name, config_override)
        else:
            llm_model = cls.create_default_llm_model(config_override)
        
        if not isinstance(llm_model, OpenAILLMModel):
            raise ValueError("Prompt processor currently only supports OpenAI models")
        
        return OpenAIPromptProcessor(llm_model)
    
    @classmethod
    def list_configured_llm_models(cls) -> Dict[str, Dict[str, Any]]:
        """List all LLM models configured in the configuration files.
        
        Returns:
            Dictionary of all configured LLM models with their metadata
        """
        config = ConfigLoader.load_llm_config()
        models_info = {}
        
        for model_type, models in config.get("models", {}).items():
            for model_name, model_config in models.items():
                key = f"{model_type}/{model_name}"
                models_info[key] = {
                    "type": model_type,
                    "name": model_name,
                    "base_url": model_config.get("base_url"),
                    "description": model_config.get("description", ""),
                    "temperature": model_config.get("temperature", 0.0),
                    "max_tokens": model_config.get("max_tokens"),
                    "verify_ssl": model_config.get("verify_ssl", False)
                }
        
        return models_info
