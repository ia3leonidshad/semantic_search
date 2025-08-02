"""Configuration loading utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from config.settings import CONFIG_DIR, MODELS_CONFIG_DIR

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Utility class for loading YAML configuration files."""
    
    @staticmethod
    def load_yaml(file_path: Path) -> Dict[str, Any]:
        """Load a YAML configuration file.
        
        Args:
            file_path: Path to the YAML file
            
        Returns:
            Dictionary containing the configuration
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            yaml.YAMLError: If the YAML file is malformed
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {file_path}")
                return config or {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {file_path}: {e}")
            raise
    
    @staticmethod
    def load_model_config(model_type: Optional[str] = None) -> Dict[str, Any]:
        """Load embedding model configuration.
        
        Args:
            model_type: Optional specific model type to load
            
        Returns:
            Dictionary containing model configurations
        """
        config_path = MODELS_CONFIG_DIR / "embedding_models.yaml"
        config = ConfigLoader.load_yaml(config_path)
        
        if model_type and model_type in config.get("models", {}):
            return {
                "models": {model_type: config["models"][model_type]},
                "default": config.get("default", {}),
                "settings": config.get("settings", {})
            }
        
        return config
    
    @staticmethod
    def load_llm_config(model_type: Optional[str] = None) -> Dict[str, Any]:
        """Load LLM model configuration.
        
        Args:
            model_type: Optional specific model type to load
            
        Returns:
            Dictionary containing LLM model configurations
        """
        config_path = MODELS_CONFIG_DIR / "llm_models.yaml"
        config = ConfigLoader.load_yaml(config_path)
        
        if model_type and model_type in config.get("models", {}):
            return {
                "models": {model_type: config["models"][model_type]},
                "default": config.get("default", {}),
                "settings": config.get("settings", {})
            }
        
        return config
    
    @staticmethod
    def get_model_config(model_type: str, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model.
        
        Args:
            model_type: Type of model (e.g., 'sentence_transformers')
            model_name: Name of the specific model
            
        Returns:
            Dictionary containing the model configuration
            
        Raises:
            KeyError: If the model configuration is not found
        """
        config = ConfigLoader.load_model_config()
        
        try:
            model_config = config["models"][model_type][model_name]
            # Merge with global settings
            settings = config.get("settings", {})
            return {**model_config, **settings}
        except KeyError:
            available_models = []
            for mtype, models in config.get("models", {}).items():
                for mname in models.keys():
                    available_models.append(f"{mtype}/{mname}")
            
            raise KeyError(
                f"Model '{model_type}/{model_name}' not found in configuration. "
                f"Available models: {', '.join(available_models)}"
            )
    
    @staticmethod
    def get_llm_config(model_type: str, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific LLM model.
        
        Args:
            model_type: Type of LLM model (e.g., 'openai')
            model_name: Name of the specific model
            
        Returns:
            Dictionary containing the LLM model configuration
            
        Raises:
            KeyError: If the model configuration is not found
        """
        config = ConfigLoader.load_llm_config()
        
        try:
            model_config = config["models"][model_type][model_name]
            # Merge with global settings
            settings = config.get("settings", {})
            return {**model_config, **settings}
        except KeyError:
            available_models = []
            for mtype, models in config.get("models", {}).items():
                for mname in models.keys():
                    available_models.append(f"{mtype}/{mname}")
            
            raise KeyError(
                f"LLM Model '{model_type}/{model_name}' not found in configuration. "
                f"Available models: {', '.join(available_models)}"
            )
    
    @staticmethod
    def get_default_model_config() -> Dict[str, Any]:
        """Get the default model configuration.
        
        Returns:
            Dictionary containing the default model configuration
        """
        config = ConfigLoader.load_model_config()
        default = config.get("default", {})
        
        model_type = default.get("model_type", "sentence_transformers")
        model_name = default.get("model_name", "all-MiniLM-L6-v2")
        
        return ConfigLoader.get_model_config(model_type, model_name)
    
    @staticmethod
    def get_default_llm_config() -> Dict[str, Any]:
        """Get the default LLM model configuration.
        
        Returns:
            Dictionary containing the default LLM model configuration
        """
        config = ConfigLoader.load_llm_config()
        default = config.get("default", {})

        model_type = default.get("type", "openai")
        model_name = default.get("model_name", "gpt-4.1-mini")

        # Get the actual model configuration
        model_config = ConfigLoader.get_llm_config(model_type, model_name)
        
        # Return merged configuration with type and model_name fields
        return {
            "type": model_type,
            "model_name": model_name,
            **model_config
        }
