"""LLM model implementations for text generation and completion."""

import json
import ssl
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class BaseLLMModel(ABC):
    """Abstract base class for LLM models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the LLM model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model_name = config.get("model_name")
        self.base_url = config.get("base_url")
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", None)
        self.service_tier = config.get("service_tier", "auto")
        
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def generate_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate JSON response from prompt.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional generation parameters
            
        Returns:
            Parsed JSON response as dictionary
        """
        pass
    
    @abstractmethod
    def load_model(self):
        """Load/initialize the LLM model."""
        pass


class OpenAILLMModel(BaseLLMModel):
    """OpenAI LLM model implementation with custom SSL configuration."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = None
        self.verify_ssl = config.get("verify_ssl", False)
        self._configure_ssl()
        self.load_model()
    
    def _configure_ssl(self):
        """Configure SSL settings for OpenAI client."""
        if not self.verify_ssl:
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                # Legacy Python that doesn't verify HTTPS certificates by default
                pass
            else:
                # Handle target environment that doesn't support HTTPS verification
                ssl._create_default_https_context = _create_unverified_https_context
            
            # Set OpenAI SSL verification
            try:
                import openai
                openai.verify_ssl_certs = False
            except ImportError:
                logger.warning("OpenAI package not found. SSL configuration may not be applied.")
    
    def load_model(self):
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI
            
            logger.info(f"Initializing OpenAI LLM model: {self.model_name}")
            
            # Initialize client with base URL
            client_kwargs = {}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            
            self.client = OpenAI(**client_kwargs)
            
        except ImportError:
            raise ImportError(
                "openai is required for OpenAILLMModel. "
                "Install it with: pip install openai"
            )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt using OpenAI chat completions.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated text response
        """
        if not self.client:
            raise RuntimeError("OpenAI client not initialized. Call load_model() first.")
        
        # Merge config defaults with kwargs
        generation_kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "service_tier": self.service_tier,
            "temperature": kwargs.get("temperature", self.temperature),
        }
        
        # Add max_tokens if specified
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens:
            generation_kwargs["max_tokens"] = max_tokens
        
        try:
            completion = self.client.chat.completions.create(**generation_kwargs)
            return completion.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    def generate_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate JSON response from prompt.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional generation parameters
            
        Returns:
            Parsed JSON response as dictionary
        """
        response_text = self.generate(prompt, **kwargs)
        
        try:
            # Clean response text (remove markdown code blocks if present)
            cleaned_text = response_text.strip()
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]  # Remove ```json
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]  # Remove ```
            cleaned_text = cleaned_text.strip()
            
            return json.loads(cleaned_text)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {response_text}")
            return {}


class OpenAIPromptProcessor:
    """High-level processor for OpenAI prompts with predefined templates."""
    
    def __init__(self, model: OpenAILLMModel):
        """Initialize with an OpenAI LLM model.
        
        Args:
            model: Initialized OpenAILLMModel instance
        """
        self.model = model
    
    def judge_relevance(self, query: str, item: str) -> Dict[str, Any]:
        """Judge relevance between query and item.
        
        Args:
            query: User search query
            item: Item information to evaluate
            
        Returns:
            Dictionary with 'reason' and 'score' keys
        """
        from src.models.prompts.judge_prompts import JUDGE_PROMPT
        
        prompt = JUDGE_PROMPT.format(query=query, item=item)
        return self.model.generate_json(prompt)
    
    def rewrite_query(self, query: str) -> Dict[str, Any]:
        """Rewrite query from Portuguese to English.
        
        Args:
            query: Portuguese query to rewrite
            
        Returns:
            Dictionary with 'thoughts' and 'query' keys
        """
        from src.models.prompts.query_prompts import QUERY_REWRITE_PROMPT
        
        prompt = QUERY_REWRITE_PROMPT.format(query=query)
        return self.model.generate_json(prompt)
    
    def extend_query(self, query: str) -> Dict[str, Any]:
        """Extend query with possible product names and categories.
        
        Args:
            query: User search query
            
        Returns:
            Dictionary with 'category' and 'names' keys
        """
        from src.models.prompts.query_prompts import QUERY_EXTEND_PROMPT
        
        prompt = QUERY_EXTEND_PROMPT.format(query=query)
        return self.model.generate_json(prompt)
    
    def extend_query_english(self, query: str) -> Dict[str, Any]:
        """Extend Portuguese query with English product names.
        
        Args:
            query: Portuguese search query
            
        Returns:
            Dictionary with 'names' key containing list of English names
        """
        from src.models.prompts.query_prompts import QUERY_EXTEND_ENGLISH_PROMPT
        
        prompt = QUERY_EXTEND_ENGLISH_PROMPT.format(query=query)
        return self.model.generate_json(prompt)
    
    def generate_query_for_item(self, queries: list, item: str) -> Dict[str, Any]:
        """Generate a query that would match the given item.
        
        Args:
            queries: List of example queries for style reference
            item: Item information to generate query for
            
        Returns:
            Dictionary with 'query' key
        """
        from src.models.prompts.generation_prompts import QUERY_GENERATION_DATASET_PROMPT
        
        prompt = QUERY_GENERATION_DATASET_PROMPT.format(
            queries='\n'.join(queries), 
            item=item
        )
        return self.model.generate_json(prompt)
    
    def generate_query_from_examples(self, queries: list) -> Dict[str, Any]:
        """Generate a query based on example queries.
        
        Args:
            queries: List of example queries for style reference
            
        Returns:
            Dictionary with 'query' key
        """
        from src.models.prompts.generation_prompts import QUERY_GENERATION_DATASET_2_PROMPT
        
        prompt = QUERY_GENERATION_DATASET_2_PROMPT.format(queries='\n'.join(queries))
        return self.model.generate_json(prompt)
