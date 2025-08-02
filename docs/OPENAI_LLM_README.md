# OpenAI LLM Wrapper

This document describes the OpenAI LLM wrapper classes that provide a clean, configuration-driven interface for working with OpenAI's language models.

## Overview

The OpenAI LLM wrapper consists of several components:

- **BaseLLMModel**: Abstract base class for LLM implementations
- **OpenAILLMModel**: OpenAI-specific implementation with SSL configuration
- **OpenAIPromptProcessor**: High-level interface for common prompt operations
- **ModelFactory**: Factory class for creating and managing LLM models
- **Prompt Templates**: Organized prompt templates for different tasks

## Quick Start

### Basic Usage

```python
from src.models.model_factory import ModelFactory

# Create a prompt processor (recommended for most use cases)
processor = ModelFactory.create_prompt_processor()

# Judge relevance between query and item
result = processor.judge_relevance("pizza margherita", "Pizza with tomato and mozzarella")
print(result)  # {"reason": "...", "score": 2}

# Rewrite Portuguese query to English
result = processor.rewrite_query("pizza de calabresa")
print(result)  # {"thoughts": "...", "query": "pepperoni pizza"}
```

### Direct LLM Model Usage

```python
# Create LLM model directly
llm_model = ModelFactory.create_llm_model("openai", "gpt-4.1-mini")

# Generate text
response = llm_model.generate("Hello, how are you?")
print(response)

# Generate JSON response
json_result = llm_model.generate_json("List 3 colors in JSON format")
print(json_result)  # {"colors": ["red", "green", "blue"]}
```

## Configuration

### LLM Model Configuration

Models are configured in `config/models/llm_models.yaml`:

```yaml
models:
  openai:
    gpt-4.1-mini:
      model_name: "gpt-4.1-mini"
      base_url: "https://pd67dqn1bd.execute-api.eu-west-1.amazonaws.com"
      temperature: 0.0
      max_tokens: null
      service_tier: "auto"
      verify_ssl: false
      description: "GPT-4.1 Mini model for text generation"

default:
  type: "openai"
  model_name: "gpt-4.1-mini"
```

### Custom Configuration

```python
# Override configuration at runtime
custom_config = {
    "temperature": 0.7,
    "max_tokens": 100
}

llm_model = ModelFactory.create_llm_model(
    "openai", 
    "gpt-4.1-mini", 
    config_override=custom_config
)
```

## Available Methods

### OpenAIPromptProcessor Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `judge_relevance()` | Judge relevance between query and item | `query: str, item: str` | `{"reason": str, "score": int}` |
| `rewrite_query()` | Rewrite Portuguese query to English | `query: str` | `{"thoughts": str, "query": str}` |
| `extend_query()` | Extend query with product names and categories | `query: str` | `{"category": str, "names": list}` |
| `extend_query_english()` | Extend Portuguese query with English names | `query: str` | `{"names": list}` |
| `generate_query_for_item()` | Generate query that matches given item | `queries: list, item: str` | `{"query": str}` |
| `generate_query_from_examples()` | Generate query based on examples | `queries: list` | `{"query": str}` |

### OpenAILLMModel Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `generate()` | Generate text from prompt | `prompt: str, **kwargs` | `str` |
| `generate_json()` | Generate JSON response from prompt | `prompt: str, **kwargs` | `dict` |
| `load_model()` | Initialize the OpenAI client | None | None |

### ModelFactory Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `create_llm_model()` | Create specific LLM model | `model_type: str, model_name: str, config_override: dict` | `BaseLLMModel` |
| `create_default_llm_model()` | Create default LLM model | `config_override: dict` | `BaseLLMModel` |
| `create_prompt_processor()` | Create prompt processor | `model_type: str, model_name: str, config_override: dict` | `OpenAIPromptProcessor` |
| `list_configured_llm_models()` | List available LLM models | None | `dict` |

## Prompt Templates

Prompt templates are organized in separate modules:

- `src/models/prompts/judge_prompts.py` - Relevance judging prompts
- `src/models/prompts/query_prompts.py` - Query processing prompts
- `src/models/prompts/generation_prompts.py` - Query generation prompts

### Adding Custom Prompts

1. Add your prompt to the appropriate module:

```python
# src/models/prompts/custom_prompts.py
CUSTOM_PROMPT = """
Your custom prompt template here.
Use {variable} for placeholders.
"""
```

2. Import and use in your processor:

```python
from src.models.prompts.custom_prompts import CUSTOM_PROMPT

class CustomPromptProcessor(OpenAIPromptProcessor):
    def custom_method(self, variable: str) -> dict:
        prompt = CUSTOM_PROMPT.format(variable=variable)
        return self.model.generate_json(prompt)
```

## Migration from Original Code

### Before (Original Code)

```python
import ssl
from openai import OpenAI

# Manual SSL setup
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Manual client setup
client = OpenAI(base_url='https://pd67dqn1bd.execute-api.eu-west-1.amazonaws.com')

# Manual function
def call_judge(query, item):
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": judge_prompt.format(query=query, item=item)}],
        service_tier="auto",
    )
    try:
        return json.loads(completion.choices[0].message.content.strip().strip('```json').strip('```'))
    except:
        return {}
```

### After (New Class Structure)

```python
from src.models.model_factory import ModelFactory

# Simple one-liner
processor = ModelFactory.create_prompt_processor()

# Clean method call
result = processor.judge_relevance(query, item)
```

## Error Handling

The wrapper includes comprehensive error handling:

- **SSL Configuration**: Automatic SSL setup for custom endpoints
- **JSON Parsing**: Robust JSON parsing with fallback for malformed responses
- **Model Loading**: Clear error messages for configuration issues
- **API Errors**: Proper exception handling for OpenAI API errors

## Features

### ✅ Configuration-Driven
- Models configured in YAML files
- Easy switching between models
- Runtime configuration overrides

### ✅ SSL Support
- Automatic SSL configuration for custom endpoints
- Configurable SSL verification

### ✅ Error Handling
- Robust JSON parsing with fallbacks
- Clear error messages
- Graceful degradation

### ✅ Organized Prompts
- Prompts separated into logical modules
- Easy to maintain and extend
- Consistent formatting

### ✅ Factory Pattern
- Easy model creation and management
- Consistent interfaces
- Extensible architecture

### ✅ Type Safety
- Proper typing throughout
- Abstract base classes
- Clear interfaces

## Examples

See the following files for complete examples:

- `openai_llm_example.py` - Comprehensive usage examples
- `migration_example.py` - Migration guide from original code

## Extending the System

### Adding New LLM Providers

1. Create a new LLM model class:

```python
class AnthropicLLMModel(BaseLLMModel):
    def generate(self, prompt: str, **kwargs) -> str:
        # Implementation for Anthropic API
        pass
    
    def generate_json(self, prompt: str, **kwargs) -> dict:
        # Implementation for JSON generation
        pass
```

2. Register with the factory:

```python
ModelFactory.register_llm_model("anthropic", AnthropicLLMModel)
```

3. Add configuration to `llm_models.yaml`:

```yaml
models:
  anthropic:
    claude-3:
      model_name: "claude-3-sonnet"
      # ... other config
```

### Adding New Prompt Templates

1. Create new prompt module
2. Add prompts as constants
3. Import in `__init__.py`
4. Use in processor methods

This architecture makes it easy to extend the system while maintaining consistency with the existing codebase.
