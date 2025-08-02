"""Example usage of the OpenAI LLM wrapper classes."""

import json
import logging
from src.models.model_factory import ModelFactory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate the OpenAI LLM wrapper functionality."""
    
    print("=== OpenAI LLM Wrapper Example ===\n")
    
    # Example 1: Create LLM model directly
    print("1. Creating OpenAI LLM model directly...")
    try:
        llm_model = ModelFactory.create_llm_model("openai", "gpt-4.1-mini")
        print(f"✓ Created LLM model: {llm_model.model_name}")
        
        # Test basic generation
        response = llm_model.generate("Hello, how are you?")
        print(f"Response: {response}\n")
        
    except Exception as e:
        print(f"✗ Failed to create LLM model: {e}\n")
    
    # Example 2: Create default LLM model
    print("2. Creating default LLM model...")
    try:
        default_llm = ModelFactory.create_default_llm_model()
        print(f"✓ Created default LLM model: {default_llm.model_name}\n")
        
    except Exception as e:
        print(f"✗ Failed to create default LLM model: {e}\n")
    
    # Example 3: Create prompt processor (high-level interface)
    print("3. Creating prompt processor...")
    try:
        processor = ModelFactory.create_prompt_processor()
        print(f"✓ Created prompt processor with model: {processor.model.model_name}")
        
        # Test relevance judging
        print("\n--- Testing relevance judging ---")
        query = "pizza margherita"
        item = "Pizza Margherita - Traditional Italian pizza with tomato sauce, mozzarella cheese, and fresh basil"
        
        result = processor.judge_relevance(query, item)
        print(f"Query: {query}")
        print(f"Item: {item}")
        print(f"Judgment: {json.dumps(result, indent=2)}")
        
        # Test query rewriting
        print("\n--- Testing query rewriting ---")
        portuguese_query = "pizza de calabresa"
        
        result = processor.rewrite_query(portuguese_query)
        print(f"Portuguese query: {portuguese_query}")
        print(f"Rewrite result: {json.dumps(result, indent=2)}")
        
        # Test query extension
        print("\n--- Testing query extension ---")
        simple_query = "hamburguer"
        
        result = processor.extend_query(simple_query)
        print(f"Simple query: {simple_query}")
        print(f"Extension result: {json.dumps(result, indent=2)}")
        
        # Test English query extension
        print("\n--- Testing English query extension ---")
        result = processor.extend_query_english(simple_query)
        print(f"Portuguese query: {simple_query}")
        print(f"English extension: {json.dumps(result, indent=2)}")
        
        # Test query generation
        print("\n--- Testing query generation ---")
        example_queries = [
            "pizza de calabresa",
            "hamburguer artesanal",
            "sushi salmão",
            "açaí com granola"
        ]
        item_info = "Hamburguer Bacon - Delicious beef burger with crispy bacon, lettuce, tomato and special sauce"
        
        result = processor.generate_query_for_item(example_queries, item_info)
        print(f"Example queries: {example_queries}")
        print(f"Item: {item_info}")
        print(f"Generated query: {json.dumps(result, indent=2)}")
        
    except Exception as e:
        print(f"✗ Failed to create or use prompt processor: {e}")
        import traceback
        traceback.print_exc()
    
    # Example 4: List available LLM models
    print("\n4. Listing available LLM models...")
    try:
        llm_models = ModelFactory.list_configured_llm_models()
        print("Available LLM models:")
        for model_key, model_info in llm_models.items():
            print(f"  - {model_key}: {model_info['description']}")
            print(f"    Base URL: {model_info['base_url']}")
            print(f"    Temperature: {model_info['temperature']}")
            print()
            
    except Exception as e:
        print(f"✗ Failed to list LLM models: {e}")
    
    # Example 5: Custom configuration override
    print("5. Using custom configuration...")
    try:
        custom_config = {
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        custom_llm = ModelFactory.create_llm_model(
            "openai", 
            "gpt-4.1-mini", 
            config_override=custom_config
        )
        
        print(f"✓ Created custom LLM with temperature: {custom_llm.temperature}")
        
        response = custom_llm.generate("Write a creative short story about a robot chef.")
        print(f"Creative response: {response}")
        
    except Exception as e:
        print(f"✗ Failed to create custom LLM: {e}")


if __name__ == "__main__":
    main()
