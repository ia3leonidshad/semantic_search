"""Migration example showing how to convert original OpenAI code to use the new class structure."""

import json
from src.models.model_factory import ModelFactory

# Original functions (your old code style)
def old_style_usage():
    """Example of how your original code worked."""
    print("=== Original Code Style ===")
    
    # This is how you had to do it before (manually setting up everything)
    import ssl
    from openai import OpenAI
    
    # SSL configuration
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Manual client setup
    client = OpenAI(base_url='https://pd67dqn1bd.execute-api.eu-west-1.amazonaws.com')
    
    # Manual prompt formatting and calling
    judge_prompt = """
Your task is to judge the retrieval system of food/groceries delivery service.
You'll be presented the user query and retrieved item.
You need evaluate how well item matches the query and assign on of 3 scores:
2 = Highly relevant (exact dish match)
1 = Partially relevant (shares key attributes, could satisfy)
0 = Irrelevant

User query:
{query}

Item info:
{item}

Reply in English.

Reply in the following json format:
{{
    "reason": string, // 3-5 sentences reasoning about your judgement, what user asked for, what they got, how well it matches the intent
    "score": int, // relevance score, one of {{0, 1, 2}}
}}
"""
    
    def call_judge(query, item):
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": judge_prompt.format(query=query, item=item),
                },        
            ],
            service_tier="auto",
        )
        try:
            return json.loads(completion.choices[0].message.content.strip().strip('```json').strip('```'))
        except:
            print(completion.choices[0].message.content)
            return {}
    
    # Test the old way
    query = "pizza margherita"
    item = "Pizza Margherita - Traditional Italian pizza with tomato sauce, mozzarella cheese, and fresh basil"
    result = call_judge(query, item)
    print(f"Old style result: {result}")


def new_style_usage():
    """Example of how to use the new class structure."""
    print("\n=== New Class-Based Style ===")
    
    # Method 1: Using the high-level prompt processor (recommended)
    print("Method 1: High-level prompt processor")
    processor = ModelFactory.create_prompt_processor()
    
    query = "pizza margherita"
    item = "Pizza Margherita - Traditional Italian pizza with tomato sauce, mozzarella cheese, and fresh basil"
    
    # Much simpler - just call the method!
    result = processor.judge_relevance(query, item)
    print(f"New style result: {result}")
    
    # Method 2: Using the LLM model directly (for custom prompts)
    print("\nMethod 2: Direct LLM model usage")
    llm_model = ModelFactory.create_default_llm_model()
    
    custom_prompt = "Explain the difference between pizza and pasta in 2 sentences."
    response = llm_model.generate(custom_prompt)
    print(f"Custom prompt response: {response}")
    
    # Method 3: JSON generation with error handling
    print("\nMethod 3: JSON generation")
    json_prompt = """
    List 3 popular Italian dishes in JSON format:
    {
        "dishes": [
            {"name": "dish name", "description": "brief description"}
        ]
    }
    """
    
    json_result = llm_model.generate_json(json_prompt)
    print(f"JSON result: {json.dumps(json_result, indent=2)}")


def migration_benefits():
    """Show the benefits of the new approach."""
    print("\n=== Migration Benefits ===")
    
    print("✓ Configuration-driven: Models configured in YAML files")
    print("✓ SSL handling: Automatic SSL configuration")
    print("✓ Error handling: Built-in JSON parsing with fallbacks")
    print("✓ Organized prompts: Prompts separated into modules")
    print("✓ Factory pattern: Easy model creation and switching")
    print("✓ Type safety: Proper typing and abstractions")
    print("✓ Extensible: Easy to add new LLM providers")
    print("✓ Consistent: Follows existing project patterns")
    
    # Show how easy it is to switch models
    print("\n--- Easy model switching ---")
    
    # Default model
    processor1 = ModelFactory.create_prompt_processor()
    print(f"Default model: {processor1.model.model_name}")
    
    # Different model
    processor2 = ModelFactory.create_prompt_processor("openai", "gpt-4.1-mini")
    print(f"Different model: {processor2.model.model_name}")
    
    # Custom configuration
    processor3 = ModelFactory.create_prompt_processor(
        config_override={"temperature": 0.8}
    )
    print(f"Custom config model: {processor3.model.model_name} (temp: {processor3.model.temperature})")


def equivalent_functions():
    """Show how your original functions map to new methods."""
    print("\n=== Function Mapping ===")
    
    processor = ModelFactory.create_prompt_processor()
    
    # Original: call_judge(query, item)
    # New: processor.judge_relevance(query, item)
    print("call_judge() → processor.judge_relevance()")
    
    # Original: call_rewrite(query)
    # New: processor.rewrite_query(query)
    print("call_rewrite() → processor.rewrite_query()")
    
    # Original: call_extend(query)
    # New: processor.extend_query(query)
    print("call_extend() → processor.extend_query()")
    
    # Original: call_extend_english(query)
    # New: processor.extend_query_english(query)
    print("call_extend_english() → processor.extend_query_english()")
    
    # Original: call_generate_query(queries, item)
    # New: processor.generate_query_for_item(queries, item)
    print("call_generate_query() → processor.generate_query_for_item()")
    
    print("\nAll functions now have:")
    print("- Better error handling")
    print("- Consistent interfaces")
    print("- Configuration management")
    print("- Organized prompt templates")


if __name__ == "__main__":
    # Demonstrate the migration
    old_style_usage()
    new_style_usage()
    migration_benefits()
    equivalent_functions()
