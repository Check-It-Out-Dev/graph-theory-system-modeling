"""Debug script for testing embedding issues."""
import json
import sys
sys.path.insert(0, 'src')

from pydantic import BaseModel, Field

# Test 1: JSON Schema
class TestTexts(BaseModel):
    texts: list[str] | str = Field(...)

print("=== TEST 1: JSON Schema ===")
print(json.dumps(TestTexts.model_json_schema(), indent=2))

# Test 2: Parsing list
print("\n=== TEST 2: Parsing list ===")
t = TestTexts(texts=['a', 'b', 'c'])
print(f"Type: {type(t.texts)}, Value: {t.texts}")

# Test 3: Parsing string
print("\n=== TEST 3: Parsing string ===")
t2 = TestTexts(texts="single string")
print(f"Type: {type(t2.texts)}, Value: {t2.texts}")

# Test 4: Check model prompts (without loading full model)
print("\n=== TEST 4: Check available prompts ===")
try:
    from sentence_transformers import SentenceTransformer
    # Just check config without loading weights
    from huggingface_hub import hf_hub_download
    import json as json_mod
    
    config_path = hf_hub_download("Qwen/Qwen3-Embedding-8B", "config_sentence_transformers.json")
    with open(config_path) as f:
        st_config = json_mod.load(f)
    print(f"Prompts in config: {st_config.get('prompts', 'NOT FOUND')}")
except Exception as e:
    print(f"Could not check prompts: {e}")

# Test 5: Direct embedding test (if model loaded)
print("\n=== TEST 5: Embedding batch test ===")
try:
    from qwen3_embedding_mcp.embedding_engine import get_engine
    engine = get_engine()
    if engine.is_loaded:
        result = engine.embed(['text one', 'text two', 'text three'])
        print(f"Input: 3 texts")
        print(f"Output num_texts: {result.num_texts}")
        print(f"Output embeddings count: {len(result.embeddings)}")
    else:
        print("Model not loaded - skipping (would take too long)")
except Exception as e:
    print(f"Error: {e}")
