from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import torch

# Performance optimization
torch.set_float32_matmul_precision('high')

# Complete compatibility wrapper for old Phi-3 model code
class Phi3LegacyCache(DynamicCache):
    """Compatibility wrapper that adds legacy Cache API methods for Phi-3"""
    
    def __init__(self):
        super().__init__()
        self._seen_tokens = 0
    
    @property
    def seen_tokens(self):
        """Legacy property for tracking total tokens seen"""
        # DynamicCache stores data in self.layers
        if len(self.layers) == 0:
            return 0
        return self.get_seq_length()
    
    @seen_tokens.setter
    def seen_tokens(self, value):
        self._seen_tokens = value
    
    def get_max_length(self):
        """Legacy method - returns None for dynamic (unlimited) cache"""
        return None
    
    def get_usable_length(self, new_seq_length=None, layer_idx=0):
        """Legacy method - returns current sequence length"""
        return self.get_seq_length(layer_idx)
    
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """Override to track seen tokens"""
        result = super().update(key_states, value_states, layer_idx, cache_kwargs)
        self._seen_tokens = self.get_seq_length(layer_idx)
        return result


model_name = "microsoft/Phi-3-mini-4k-instruct"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="mps",
    dtype=torch.float16,
    trust_remote_code=True,
    attn_implementation="eager",
    low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

print("Model loaded. Generating response...\n")

prompt = "Write a short email apologizing to Sarah for the gardening mishap."

inputs = tokenizer(prompt, return_tensors="pt").to("mps")

# Generate with optimized settings - WITH CACHE!
with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        use_cache=True,  # ✅ ENABLED!
        past_key_values=Phi3LegacyCache(),  # Use our compatibility wrapper
        pad_token_id=tokenizer.eos_token_id,
        num_beams=1
    )

response = tokenizer.decode(output[0])
print(output)
for id in inputs["input_ids"][0]:
    print(tokenizer.decode(id))
print("-------------------")
print("-------------------")
print("-------------------")
print("-------------------")
print("-------------------")
print(tokenizer.decode(2))
print(tokenizer.decode(23))
print(tokenizer.decode([3323, 116122]))
print(tokenizer.decode(9901))
print(response)
print("\n✓ Generation complete with KV cache enabled!")
