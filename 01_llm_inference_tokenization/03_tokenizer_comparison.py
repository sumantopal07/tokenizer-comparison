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

prompt = "Whats India's score right now?"

inputs = tokenizer(prompt, return_tensors="pt").to("mps")

# Generate with optimized settings - WITH CACHE!
with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=50,
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

# ============================================================
# FLAN-T5 INFERENCE (2022) - Google's Instruction-Tuned T5
# ============================================================

print("\n\n" + "="*60)
print("FLAN-T5 (2022) - Encoder-Decoder Seq2Seq Model")
print("="*60 + "\n")

from transformers import AutoModelForSeq2SeqLM

# Flan-T5 is an encoder-decoder model (Seq2Seq), unlike Phi-3 which is decoder-only
flan_model_name = "google/flan-t5-base"  # Options: flan-t5-small, base, large, xl, xxl

print("Loading Flan-T5 model...")
print(f"Model: {flan_model_name}")
print("Note: Flan-T5 is a Seq2Seq model optimized for instruction-following tasks\n")

flan_model = AutoModelForSeq2SeqLM.from_pretrained(
    flan_model_name,
    device_map="mps",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

flan_tokenizer = AutoTokenizer.from_pretrained(flan_model_name)

print("Flan-T5 loaded. Generating response...\n")

# Flan-T5 works best with clear instructions/questions
flan_prompt = "Translate to French: How are you today?"

print(f"Prompt: {flan_prompt}\n")

flan_inputs = flan_tokenizer(flan_prompt, return_tensors="pt").to("mps")

# Generate with Flan-T5
with torch.inference_mode():
    flan_output = flan_model.generate(
        **flan_inputs,
        max_new_tokens=100,
        num_beams=4,  # Beam search for better quality
        early_stopping=True,
        temperature=0.7,
        do_sample=False,  # Use greedy/beam search for factual answers
    )

flan_response = flan_tokenizer.decode(flan_output[0], skip_special_tokens=True)

print("Generated Response:", flan_response)
print("\nFull output (with special tokens):", flan_tokenizer.decode(flan_output[0], skip_special_tokens=False))
print("Token IDs:", flan_output[0].tolist())

print("\n" + "-"*60)
print("Flan-T5 Special Tokens:")
print("-"*60)
print(f"  PAD token: '{flan_tokenizer.pad_token}' (ID: {flan_tokenizer.pad_token_id})")
print(f"  EOS token: '{flan_tokenizer.eos_token}' (ID: {flan_tokenizer.eos_token_id})")
print(f"  UNK token: '{flan_tokenizer.unk_token}' (ID: {flan_tokenizer.unk_token_id})")

print(f"\nNote: T5/Flan-T5 uses a different architecture than BERT:")
print(f"  - Uses EOS (</s>) instead of SEP")
print(f"  - Uses PAD (<pad>) for padding")
print(f"  - NO CLS token (encoder-decoder, not classification-focused)")
print(f"  - NO MASK token (uses span corruption during pretraining)")

print("\n✓ Flan-T5 generation complete!")

# ============================================================
# COMPREHENSIVE TOKENIZER COMPARISON WITH STATISTICS
# ============================================================

print("\n\n" + "="*80)
print("COMPREHENSIVE TOKENIZER COMPARISON & ANALYSIS")
print("="*80 + "\n")

# Test text with various types of content
test_text = """English and CAPITALIZATION
show_tokens False None elif == >= else: two tabs:"\t\t" Three tabs:\t\t\t
12.0*50=600
🫡👋🏾🤗🥴😶‍🌫️
ہندی中文العربية한국어"""

# Color palette for token visualization
colors_list = [
    '102;194;165', '252;141;98', '141;160;203',
    '231;138;195', '166;216;84', '255;217;47',
    '255;127;0', '202;178;214', '106;61;154'
]

# List of tokenizers to compare
tokenizers_to_test = [
    ("bert-base-uncased", "BERT base model (uncased)"),
    ("bert-base-cased", "BERT base model (cased)"),
    ("gpt2", "GPT-2"),
    ("google/flan-t5-base", "FLAN-T5"),
    ("Salesforce/codegen-350M-mono", "CodeGen (similar to StarCoder)"),
    ("facebook/galactica-125m", "Galactica"),
    ("microsoft/Phi-3-mini-4k-instruct", "Phi-3 and Llama 2"),
]

def analyze_tokenizer(tokenizer_name, display_name):
    """Comprehensive tokenizer analysis with statistics"""
    print("\n" + "="*80)
    print(f"📊 {display_name}")
    print("="*80)
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        
        # Tokenize
        encoding = tokenizer(test_text, return_tensors="pt", add_special_tokens=True)
        token_ids = encoding.input_ids[0].tolist()
        tokens = [tokenizer.decode([tid]) for tid in token_ids]
        
        # ============ BASIC STATISTICS ============
        print("\n📈 BASIC STATISTICS:")
        print(f"  • Total tokens: {len(token_ids)}")
        print(f"  • Original text length: {len(test_text)} characters")
        print(f"  • Compression ratio: {len(test_text)/len(token_ids):.2f} chars/token")
        print(f"  • Vocabulary size: {tokenizer.vocab_size:,}")
        
        # ============ SPECIAL TOKENS ============
        print("\n🔖 SPECIAL TOKENS:")
        special_tokens_map = {
            'PAD': tokenizer.pad_token,
            'UNK': tokenizer.unk_token,
            'CLS': getattr(tokenizer, 'cls_token', None),
            'SEP': getattr(tokenizer, 'sep_token', None),
            'MASK': getattr(tokenizer, 'mask_token', None),
            'BOS': getattr(tokenizer, 'bos_token', None),
            'EOS': tokenizer.eos_token,
        }
        
        for token_type, token_value in special_tokens_map.items():
            if token_value:
                token_id = tokenizer.convert_tokens_to_ids(token_value) if hasattr(tokenizer, 'convert_tokens_to_ids') else 'N/A'
                print(f"  • {token_type}: '{token_value}' (ID: {token_id})")
        
        # ============ TOKENIZATION STRATEGY ============
        print("\n🔧 TOKENIZATION STRATEGY:")
        tokenizer_class = tokenizer.__class__.__name__
        print(f"  • Tokenizer class: {tokenizer_class}")
        
        # Detect strategy
        if 'WordPiece' in tokenizer_class or 'Bert' in tokenizer_class:
            strategy = "WordPiece (BERT-style)"
        elif 'BPE' in tokenizer_class or 'GPT' in tokenizer_class:
            strategy = "Byte-Pair Encoding (BPE)"
        elif 'SentencePiece' in tokenizer_class or 'T5' in tokenizer_class or 'Llama' in tokenizer_class:
            strategy = "SentencePiece (Unigram)"
        else:
            strategy = "Unknown/Custom"
        print(f"  • Strategy: {strategy}")
        
        # Case handling
        model_max_length = getattr(tokenizer, 'model_max_length', 'N/A')
        print(f"  • Model max length: {model_max_length}")
        
        # ============ TOKEN ANALYSIS ============
        print("\n📊 TOKEN BREAKDOWN ANALYSIS:")
        
        # Count special tokens, unknown tokens, subwords
        special_token_count = sum(1 for t in tokens if t in special_tokens_map.values())
        unknown_token_count = sum(1 for tid in token_ids if tid == tokenizer.unk_token_id) if tokenizer.unk_token_id else 0
        
        # Detect subword tokens (those starting with ##, Ġ, or similar)
        subword_indicators = ['##', 'Ġ', '▁', '</w>']
        subword_count = sum(1 for t in tokens if any(t.startswith(ind) or ind in t for ind in subword_indicators))
        
        print(f"  • Special tokens in output: {special_token_count}")
        print(f"  • Unknown tokens [UNK]: {unknown_token_count}")
        print(f"  • Subword tokens: {subword_count}")
        print(f"  • Regular tokens: {len(tokens) - special_token_count - subword_count}")
        
        # ============ CONTENT-SPECIFIC ANALYSIS ============
        print("\n🔍 CONTENT-SPECIFIC HANDLING:")
        
        # Test individual components
        test_cases = {
            "Capitalization": "CAPITALIZATION",
            "Python keywords": "False None elif",
            "Operators": "== >=",
            "Numbers": "12.0*50=600",
            "Emojis": "🫡👋🏾🤗",
            "Non-English": "ہندی中文العربية한국어",
        }
        
        for component_name, component_text in test_cases.items():
            comp_tokens = tokenizer(component_text, add_special_tokens=False).input_ids
            avg_chars_per_token = len(component_text) / len(comp_tokens) if len(comp_tokens) > 0 else 0
            print(f"  • {component_name}: {len(comp_tokens)} tokens (avg {avg_chars_per_token:.2f} chars/token)")
        
        # ============ VISUAL TOKEN DISPLAY ============
        print("\n🎨 VISUAL TOKENIZATION:")
        print("  ", end='')
        
        for idx, (tid, token) in enumerate(zip(token_ids, tokens)):
            # Clean display (replace tabs/newlines with visible characters)
            display_token = token.replace('\t', '\\t').replace('\n', '\\n').replace(' ', '·')
            if not display_token.strip():
                display_token = '·'
            
            print(
                f'\x1b[0;30;48;2;{colors_list[idx % len(colors_list)]}m' +
                f'{display_token}' +
                '\x1b[0m',
                end=' '
            )
        
        print("\n")
        
        # ============ DETAILED TOKEN LIST ============
        print("📝 DETAILED TOKEN LIST (first 50 tokens):")
        for idx, (tid, token) in enumerate(zip(token_ids[:50], tokens[:50])):
            display_token = repr(token).strip("'")
            print(f"  [{idx:2d}] ID:{tid:6d} → {display_token}")
        
        if len(token_ids) > 50:
            print(f"  ... and {len(token_ids) - 50} more tokens")
        
    except Exception as e:
        print(f"\n❌ Error loading tokenizer: {str(e)}")
        print(f"   This tokenizer may require authentication or special setup.")

# ============================================================
# RUN ANALYSIS FOR ALL TOKENIZERS
# ============================================================

print("\n📋 Test text overview:")
print("-" * 80)
print(test_text)
print("-" * 80)
print(f"\nTotal characters: {len(test_text)}")
print("Contains: English, capitalization, Python code, operators, numbers,")
print("          emojis, non-English scripts (Urdu, Hindi, Chinese, Arabic, Korean)")

# Analyze each tokenizer
for tokenizer_name, display_name in tokenizers_to_test:
    analyze_tokenizer(tokenizer_name, display_name)

# ============================================================
# SUMMARY COMPARISON TABLE
# ============================================================

print("\n\n" + "="*80)
print("📊 SUMMARY COMPARISON TABLE")
print("="*80)

print(f"\n{'Tokenizer':<35} {'Tokens':<10} {'Vocab Size':<15} {'Compression':<15}")
print("-" * 80)

for tokenizer_name, display_name in tokenizers_to_test:
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        token_count = len(tokenizer(test_text).input_ids)
        vocab_size = tokenizer.vocab_size
        compression = len(test_text) / token_count
        
        print(f"{display_name:<35} {token_count:<10} {vocab_size:<15,} {compression:<15.2f}")
    except:
        print(f"{display_name:<35} {'ERROR':<10} {'N/A':<15} {'N/A':<15}")

print("\n" + "="*80)
