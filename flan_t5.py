from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Performance optimization
torch.set_float32_matmul_precision('high')

# Flan-T5 is an encoder-decoder model (Seq2Seq), unlike Phi-3 which is decoder-only
model_name = "google/flan-t5-base"  # Options: flan-t5-small, base, large, xl, xxl

print("Loading Flan-T5 model...")
print(f"Model: {model_name}")
print("Note: Flan-T5 is a Seq2Seq model optimized for instruction-following tasks\n")

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    device_map="mps",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Model loaded. Generating response...\n")

# Flan-T5 works best with clear instructions/questions
prompt = "Translate to French: How are you today?"

# Alternative prompts to try:
# prompt = "Answer the question: What is the capital of France?"
# prompt = "Summarize: The quick brown fox jumps over the lazy dog multiple times."
# prompt = "What is 25 + 37?"

print(f"Prompt: {prompt}\n")

inputs = tokenizer(prompt, return_tensors="pt").to("mps")

# Generate with Flan-T5
with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        num_beams=4,  # Beam search for better quality
        early_stopping=True,
        temperature=0.7,
        do_sample=False,  # Use greedy/beam search for factual answers
    )

response = tokenizer.decode(output[0], skip_special_tokens=True)

print("="*60)
print("Generated Response:")
print("="*60)
print(response)
print("\n")

# Show the tokens (with special tokens)
print("="*60)
print("Token Analysis (with special tokens):")
print("="*60)
full_response = tokenizer.decode(output[0], skip_special_tokens=False)
print(f"Full output: {full_response}")
print(f"Token IDs: {output[0].tolist()}")

print("\n✓ Generation complete!")

# ============================================================
# SPECIAL TOKENS DEMONSTRATION
# ============================================================

print("\n\n" + "="*60)
print("FLAN-T5 SPECIAL TOKENS")
print("="*60 + "\n")

print("Flan-T5 special tokens:")
print(f"  PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
print(f"  EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
print(f"  UNK token: '{tokenizer.unk_token}' (ID: {tokenizer.unk_token_id})")

# T5 doesn't use CLS, SEP, or MASK tokens like BERT
print(f"\nNote: T5/Flan-T5 uses a different architecture than BERT:")
print(f"  - Uses EOS (</s>) instead of SEP")
print(f"  - Uses PAD (<pad>) for padding")
print(f"  - NO CLS token (encoder-decoder, not classification-focused)")
print(f"  - NO MASK token (uses span corruption during pretraining)")

# ============================================================
# TOKENIZER COMPARISON WITH FLAN-T5
# ============================================================

print("\n\n" + "="*60)
print("FLAN-T5 TOKENIZATION EXAMPLES")
print("="*60 + "\n")

test_sentences = [
    "What is the capital of India?",
    "Translate to Spanish: Hello, world!",
    "🤗 Emojis and 中文 text",
    "Code: def hello(): return 'Hi'"
]

for sentence in test_sentences:
    tokens = tokenizer(sentence, return_tensors="pt")
    token_ids = tokens.input_ids[0].tolist()
    
    print(f"\nInput: {sentence}")
    print(f"Token count: {len(token_ids)}")
    print(f"Tokens: {[tokenizer.decode([tid]) for tid in token_ids]}")
    print(f"IDs: {token_ids}")

print("\n" + "="*60)
