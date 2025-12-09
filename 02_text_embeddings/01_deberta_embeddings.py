from transformers import AutoModel, AutoTokenizer

# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

# Load a language model
model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall", use_safetensors=True)

# Tokenize the sentence
tokens = tokenizer('Hello world', return_tensors='pt')

print(tokens)

# Process the tokens
output = model(**tokens)[0]

# Print the output shape
print("Output shape:")
print(output)
print()

# Inspect individual tokens
print("Token breakdown:")
for token in tokens['input_ids'][0]:
    print(tokenizer.decode(token))
