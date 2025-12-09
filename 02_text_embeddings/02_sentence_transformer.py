from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Convert text to text embeddings
vector = model.encode("Best movie ever!")

# Print the shape of the embedding vector
print("Embedding shape:", vector.shape)
print("Number of dimensions:", len(vector))
print("\nFirst 10 values of the embedding:")
print(vector[:10])
print("\nFull embedding vector:")
print(vector)
