import os
import torch
import numpy as np
from transformers import T5ForConditionalGeneration, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIG ---
repo1_path = "path/to/first/repo"
repo2_path = "path/to/second/repo"

# --- DEVICE ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- INITIALIZE MODEL ---
checkpoint = "Salesforce/codet5p-770m"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# load the encoder-decoder model; we'll use the encoder outputs as embeddings
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)
model.eval()


# --- HELPER FUNCTIONS ---
def get_java_files(repo_path):
    """Recursively collect all Java files in a repository."""
    java_files = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".java"):
                java_files.append(os.path.join(root, file))
    return java_files


def embed_code_chunked(code, chunk_size=512, overlap=64):
    """
    Embed long code by splitting into overlapping chunks and averaging encoder embeddings.

    Args:
        code: Source code string
        chunk_size: Maximum tokens per chunk (default 512)
        overlap: Number of overlapping tokens between chunks (default 64)

    Returns:
        Averaged embedding representing the entire code file (numpy array)
    """
    # Tokenize into token ids (not necessarily final input ids length but workable for chunking)
    tokens = tokenizer.tokenize(code)

    if len(tokens) == 0:
        # fallback to empty input embedding
        inputs = tokenizer("", return_tensors="pt").to(device)
        with torch.no_grad():
            encoder_outputs = model.get_encoder()(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        # mean pool over sequence
        return encoder_outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    if len(tokens) <= chunk_size:
        inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=chunk_size).to(device)
        with torch.no_grad():
            encoder_outputs = model.get_encoder()(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        return encoder_outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    # Split tokens into overlapping chunks
    chunk_embeddings = []
    start_idx = 0

    print(f"Processing {len(tokens)} tokens in {len(tokens)//chunk_size + 1} chunks...")

    while start_idx < len(tokens):
        end_idx = min(start_idx + chunk_size, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]

        # Convert tokens back to text
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)

        inputs = tokenizer(chunk_text, return_tensors="pt", truncation=True, max_length=chunk_size).to(device)
        with torch.no_grad():
            encoder_outputs = model.get_encoder()(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        chunk_embedding = encoder_outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        chunk_embeddings.append(chunk_embedding)

        start_idx += chunk_size - overlap

        if end_idx >= len(tokens):
            break

    if chunk_embeddings:
        print(f"Averaged {len(chunk_embeddings)} chunk embeddings")
        return np.mean(chunk_embeddings, axis=0)
    else:
        inputs = tokenizer("", return_tensors="pt").to(device)
        with torch.no_grad():
            encoder_outputs = model.get_encoder()(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        return encoder_outputs.last_hidden_state.mean(dim=1).cpu().numpy()


def embed_code(code):
    """Return the mean-pooled embedding of code using CodeT5+ with chunking for long files."""
    return embed_code_chunked(code)


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


# --- LOAD FILES ---
repo1_files = get_java_files(repo1_path)
repo2_files = get_java_files(repo2_path)

print(f"Found {len(repo1_files)} Java files in repo1")
print(f"Found {len(repo2_files)} Java files in repo2")

# --- EMBEDDINGS WITH PROGRESS ---
print("\nEmbedding repo1 files...")
repo1_embeddings = []
for i, f in enumerate(repo1_files):
    print(f"Processing file {i+1}/{len(repo1_files)}: {os.path.basename(f)}")
    code = read_file(f)
    embedding = embed_code(code)
    repo1_embeddings.append(embedding)

print("\nEmbedding repo2 files...")
repo2_embeddings = []
for i, f in enumerate(repo2_files):
    print(f"Processing file {i+1}/{len(repo2_files)}: {os.path.basename(f)}")
    code = read_file(f)
    embedding = embed_code(code)
    repo2_embeddings.append(embedding)

# --- COMPUTE SIMILARITY MATRIX ---
print("\nComputing similarity matrix...")
similarity_matrix = np.zeros((len(repo1_files), len(repo2_files)))

for i, emb1 in enumerate(repo1_embeddings):
    for j, emb2 in enumerate(repo2_embeddings):
        similarity_matrix[i, j] = cosine_similarity(emb1, emb2)[0][0]

# --- OUTPUT MOST SIMILAR FILES ---
print("\n--- MOST SIMILAR FILES ---")
for i, file1 in enumerate(repo1_files):
    j_max = similarity_matrix[i].argmax()
    print(f"File {os.path.basename(file1)} is most similar to {os.path.basename(repo2_files[j_max])} with score {similarity_matrix[i, j_max]:.3f}")

# --- CALCULATE OVERALL SEMANTIC SIMILARITY ---
max_similarities = np.max(similarity_matrix, axis=1)
overall_similarity = np.mean(max_similarities)

print(f"\n--- OVERALL SEMANTIC SIMILARITY ---")
print(f"Raw average of maximum similarities per file: {overall_similarity:.4f}")

# --- SAVE FULL MATRIX ---
np.savetxt("repo_similarity_matrix_chunked_codet5.csv", similarity_matrix, delimiter=",", fmt="%.4f")
print(f"\nFull similarity matrix saved to repo_similarity_matrix_chunked_codet5.csv")
