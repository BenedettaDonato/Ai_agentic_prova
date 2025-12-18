#!/usr/bin/env python3
"""
Compare two SRS Markdown documents using BERT/Sentence-Transformer embeddings.

This script:
- Reads two Markdown files (not code) and strips fenced code blocks
- Tokenizes and chunks text to respect the model's max sequence length
- Computes mean-pooled embeddings per chunk with attention masking
- Aggregates chunk embeddings into a single document embedding (token-weighted)
- Prints cosine similarity between the two document embeddings

Default model: 'sentence-transformers/all-MiniLM-L6-v2'

Usage:
  python tools/bert_comparison.py --file1 path/to/a.md --file2 path/to/b.md

Optional flags:
  --model MODEL_NAME          HuggingFace model id (default: sentence-transformers/all-MiniLM-L6-v2)
  --max-length N              Max tokens per chunk (default: 512)
  --lowercase                 Lowercase text before processing (default: False)
  --show-details              Print per-document chunk stats (default: False)

Note: Ensure dependencies from tools/requirements.txt are installed.
"""

import argparse
import os
import re
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def read_markdown(path: str, lowercase: bool = False) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    # Remove fenced code blocks to focus on prose
    text = re.sub(r"```[\s\S]*?```", " ", text)
    # Collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.lower() if lowercase else text


def split_sentences(text: str) -> List[str]:
    # Simple sentence splitter without external deps
    # Split on ., !, ? followed by whitespace/newline; keep bullets and headings intact
    parts = re.split(r"(?<=[.!?])\s+", text)
    # Further split on headings/bullets to avoid giant lines
    expanded: List[str] = []
    for p in parts:
        # Split on newlines but keep non-empty
        expanded.extend([s.strip() for s in p.splitlines() if s.strip()])
    # Remove leftover markdown artifacts that don't add semantic value
    cleaned: List[str] = []
    for s in expanded:
        s = re.sub(r"^#+\s*", "", s)  # headings
        s = re.sub(r"^[-*]\s+", "", s)  # bullets
        s = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", s)  # links [text](url)
        s = s.strip()
        if s:
            cleaned.append(s)
    return cleaned


def pack_chunks(sentences: List[str], tokenizer, max_length: int) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    # Reserve 2 tokens for special tokens [CLS]/[SEP]
    budget = max(8, max_length - 2)

    for s in sentences:
        # Estimate token length without special tokens
        s_tokens = tokenizer.encode(s, add_special_tokens=False)
        s_len = len(s_tokens)
        if s_len > budget:
            # If a single sentence is too long, hard-split it by words
            words = s.split()
            left = 0
            while left < len(words):
                # Grow a sub-sentence until token budget
                lo = left
                hi = min(len(words), left + 50)  # start with ~50 words then adjust
                best_hi = lo
                while lo < len(words) and hi <= len(words):
                    sub = " ".join(words[lo:hi])
                    tlen = len(tokenizer.encode(sub, add_special_tokens=False))
                    if tlen <= budget:
                        best_hi = hi
                        hi += 25
                    else:
                        break
                if best_hi == lo:  # worst case fallback: take at least something
                    best_hi = min(len(words), lo + 30)
                sub = " ".join(words[lo:best_hi])
                if current and current_len + len(tokenizer.encode(sub, add_special_tokens=False)) > budget:
                    chunks.append(" ".join(current))
                    current, current_len = [], 0
                current.append(sub)
                current_len += len(tokenizer.encode(sub, add_special_tokens=False))
                left = best_hi
            continue

        if current_len + s_len > budget:
            chunks.append(" ".join(current))
            current, current_len = [s], s_len
        else:
            current.append(s)
            current_len += s_len

    if current:
        chunks.append(" ".join(current))
    return chunks


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden_state: [B, T, H]; attention_mask: [B, T]
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, T, 1]
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed_chunks(chunks: List[str], tokenizer, model, device, max_length: int) -> Tuple[np.ndarray, np.ndarray]:
    if not chunks:
        return np.zeros((1, model.config.hidden_size), dtype=np.float32), np.array([1.0], dtype=np.float32)

    all_vecs: List[np.ndarray] = []
    all_weights: List[float] = []

    model.eval()
    with torch.no_grad():
        for chunk in chunks:
            inputs = tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state  # [1, T, H]
            pooled = mean_pool(last_hidden_state, inputs["attention_mask"])  # [1, H]
            # L2 normalize
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vec = pooled.squeeze(0).cpu().numpy().astype(np.float32)
            all_vecs.append(vec)
            # weight by number of valid tokens
            weight = float(inputs["attention_mask"].sum().item())
            all_weights.append(weight)

    return np.stack(all_vecs, axis=0), np.array(all_weights, dtype=np.float32)


def aggregate_document_embedding(chunk_vecs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if chunk_vecs.ndim == 1:
        return chunk_vecs
    w = weights.reshape(-1, 1)
    weighted = (chunk_vecs * w).sum(axis=0)
    denom = max(float(w.sum()), 1e-9)
    doc_vec = weighted / denom
    # L2 normalize final doc vector
    norm = np.linalg.norm(doc_vec) or 1.0
    return (doc_vec / norm).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)


def main():
    parser = argparse.ArgumentParser(description="Compare two Markdown SRS docs using BERT embeddings")
    parser.add_argument("--file1", required=True, help="Path to first Markdown file")
    parser.add_argument("--file2", required=True, help="Path to second Markdown file")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="HuggingFace model id")
    parser.add_argument("--max-length", type=int, default=512, help="Max tokens per chunk (default: 512)")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase text before processing")
    parser.add_argument("--show-details", action="store_true", help="Print chunking and stats details")
    args = parser.parse_args()

    text1 = read_markdown(args.file1, lowercase=args.lowercase)
    text2 = read_markdown(args.file2, lowercase=args.lowercase)

    # Choose device (use MPS on Apple Silicon if available)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)

    sents1 = split_sentences(text1)
    sents2 = split_sentences(text2)

    chunks1 = pack_chunks(sents1, tokenizer, args.max_length)
    chunks2 = pack_chunks(sents2, tokenizer, args.max_length)

    vecs1, w1 = embed_chunks(chunks1, tokenizer, model, device, args.max_length)
    vecs2, w2 = embed_chunks(chunks2, tokenizer, model, device, args.max_length)

    doc1 = aggregate_document_embedding(vecs1, w1)
    doc2 = aggregate_document_embedding(vecs2, w2)

    sim = cosine_similarity(doc1, doc2)

    if args.show_details:
        print("--- Details ---")
        print(f"Model: {args.model}")
        print(f"Chunks (file1): {len(chunks1)} | Tokens (approx): {int(w1.sum())}")
        print(f"Chunks (file2): {len(chunks2)} | Tokens (approx): {int(w2.sum())}")
        print()

    print(f"Similarity (cosine): {sim:.4f}")


if __name__ == "__main__":
    main()
