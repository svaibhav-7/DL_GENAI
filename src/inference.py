import faiss

def embed_texts(texts, embedder):
    """Returns L2-normalized embeddings (so dot product = cosine similarity).
    embedder should be a sentence-transformers model.
    """
    emb = embedder.encode(list(texts), show_progress_bar=False, convert_to_numpy=True)
    faiss.normalize_L2(emb)
    return emb

def retrieve(query_prompts, index, corpus_df, k, embedder):
    """Retrieves top k matches from the index for the given query_prompts."""
    q_emb = embed_texts(query_prompts, embedder)
    sims, idxs = index.search(q_emb, k)
    return sims, idxs  # (N, k) similarity scores and corpus row indices
