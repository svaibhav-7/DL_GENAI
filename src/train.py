import torch
import torch.nn as nn

class BiLSTMEncoder(nn.Module):
    """Encodes a batch of token-id sequences into a single pooled vector each."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, bidirectional, dropout, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids, lengths):
        embedded = self.dropout(self.embedding(token_ids))
        lengths_clamped = lengths.clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths_clamped, batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, c_n) = self.lstm(packed)
        if self.lstm.bidirectional:
            h_fwd = h_n[-2]
            h_bwd = h_n[-1]
            pooled = torch.cat([h_fwd, h_bwd], dim=-1)
        else:
            pooled = h_n[-1]
        return self.dropout(pooled)

def build_index(corpus_df, embedder):
    """corpus_df needs columns: core_prompt, correct_text
    embedder is expected to be a SentenceTransformer model or similar.
    Requires faiss to be imported where this is used.
    """
    import faiss

    def embed_texts(texts):
        emb = embedder.encode(list(texts), show_progress_bar=False, convert_to_numpy=True)
        faiss.normalize_L2(emb)
        return emb

    query_emb = embed_texts(corpus_df["core_prompt"].tolist())
    EMBED_DIM = query_emb.shape[1]
    index = faiss.IndexFlatIP(EMBED_DIM)  # inner product on normalized vecs = cosine sim
    index.add(query_emb)
    answer_emb = embed_texts(corpus_df["correct_text"].tolist())
    return index, answer_emb, corpus_df.reset_index(drop=True)
