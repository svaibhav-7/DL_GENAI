
# Import models
sys.path.append(os.path.abspath("src"))
from train import BiLSTMEncoder, build_index
from inference import compute_features, rank_options, strip_wrapper

st.set_page_config(page_title="Smart MCQ Models", layout="wide")
st.title("Smart MCQ Models")

model_choice = st.sidebar.selectbox("Select Model", ["TF-IDF Baseline", "Bi-LSTM from Scratch", "BERT + LoRA", "RAG with Reranking"])

prompt = st.text_area("Question/Prompt:")
col1, col2 = st.columns(2)
with col1:
    A = st.text_input("Option A:")
    C = st.text_input("Option C:")
    E = st.text_input("Option E:")
with col2:
    B = st.text_input("Option B:")
    D = st.text_input("Option D:")

@st.cache_resource
def load_tfidf():
    return joblib.load('models/tfidf_baseline_model.joblib')

class SiameseMatcher(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, bidirectional, dropout, padding_idx=0):
        super().__init__()
        self.encoder = BiLSTMEncoder(vocab_size, embed_dim, hidden_dim, num_layers, bidirectional, dropout, padding_idx=padding_idx)
        enc_dim = self.encoder.out_dim
        self.scorer = nn.Sequential(
            nn.Linear(enc_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, prompt_ids, prompt_len, option_ids, option_len):
        B, num_opts, Lo = option_ids.shape
        prompt_vec = self.encoder(prompt_ids, prompt_len)
        prompt_vec_exp = prompt_vec.unsqueeze(1).expand(-1, num_opts, -1)
        opts_flat = option_ids.view(B * num_opts, Lo)
        opt_lens_flat = option_len.view(B * num_opts)
        option_vec = self.encoder(opts_flat, opt_lens_flat)
        option_vec = option_vec.view(B, num_opts, -1)
        diff = torch.abs(prompt_vec_exp - option_vec)
        prod = prompt_vec_exp * option_vec
        match_feats = torch.cat([prompt_vec_exp, option_vec, prod, diff], dim=-1)
        logits = self.scorer(match_feats).squeeze(-1)
        return logits

@st.cache_resource
def load_bilstm():
    checkpoint = torch.load('models/bilstm_scratch_model.pt', map_location='cpu', weights_only=False)
    word2idx = checkpoint['word2idx']
    config = checkpoint['config']

    UNK = "<UNK>"
    PAD = "<PAD>"
    pad_idx = word2idx.get(PAD, 0)
    vocab_size = len(word2idx)
    model = SiameseMatcher(
        vocab_size=vocab_size,
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_lstm_layers'],
        bidirectional=config['bidirectional'],
        dropout=config['dropout'],
        padding_idx=pad_idx
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, word2idx, config, pad_idx

@st.cache_resource
def load_bert():
    base_model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForMultipleChoice.from_pretrained(base_model_name)

    peft_model_id = "models"
    model = PeftModel.from_pretrained(base_model, peft_model_id)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_rag():
    import faiss
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    reranker = joblib.load('models/rag_reranker.joblib')
    return embedder, reranker

def predict_tfidf():
    vectorizer = load_tfidf()
    p_vec = vectorizer.transform([prompt])
    opts = [A, B, C, D, E]
    sims = []
    for opt in opts:
        if opt:
             o_vec = vectorizer.transform([opt])
             sims.append(cosine_similarity(p_vec, o_vec)[0][0])
        else:
             sims.append(-1)

    top_3_idx = np.argsort(sims)[-3:][::-1]
    labels = ["A", "B", "C", "D", "E"]
    return [labels[i] for i in top_3_idx]

def predict_bilstm():
    model, word2idx, config, pad_idx = load_bilstm()
    UNK = "<UNK>"

    def tokenize(text, max_len):
        import re
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        tokens = text.split()
        ids = [word2idx.get(w, word2idx.get(UNK, 1)) for w in tokens]
        if len(ids) == 0:
            ids = [pad_idx]
        length = min(len(ids), max_len)
        ids = ids[:max_len]
        ids = ids + [pad_idx] * (max_len - len(ids))
        return ids, length

    p_ids, p_len = tokenize(prompt, config['max_prompt_len'])

    opts = [A, B, C, D, E]
    o_ids = []
    o_lens = []
    for opt in opts:
        ids, length = tokenize(opt, config['max_option_len'])
        o_ids.append(ids)
        o_lens.append(length)

    p_ids_t = torch.tensor([p_ids], dtype=torch.long)
    p_len_t = torch.tensor([p_len], dtype=torch.long)
    o_ids_t = torch.tensor([o_ids], dtype=torch.long)
    o_lens_t = torch.tensor([o_lens], dtype=torch.long)

    with torch.no_grad():
        logits = model(p_ids_t, p_len_t, o_ids_t, o_lens_t)[0]

    top_3_idx = torch.argsort(logits, descending=True)[:3].tolist()
    labels = ["A", "B", "C", "D", "E"]
    return [labels[i] for i in top_3_idx]

def predict_bert():
    tokenizer, model = load_bert()

    opts = [A, B, C, D, E]
    prompts = [prompt] * len(opts)

    inputs = tokenizer(
        prompts,
        opts,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    top_3_idx = torch.argsort(logits[0], descending=True)[:3].tolist()
    labels = ["A", "B", "C", "D", "E"]
    return [labels[i] for i in top_3_idx]

def predict_rag():
    import pandas as pd
    embedder, reranker = load_rag()

    # We will simulate RAG with just the inputs provided since we don't have the full corpus loaded in app context
    import faiss
    def embed_texts(texts):
        emb = embedder.encode(list(texts), show_progress_bar=False, convert_to_numpy=True)
        faiss.normalize_L2(emb)
        return emb

    opts = [A, B, C, D, E]
    p_emb = embed_texts([prompt])
    o_embs = embed_texts(opts)

    sims = cosine_similarity(p_emb, o_embs)[0]

    # For standalone inputs, simulating retrieval match with self features
    retrieval_sim = sims
    direct_sim = sims

    feats = np.column_stack([direct_sim, retrieval_sim])

    logits = reranker.decision_function(feats)
    top_3_idx = np.argsort(logits)[-3:][::-1]

    labels = ["A", "B", "C", "D", "E"]
    return [labels[i] for i in top_3_idx]


if st.button("Predict"):
    if not prompt or not all([A, B, C, D, E]):
        st.warning("Please fill out the prompt and all options.")
    else:
        with st.spinner(f"Running {model_choice}..."):
            try:
                if model_choice == "TF-IDF Baseline":
                    preds = predict_tfidf()
                elif model_choice == "Bi-LSTM from Scratch":
                    preds = predict_bilstm()
                elif model_choice == "BERT + LoRA":
                    preds = predict_bert()
                elif model_choice == "RAG with Reranking":
                    preds = predict_rag()
                st.success(f"Top 3 Predictions: {', '.join(preds)}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
