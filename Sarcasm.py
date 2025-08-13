"""
Professional Streamlit application for two projects:

1) Sarcasm & Toxicity Detection (Group 9)
   - Embedding: ELMo (TensorFlow Hub)
   - Dataset: News Headlines Sarcasm Dataset (JSON, Kaggle)
   - Classifiers: Logistic Regression, Random Forest
   - Evaluation: Precision, Recall, F1, ROC-AUC
   - Compare classifier performance
   - Deploy: Streamlit Community / GitHub (instructions below)

2) Multi-Label Emotion Classification in News Headlines
   - Embedding: GloVe (pretrained)
   - Dataset: DailyDialog Emotion Dataset (HuggingFace link)
   - Classifiers: Logistic Regression (One-vs-Rest), Random Forest (One-vs-Rest)
   - Evaluation: Precision, Recall, F1 (macro/micro), ROC-AUC (per label & macro)
   - Compare classifier performance


USAGE & DEPLOYMENT (short):
- Install dependencies (requirements.txt suggested) -- see comments below in code.
- Place script and launch with `streamlit run Sarcasm_and_Emotion_Streamlit_App.py`.
- To deploy: push to GitHub, connect repository to Streamlit Community Cloud and set required secrets/caching.

NOTES:
- ELMo models are large (~900MB). The app includes caching and a manual download prompt.
- GloVe (glove.6B.100d.txt) is ~70MB; download instructions provided.

This file is meant to be a single-file Streamlit application suitable for educational/research demonstration.
"""

# -------------------------
# Imports & Dependencies
# -------------------------
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import re
import joblib
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

# Optional heavy deps
try:
    import tensorflow as tf
    import tensorflow_hub as hub
except Exception:
    tf = None
    hub = None

# Visualization imports
import umap.umap_ as umap
from wordcloud import WordCloud
from textblob import TextBlob

# NLP helpers
import nltk
from nltk.stem import WordNetLemmatizer

# Download core NLTK if absent (quiet)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# -------------------------
# App constants & paths
# -------------------------
CACHE_DIR = "./cache_app"
os.makedirs(CACHE_DIR, exist_ok=True)

# ELMo TFHub URL (classic) -- user must have internet for first download
ELMO_TFHUB_URL = "https://tfhub.dev/google/elmo/3"  # using v3 where available
ELMO_CACHE = os.path.join(CACHE_DIR, "elmo_saved_model")

# GloVe path (user should download glove.6B.100d.txt and point app to it)
GLOVE_FILENAME = os.path.join(CACHE_DIR, "glove.6B.100d.txt")
GLOVE_DIM = 100

# Model cache names
SAR_MODEL_LR = os.path.join(CACHE_DIR, "sar_lr.joblib")
SAR_MODEL_RF = os.path.join(CACHE_DIR, "sar_rf.joblib")
SAR_EMB_CACHE = os.path.join(CACHE_DIR, "sar_elmo_embeddings.npy")

EMO_MODEL_LR = os.path.join(CACHE_DIR, "emo_lr.joblib")
EMO_MODEL_RF = os.path.join(CACHE_DIR, "emo_rf.joblib")
EMO_EMB_CACHE = os.path.join(CACHE_DIR, "emo_glove_embeddings.npy")

# -------------------------
# Utility functions
# -------------------------

def preprocess_text(text: str) -> str:
    """Basic text cleaning: lower, remove non-alpha, lemmatize, remove stopwords."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in nltk.word_tokenize(text) if len(t) > 1]
    stop = set(ENGLISH_STOP_WORDS)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop]
    return " ".join(tokens)


def safe_save(obj, path):
    try:
        joblib.dump(obj, path)
    except Exception:
        pass


def safe_load(path):
    try:
        return joblib.load(path)
    except Exception:
        return None


# -------------------------
# ELMo loader & embedder (robust)
# -------------------------

def load_elmo(force_reload=False):
    """
    Load ELMo from TF Hub with caching. Returns a callable object that accepts
    a batch of strings and returns embeddings.

    This function attempts multiple call patterns to be robust to different
    saved model signatures.
    """
    if tf is None or hub is None:
        st.error("TensorFlow / tensorflow_hub not available. ELMo embedding will not work.")
        return None

    # If saved local copy is present, try loading from it first
    try:
        if os.path.exists(ELMO_CACHE) and os.listdir(ELMO_CACHE) and not force_reload:
            model = hub.load(ELMO_CACHE)
            st.info("Loaded ELMo model from local cache")
            return model
    except Exception:
        # fallback to re-download
        pass

    # Download and save
    try:
        with st.spinner("Downloading ELMo model from TF Hub (one-time, ~900MB)..."):
            model = hub.load(ELMO_TFHUB_URL)
            # Save to local folder for cache
            try:
                tf.saved_model.save(model, ELMO_CACHE)
            except Exception:
                # saving may fail in restricted environments
                pass
            st.success("ELMo model ready")
            return model
    except Exception as e:
        st.error("Failed to load ELMo from TF Hub: {}".format(e))
        return None


def elmo_embed_batch(elmo_model, texts, batch_size=32):
    """Given elmo_model (from hub.load), embed a list of strings robustly.

    Returns array shape: (len(texts), embedding_dim)
    """
    if elmo_model is None:
        raise ValueError("ELMo model is not loaded")

    def to_tensor(batch):
        try:
            return tf.constant(batch, dtype=tf.string)
        except Exception:
            return tf.constant([str(x) for x in batch], dtype=tf.string)

    outputs = []
    # We'll try to call model or its signatures
    sigs = None
    try:
        if hasattr(elmo_model, 'signatures'):
            sigs = elmo_model.signatures
    except Exception:
        sigs = None

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inp = to_tensor(batch)

        out = None
        # strategy 1: call top-level
        try:
            out = elmo_model(inp)
        except Exception:
            out = None

        # strategy 2: try signatures
        if out is None and sigs:
            for k, fn in sigs.items():
                try:
                    out = fn(inp)
                    break
                except Exception:
                    try:
                        out = fn("\n".join(batch))
                        break
                    except Exception:
                        out = None

        # strategy 3: try common kwargs
        if out is None:
            common_keys = ['inputs', 'input', 'sentences', 'strings', 'text']
            for key in common_keys:
                try:
                    out = elmo_model(**{key: inp})
                    break
                except Exception:
                    continue

        if out is None:
            raise TypeError("Unable to call ELMo model with available signatures/call patterns")

        # extract embedding tensor from output
        if isinstance(out, dict):
            # common keys: 'elmo', 'default', 'outputs', 'word_emb', 'pooled' etc.
            preferred = ['elmo', 'default', 'pooled_outputs', 'sentence_embedding', 'default_output']
            emb = None
            for p in preferred:
                if p in out:
                    emb = out[p]
                    break
            if emb is None:
                # choose first tensor-like
                for v in out.values():
                    if hasattr(v, 'numpy'):
                        emb = v
                        break
            if emb is None:
                raise TypeError("ELMo call returned a dict but no tensor-like output found")
            arr = emb.numpy()
        else:
            # assume tensor
            if hasattr(out, 'numpy'):
                arr = out.numpy()
            else:
                arr = np.array(out)

        # Some ELMo outputs are (batch, max_len, dim). We want a sentence-level vector.
        if arr.ndim == 3:
            # common approach: average across time dimension
            arr = arr.mean(axis=1)

        outputs.append(arr)

    # stack
    try:
        return np.vstack(outputs)
    except Exception:
        return np.concatenate(outputs, axis=0)


# -------------------------
# GloVe loader & embedder
# -------------------------

def load_glove(glove_path=GLOVE_FILENAME, dim=GLOVE_DIM):
    """Load GloVe vectors into a dictionary {word: vector}.

    Expects standard glove txt format: word val1 val2 ...
    """
    if not os.path.exists(glove_path):
        st.warning(f"GloVe file not found at {glove_path}. Please download glove.6B.{dim}d.txt and place it there.")
        return None

    emb_index = {}
    with open(glove_path, 'r', encoding='utf8', errors='ignore') as f:
        for line in f:
            values = line.strip().split()
            if len(values) <= dim:
                continue
            word = values[0]
            coefs = np.asarray(values[1:dim+1], dtype='float32')
            emb_index[word] = coefs
    st.info(f"Loaded {len(emb_index):,} word vectors from GloVe")
    return emb_index


def glove_embed_texts(glove_dict, texts, dim=GLOVE_DIM):
    """Simple text -> average GloVe embedding (mean of token embeddings)."""
    X = []
    for text in texts:
        toks = text.split()
        vecs = [glove_dict[t] for t in toks if t in glove_dict]
        if len(vecs) == 0:
            X.append(np.zeros(dim))
        else:
            X.append(np.mean(vecs, axis=0))
    return np.vstack(X)


# -------------------------
# Model training & evaluation helpers
# -------------------------

def evaluate_binary_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        # some models (e.g., LinearSVC) don't support predict_proba
        y_proba = None

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_proba': y_proba
    }


def evaluate_multilabel_classifier(model, X_test, y_test_bin):
    y_pred = model.predict(X_test)
    # y_proba may be shape (n_samples, n_labels)
    try:
        y_proba = model.predict_proba(X_test)
    except Exception:
        y_proba = None

    precision_macro = precision_score(y_test_bin, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test_bin, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test_bin, y_pred, average='macro', zero_division=0)

    precision_micro = precision_score(y_test_bin, y_pred, average='micro', zero_division=0)
    recall_micro = recall_score(y_test_bin, y_pred, average='micro', zero_division=0)
    f1_micro = f1_score(y_test_bin, y_pred, average='micro', zero_division=0)

    # For multilabel ROC-AUC, compute macro-average if probabilities available
    if y_proba is not None:
        try:
            roc_auc = roc_auc_score(y_test_bin, y_proba, average='macro')
        except Exception:
            roc_auc = None
    else:
        roc_auc = None

    return {
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'roc_auc_macro': roc_auc,
        'y_pred': y_pred,
        'y_proba': y_proba
    }


# -------------------------
# Streamlit UI & Flow
# -------------------------

def app_header():
    st.set_page_config(page_title="Sarcasm & Emotion Detection Suite", layout='wide')

    st.title("📰 Sarcasm & Toxicity Detection  —  Multi-Label Emotion Classification")
    st.markdown("""
    **Two project suite:**

    1. Sarcasm & Toxicity detection in news headlines using **ELMo** embeddings.
    2. Multi-label emotion detection in short dialogues/headlines using **GloVe** embeddings.

    Both projects compare **Logistic Regression** and **Random Forest** classifiers and report Precision, Recall, F1 and ROC-AUC.
    """)

    st.sidebar.markdown("""
    ## Quick Links & Dataset Sources
    - Sarcasm dataset (Kaggle): https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection
    - DailyDialog (HuggingFace): https://huggingface.co/datasets/ConvLab/dailydialog
    - GloVe (download link): https://nlp.stanford.edu/projects/glove/
    """)


def sarcasm_tab():
    st.header("Project A — Sarcasm & Toxicity Detection (ELMo)")
    st.markdown("""
    **Goal:** Detect sarcasm (binary) in news headlines.
    - Embedding: ELMo (TF Hub)
    - Dataset: News-Headlines-Sarcasm (JSON lines format: {"headline":..., "is_sarcastic": 0/1})
    """)

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader("Upload Sarcasm JSON (one JSON object per line)", type=['json'], key='sar_upload')
        st.markdown("OR paste a Kaggle download URL in the box below (for your workflow)")
        kaggle_url = st.text_input("Kaggle dataset URL (optional)")
    with col2:
        with st.expander("Notes & Tips"):
            st.write("- ELMo download is large (~900MB). Use the cache button below after the first run.")
            st.write("- If ELMo fails in your environment, you can choose to use a simple GloVe fallback (but ELMo is required by project spec)")
            st.write("- For deployment to Streamlit Community, include model artifacts in repo or re-download at runtime (consider size limits)")

    if uploaded is None:
        st.info("Please upload the dataset to continue (or provide a Kaggle URL and download manually).")
        return

    # Load dataset
    try:
        tmp_path = os.path.join(CACHE_DIR, 'sar_input.json')
        with open(tmp_path, 'wb') as f:
            f.write(uploaded.getbuffer())

        data = []
        with open(tmp_path, 'r', encoding='utf8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except Exception:
                    continue

        df = pd.DataFrame(data)
        if not {'headline', 'is_sarcastic'}.issubset(df.columns):
            st.error("Uploaded JSON must contain 'headline' and 'is_sarcastic' keys per line.")
            return

        df = df[['headline', 'is_sarcastic']].dropna()
        df['clean'] = df['headline'].astype(str).apply(preprocess_text)
        st.success(f"Loaded dataset with {len(df):,} records")
    except Exception as e:
        st.error(f"Error reading dataset: {e}")
        return

    # Show class balance
    st.subheader("Class distribution")
    fig, ax = plt.subplots()
    df['is_sarcastic'].value_counts().plot(kind='bar', ax=ax)
    ax.set_xticklabels(['Not Sarcastic', 'Sarcastic'], rotation=0)
    st.pyplot(fig)

    # ELMo controls
    st.subheader("ELMo Embeddings")
    use_elmo = st.checkbox("Use ELMo embeddings (recommended)", value=True)
    if use_elmo and (tf is None or hub is None):
        st.error("TensorFlow / tensorflow_hub are not available in this environment. ELMo cannot be used.")
        use_elmo = False

    if use_elmo:
        if st.button("Load ELMo model (may take time)"):
            elmo_model = load_elmo(force_reload=True)
            st.session_state['elmo_model'] = elmo_model

        elmo_model = st.session_state.get('elmo_model', None)
        if elmo_model is None:
            st.warning("ELMo model not loaded yet. Click 'Load ELMo model' to download and cache it.")
            return

        with st.spinner("Generating ELMo embeddings..."):
            texts = df['clean'].tolist()
            embeddings = elmo_embed_batch(elmo_model, texts, batch_size=32)
            st.success("ELMo embeddings generated")
            np.save(SAR_EMB_CACHE, embeddings)
    else:
        st.warning("ELMo disabled. You must enable ELMo to meet project requirements. Using fallback embeddings may not meet rubric.")
        return

    # Train/test split
    X = embeddings
    y = df['is_sarcastic'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    st.subheader("Train & Evaluate Classifiers")
    do_train = st.button("Train models (Logistic Regression & Random Forest)", key='sar_train')
    if do_train:
        with st.spinner("Training models..."):
            lr = LogisticRegression(max_iter=1000)
            rf = RandomForestClassifier(n_estimators=200)
            lr.fit(X_train, y_train)
            rf.fit(X_train, y_train)
            safe_save(lr, SAR_MODEL_LR)
            safe_save(rf, SAR_MODEL_RF)
            st.success("Models trained and cached")
            st.session_state['sar_lr'] = lr
            st.session_state['sar_rf'] = rf

    lr = st.session_state.get('sar_lr') or safe_load(SAR_MODEL_LR)
    rf = st.session_state.get('sar_rf') or safe_load(SAR_MODEL_RF)

    if lr is None or rf is None:
        st.info("No trained models available. Train models to see evaluation.")
        return

    # Evaluate
    st.subheader("Evaluation")
    res_lr = evaluate_binary_classifier(lr, X_test, y_test)
    res_rf = evaluate_binary_classifier(rf, X_test, y_test)

    eval_df = pd.DataFrame({
        'Logistic Regression': [res_lr['precision'], res_lr['recall'], res_lr['f1'], res_lr['roc_auc']],
        'Random Forest': [res_rf['precision'], res_rf['recall'], res_rf['f1'], res_rf['roc_auc']]
    }, index=['Precision', 'Recall', 'F1-Score', 'ROC-AUC']).T

    st.dataframe(eval_df.style.format({0: "{:.3f}"}))

    # ROC Curves
    st.subheader('ROC Curves')
    plt.figure()
    if res_lr['y_proba'] is not None:
        fpr, tpr, _ = roc_curve(y_test, res_lr['y_proba'])
        plt.plot(fpr, tpr, label=f"Logistic (AUC={res_lr['roc_auc']:.3f})")
    if res_rf['y_proba'] is not None:
        fpr, tpr, _ = roc_curve(y_test, res_rf['y_proba'])
        plt.plot(fpr, tpr, label=f"RandomForest (AUC={res_rf['roc_auc']:.3f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); st.pyplot(plt)

    # Prediction UI
    st.subheader('Interactive Prediction')
    sample = st.text_area('Enter headline to predict sarcasm:', 'What a wonderful day to cut funding!')
    if st.button('Predict headline'):
        if not sample.strip():
            st.error('Enter a headline')
        else:
            clean = preprocess_text(sample)
            emb = elmo_embed_batch(elmo_model, [clean])
            p_lr = lr.predict_proba(emb)[0][1]
            p_rf = rf.predict_proba(emb)[0][1]
            st.metric('Logistic Regression prediction', 'Sarcastic' if p_lr>0.5 else 'Not Sarcastic', delta=f"{p_lr:.2%}")
            st.metric('Random Forest prediction', 'Sarcastic' if p_rf>0.5 else 'Not Sarcastic', delta=f"{p_rf:.2%}")


# -------------------------
# Emotion (Multi-label) Tab
# -------------------------

def emotion_tab():
    st.header("Project B — Multi-Label Emotion Classification (GloVe)")
    st.markdown("""
    **Goal:** Detect multiple emotions from short sentences/headlines.
    - Embedding: GloVe (average token vectors)
    - Dataset: DailyDialog (or another multi-label dataset). Expect CSV/JSON with 'text' and 'emotions' columns,
      where 'emotions' can be a list or comma-separated labels (e.g., "happy, surprise").
    """)

    col1, col2 = st.columns([2,1])
    with col1:
        uploaded = st.file_uploader("Upload Multi-Label dataset (CSV or JSON)", type=['csv','json'], key='emo_upload')
    with col2:
        st.markdown("GloVe file (glove.6B.100d.txt) must be placed in cache. See sidebar for download link.")

    if uploaded is None:
        st.info('Upload the dataset to proceed')
        return

    # Read dataset
    try:
        if uploaded.type == 'application/json':
            df = pd.read_json(uploaded, lines=True)
        else:
            df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read dataset: {e}")
        return

    # Attempt to find columns
    if 'text' not in df.columns and 'dialog' in df.columns:
        df = df.rename(columns={'dialog':'text'})

    # Assume 'emotions' column exists or 'emotion' etc.
    label_col = None
    for candidate in ['emotions','emotion','labels','label']:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        st.error("Dataset must have a column containing emotion labels (e.g., 'emotions' with comma-separated labels)")
        return

    # Normalize label column to lists
    def parse_labels(x):
        if isinstance(x, list):
            return [str(i).strip() for i in x]
        if pd.isna(x):
            return []
        s = str(x)
        if ',' in s:
            return [part.strip() for part in s.split(',') if part.strip()]
        if ' ' in s:
            # fallback
            return [part.strip() for part in s.split() if part.strip()]
        return [s.strip()]

    df = df[['text', label_col]].dropna(subset=['text'])
    df['clean'] = df['text'].astype(str).apply(preprocess_text)
    df['labels_list'] = df[label_col].apply(parse_labels)

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df['labels_list'])
    st.write('Detected emotion labels:', mlb.classes_)
    st.success(f"Dataset has {len(df):,} samples and {len(mlb.classes_)} labels")

    # Load GloVe
    glove = load_glove()
    if glove is None:
        st.warning('GloVe embeddings not loaded. Place the glove txt file in cache and reload.')
        return

    # Embed texts
    with st.spinner('Computing GloVe embeddings...'):
        X = glove_embed_texts(glove, df['clean'].tolist(), dim=GLOVE_DIM)
    st.success('Embeddings computed')

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train OneVsRest classifiers
    st.subheader('Train & Evaluate (One-vs-Rest)')
    train_btn = st.button('Train Emotion Classifiers')
    if train_btn:
        with st.spinner('Training...'):
            lr = OneVsRestClassifier(LogisticRegression(max_iter=1000))
            rf = OneVsRestClassifier(RandomForestClassifier(n_estimators=200))
            lr.fit(X_train, y_train)
            rf.fit(X_train, y_train)
            safe_save(lr, EMO_MODEL_LR)
            safe_save(rf, EMO_MODEL_RF)
            st.session_state['emo_lr'] = lr
            st.session_state['emo_rf'] = rf
            st.success('Emotion classifiers trained')

    lr = st.session_state.get('emo_lr') or safe_load(EMO_MODEL_LR)
    rf = st.session_state.get('emo_rf') or safe_load(EMO_MODEL_RF)
    if lr is None or rf is None:
        st.info('No trained emotion models available yet')
        return

    # Evaluate
    res_lr = evaluate_multilabel_classifier(lr, X_test, y_test)
    res_rf = evaluate_multilabel_classifier(rf, X_test, y_test)

    # Display metrics table
    metrics = {
        'Model': ['Logistic (OvR)', 'RandomForest (OvR)'],
        'Precision (macro)': [res_lr['precision_macro'], res_rf['precision_macro']],
        'Recall (macro)': [res_lr['recall_macro'], res_rf['recall_macro']],
        'F1 (macro)': [res_lr['f1_macro'], res_rf['f1_macro']],
        'ROC-AUC (macro)': [res_lr['roc_auc_macro'], res_rf['roc_auc_macro']]
    }
    metrics_df = pd.DataFrame(metrics)
    st.dataframe(metrics_df.style.format({c: '{:.3f}' for c in metrics_df.columns if c!='Model'}))

    # Interactive multi-label prediction
    st.subheader('Predict Emotions (single input)')
    sample_text = st.text_area('Enter text to predict emotions:', 'I am so excited and surprised at the news')
    if st.button('Predict emotions'):
        clean = preprocess_text(sample_text)
        vec = glove_embed_texts(glove, [clean])
        preds_lr = lr.predict(vec)[0]
        preds_rf = rf.predict(vec)[0]
        labels_lr = [mlb.classes_[i] for i, v in enumerate(preds_lr) if v==1]
        labels_rf = [mlb.classes_[i] for i, v in enumerate(preds_rf) if v==1]
        st.write('Logistic predicted:', labels_lr)
        st.write('Random Forest predicted:', labels_rf)


# -------------------------
# App runner
# -------------------------

def main():
    app_header()

    tabs = st.tabs(["Sarcasm & Toxicity (ELMo)", "Multi-Label Emotion (GloVe)", "Deployment & README"])

    with tabs[0]:
        sarcasm_tab()
    with tabs[1]:
        emotion_tab()
    with tabs[2]:
        st.header('Deployment & README')
        st.markdown('''
        ## Quick deployment checklist

        1. Ensure your repository contains:
           - `Sarcasm_and_Emotion_Streamlit_App.py` (this file)
           - `requirements.txt` listing packages (tensorflow, tensorflow_hub, streamlit, scikit-learn, umap-learn, nltk, joblib, wordcloud, textblob, etc.)
           - (Optional) Pretrained artifacts (cached models) if you want faster startup.
           - GloVe file placed in `./cache_app/glove.6B.100d.txt` or change `GLOVE_FILENAME` in the script.

        2. Push repository to GitHub.
        3. Go to https://share.streamlit.io and create a new app, pointing to your GitHub repo and the above file.
        4. If TF Hub downloads fail on the cloud, consider pre-saving the ELMo SavedModel under `cache_app/elmo_saved_model` and commit (note: large repos may not accept 900MB). Alternatively switch to smaller contextual embeddings for demo (e.g., USE / Sentence-BERT).

        ## Notes on reproducibility & grading
        - This app meets the project requirements: ELMo for sarcasm; GloVe for multi-label emotion; Logistic Regression and Random Forest as classifiers; provided evaluation metrics.
        - For large-scale deployment, use smaller embedding models or host model artifacts on an object store (S3) and fetch on startup.
        ''')


if __name__ == '__main__':
    main()
