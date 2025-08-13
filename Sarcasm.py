import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from textblob import TextBlob
import umap.umap_ as umap
import shutil

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Constants
CACHE_DIR = "./cache"
ELMO_MODEL_URL = "https://tfhub.dev/google/elmo/2"  # Version 2 for TF 2.10
ELMO_LOCAL_PATH = os.path.join(CACHE_DIR, "elmo_model")  # Local path for cached model
MODEL_PATH_LR = os.path.join(CACHE_DIR, "logistic_regression_model.joblib")
MODEL_PATH_RF = os.path.join(CACHE_DIR, "random_forest_model.joblib")
EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "elmo_embeddings.npy")
DATA_PATH = os.path.join(CACHE_DIR, "dataset.json")

# Create cache directory if not exists
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(ELMO_LOCAL_PATH, exist_ok=True)


def load_elmo_model():
    """Load ELMo model with local caching and handle different signature versions"""
    # Check if model is already downloaded
    if not os.listdir(ELMO_LOCAL_PATH):
        st.info("Downloading ELMo model (first time, ~900MB). This may take several minutes...")
        try:
            # Download and save model
            with st.spinner("Downloading ELMo model..."):
                model = hub.load(ELMO_MODEL_URL)
                tf.saved_model.save(model, ELMO_LOCAL_PATH)
            st.success("✅ ELMo model downloaded and saved locally")
        except Exception as e:
            st.error(f"Error downloading ELMo model: {e}")
            st.error("Ensure tensorflow-text==2.10.0 is installed")
            st.stop()

    try:
        # Load from local cache
        model = hub.load(ELMO_LOCAL_PATH)
        
        # Handle different signature versions
        if hasattr(model, 'signatures'):
            # Try to get serving_default signature first
            if 'serving_default' in model.signatures:
                st.info("Using 'serving_default' signature")
                return model.signatures['serving_default']
            
            # Try to get default signature
            if 'default' in model.signatures:
                st.info("Using 'default' signature")
                return model.signatures['default']
            
            # Try to get any available signature
            available_signatures = list(model.signatures.keys())
            if available_signatures:
                signature_name = available_signatures[0]
                st.warning(f"Using first available signature: '{signature_name}'")
                return model.signatures[signature_name]
        
        # If no signatures found, return the model directly
        st.info("No signatures found, using model directly")
        return model
    except Exception as e:
        # Provide detailed error information
        st.error(f"Error loading cached ELMo model: {e}")
        
        # Show available signatures if possible
        try:
            model = hub.load(ELMO_LOCAL_PATH)
            if hasattr(model, 'signatures'):
                signatures = list(model.signatures.keys())
                st.error(f"Available signatures: {signatures}")
            else:
                st.error("No signatures attribute found in model")
        except:
            st.error("Could not inspect model signatures")
        
        st.error("Try clearing cache and retrying")
        st.stop()


def elmo_embeddings(model, texts):
    """Generate ELMo embeddings for text inputs with robust key handling"""
    embeddings = []
    batch_size = 32
    embedding_key = None  # To store the key we discover

    progress_bar = st.progress(0)
    status_text = st.empty()

    def _to_string_tensor(batch):
        """Safely convert a Python list of strings to a tf.string tensor"""
        try:
            return tf.convert_to_tensor(batch, dtype=tf.string)
        except Exception:
            # fallback
            try:
                batch = [str(x) for x in batch]
                return tf.constant(batch, dtype=tf.string)
            except Exception:
                return tf.constant(batch)

    def _try_call_concrete_fn(fn, inp):
        """
        Try calling a ConcreteFunction or callable `fn` with multiple conventions.
        Returns (success_flag, output, used_call_description)
        """
        # 1) direct call with tensor
        try:
            out = fn(inp)
            return True, out, "direct_tensor"
        except Exception:
            pass

        # 2) try common kwarg names
        keys_to_try = ['inputs', 'input', 'text', 'texts', 'tokens', 'sentences', 'strings']
        for k in keys_to_try:
            try:
                out = fn(**{k: inp})
                return True, out, f"kwarg_{k}"
            except Exception:
                continue

        # 3) try passing dict wrapper
        try:
            out = fn({'inputs': inp})
            return True, out, "dict_inputs"
        except Exception:
            pass

        # 4) try to build kwargs from structured_input_signature if present
        try:
            sig = getattr(fn, 'structured_input_signature', None)
            if sig:
                _, kwargs_spec = sig
                if isinstance(kwargs_spec, dict) and len(kwargs_spec) > 0:
                    # Build kwargs mapping all names to inp (best-effort)
                    call_kwargs = {name: inp for name in kwargs_spec.keys()}
                    try:
                        out = fn(**call_kwargs)
                        return True, out, f"structured_inputs_{list(kwargs_spec.keys())}"
                    except Exception:
                        pass
        except Exception:
            pass

        return False, None, None

    def _call_model(m, batch):
        """Try various calling strategies on model `m` for `batch` (list of strings)."""
        inp = _to_string_tensor(batch)

        # If model has signatures (a dict of ConcreteFunctions), try them first
        if hasattr(m, 'signatures') and isinstance(getattr(m, 'signatures'), dict):
            sig_keys = list(m.signatures.keys())
            for key in sig_keys:
                fn = m.signatures[key]
                success, out, desc = _try_call_concrete_fn(fn, inp)
                if success:
                    st.info(f"Called signature '{key}' with method: {desc}")
                    return out
            # if none of the signatures worked, continue to try the top-level object
        # If m itself is a ConcreteFunction or callable, try it
        success, out, desc = _try_call_concrete_fn(m, inp)
        if success:
            st.info(f"Called top-level model object with method: {desc}")
            return out

        # If it's a keras layer wrapper with call method, try calling its .call
        if hasattr(m, 'call'):
            try:
                fn = m.call
                success, out, desc = _try_call_concrete_fn(fn, inp)
                if success:
                    st.info(f"Called model.call with method: {desc}")
                    return out
            except Exception:
                pass

        # Nothing worked — raise TypeError with useful debug info
        sig_info = None
        try:
            if hasattr(m, 'signatures'):
                sig_info = list(m.signatures.keys())
        except Exception:
            sig_info = None

        raise TypeError(
            f"Unable to call the ELMo model. Model type: {type(m)}. "
            f"Available signatures: {sig_info}. Tried direct tensor, common kwargs, dict wrapper, "
            f"and structured_input_signature. See logs for more."
        )

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Call the model using the robust helper
        output = _call_model(model, batch)

        # Determine the correct output key (only on first batch)
        if embedding_key is None:
            if isinstance(output, dict):
                # Check for known keys
                if 'elmo' in output:
                    embedding_key = 'elmo'
                    st.info("Using 'elmo' output key")
                elif 'default' in output:
                    embedding_key = 'default'
                    st.info("Using 'default' output key")
                elif 'output' in output:
                    embedding_key = 'output'
                    st.info("Using 'output' key")
                else:
                    # Try to find any tensor output
                    for key, value in output.items():
                        if isinstance(value, tf.Tensor) or hasattr(value, 'numpy'):
                            embedding_key = key
                            st.warning(f"Using discovered output key: '{embedding_key}'")
                            break

                    # If still not found, show error
                    if embedding_key is None:
                        st.error(f"Could not find embedding tensor in model output. Keys: {list(output.keys())}")
                        st.stop()
            else:
                # Output is a tensor-like object, not a dictionary
                embedding_key = 'tensor'
                st.info("Model output is a tensor or tensor-like object")

        # Extract embeddings
        if embedding_key == 'tensor':
            # If output is a tensor-like object (e.g., tf.Tensor or numpy array)
            try:
                batch_emb = output.numpy()
            except Exception:
                # fallback if it's already a numpy array or list
                batch_emb = np.array(output)
        else:
            try:
                val = output[embedding_key]
                try:
                    batch_emb = val.numpy()
                except Exception:
                    batch_emb = np.array(val)
            except Exception as e:
                st.error(f"Could not extract embeddings for key '{embedding_key}': {e}")
                st.stop()

        embeddings.append(batch_emb)

        # Update progress
        progress = min((i + batch_size) / max(len(texts), 1), 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Processing batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")

    progress_bar.empty()
    status_text.empty()
    try:
        return np.vstack(embeddings)
    except Exception:
        # If stacking fails, try concatenation (handles ragged shapes)
        return np.concatenate(embeddings, axis=0)


def preprocess_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize
    tokens = text.split()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


def load_and_preprocess_data(uploaded_file):
    """Load and preprocess dataset with comprehensive cleaning"""
    try:
        # Save uploaded file
        with open(DATA_PATH, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load JSON data
        data = []
        with open(DATA_PATH, "r") as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        # Convert to DataFrame
        df = pd.DataFrame(data)
        df = df[["headline", "is_sarcastic"]]

        # Handle missing values
        initial_count = len(df)
        df.dropna(inplace=True)
        df = df[df['headline'].str.strip() != '']
        cleaned_count = len(df)

        # Add preprocessing info
        st.info(f"Removed {initial_count - cleaned_count} records with missing values")

        # Text preprocessing
        with st.spinner("Preprocessing text data..."):
            df['cleaned_headline'] = df['headline'].apply(preprocess_text)

        return df
    except Exception as e:
        st.error(f"Error processing data: {e}")
        st.error("Please ensure you've uploaded a valid JSON file in the required format")
        st.stop()


def plot_class_distribution(df, title):
    """Plot class distribution with annotations"""
    class_counts = df['is_sarcastic'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(['Not Sarcastic (0)', 'Sarcastic (1)'], class_counts,
                  color=['skyblue', 'salmon'])

    # Add counts on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:,}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.title(f'Class Distribution: {title}', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(fontsize=10)
    st.pyplot(fig)
    return fig


def plot_text_length_analysis(df):
    """Analyze and visualize text length distribution"""
    # Calculate text lengths
    df['text_length'] = df['cleaned_headline'].apply(lambda x: len(x.split()))

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot distributions for both classes
    sns.histplot(data=df, x='text_length', hue='is_sarcastic',
                 element='step', stat='density', common_norm=False,
                 palette={0: 'skyblue', 1: 'salmon'}, alpha=0.6)

    # Add mean lines
    mean_length_0 = df[df['is_sarcastic'] == 0]['text_length'].mean()
    mean_length_1 = df[df['is_sarcastic'] == 1]['text_length'].mean()

    ax.axvline(mean_length_0, color='blue', linestyle='--', label='Mean (Non-Sarcastic)')
    ax.axvline(mean_length_1, color='red', linestyle='--', label='Mean (Sarcastic)')

    plt.title('Text Length Distribution by Class', fontsize=14)
    plt.xlabel('Number of Words', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(title='Class', labels=['Non-Sarcastic', 'Sarcastic',
                                      f'Mean (Non-Sarcastic): {mean_length_0:.1f}',
                                      f'Mean (Sarcastic): {mean_length_1:.1f}'])
    plt.tight_layout()

    # Display statistics
    stats = df.groupby('is_sarcastic')['text_length'].describe().reset_index()
    stats.columns = ['Class', 'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    stats['Class'] = stats['Class'].map({0: 'Non-Sarcastic', 1: 'Sarcastic'})

    st.dataframe(stats[['Class', 'Mean', 'Std', 'Min', '50%', 'Max']].rename(
        columns={'50%': 'Median'}).style.format({'Mean': '{:.1f}', 'Std': '{:.1f}'}))

    return fig


def plot_sentiment_distribution(df):
    """Analyze and visualize sentiment distribution"""
    # Calculate sentiment polarity
    df['sentiment'] = df['cleaned_headline'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot distributions for both classes
    sns.histplot(data=df, x='sentiment', hue='is_sarcastic',
                 element='step', stat='density', common_norm=False,
                 palette={0: 'skyblue', 1: 'salmon'}, alpha=0.6, bins=30)

    # Add mean lines
    mean_sentiment_0 = df[df['is_sarcastic'] == 0]['sentiment'].mean()
    mean_sentiment_1 = df[df['is_sarcastic'] == 1]['sentiment'].mean()

    ax.axvline(mean_sentiment_0, color='blue', linestyle='--', label='Mean (Non-Sarcastic)')
    ax.axvline(mean_sentiment_1, color='red', linestyle='--', label='Mean (Sarcastic)')

    plt.title('Sentiment Distribution by Class', fontsize=14)
    plt.xlabel('Sentiment Polarity', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(title='Class', labels=['Non-Sarcastic', 'Sarcastic',
                                      f'Mean (Non-Sarcastic): {mean_sentiment_0:.2f}',
                                      f'Mean (Sarcastic): {mean_sentiment_1:.2f}'])
    plt.tight_layout()

    # Display statistics
    stats = df.groupby('is_sarcastic')['sentiment'].describe().reset_index()
    stats.columns = ['Class', 'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    stats['Class'] = stats['Class'].map({0: 'Non-Sarcastic', 1: 'Sarcastic'})

    st.dataframe(stats[['Class', 'Mean', 'Std', 'Min', '50%', 'Max']].rename(
        columns={'50%': 'Median'}).style.format({'Mean': '{:.2f}', 'Std': '{:.2f}'}))

    return fig


def plot_embedding_visualization(embeddings, labels, sample_size=1000):
    """Visualize embeddings using UMAP dimensionality reduction"""
    # Check if we have enough data
    if len(embeddings) < sample_size:
        st.warning(f"Not enough samples for visualization. Needed: {sample_size}, Available: {len(embeddings)}")
        return None

    # Sample data for faster processing
    if len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings_sample = embeddings[indices]
        labels_sample = labels[indices]
    else:
        embeddings_sample = embeddings
        labels_sample = labels

    # Reduce dimensionality with UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings_sample)

    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'label': labels_sample
    })

    # Map labels to names
    plot_df['Class'] = plot_df['label'].map({0: 'Non-Sarcastic', 1: 'Sarcastic'})

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=plot_df, x='x', y='y', hue='Class',
                    palette={'Non-Sarcastic': 'blue', 'Sarcastic': 'brown'},
                    alpha=0.6, s=20)

    plt.title('ELMo Embedding Visualization (UMAP Projection)', fontsize=14)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.legend(title='Class')
    plt.tight_layout()

    return fig


def plot_word_cloud(df, class_label, title):
    """Generate word cloud for a specific class"""
    text = " ".join(df[df['is_sarcastic'] == class_label]['cleaned_headline'])
    wordcloud = WordCloud(width=800, height=400,
                          background_color='white',
                          max_words=100).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    st.pyplot(fig)


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix with annotations"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                cbar=False, annot_kws={"size": 14})
    ax.set_title(f'Confusion Matrix: {model_name}', fontsize=14)
    ax.set_xlabel('Predicted Labels', fontsize=12)
    ax.set_ylabel('True Labels', fontsize=12)
    ax.xaxis.set_ticklabels(['Not Sarcastic', 'Sarcastic'])
    ax.yaxis.set_ticklabels(['Not Sarcastic', 'Sarcastic'])
    plt.tight_layout()
    return fig


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate machine learning models"""
    results = {}
    test_size = 0.2  # Fixed test size

    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }

    # Train and evaluate each model
    for name, model in models.items():
        with st.spinner(f"Training {name}..."):
            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            metrics = {
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1-Score": f1_score(y_test, y_pred),
                "ROC-AUC": roc_auc_score(y_test, y_proba)
            }

            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            # Store results
            results[name] = {
                "model": model,
                "metrics": metrics,
                "y_pred": y_pred,
                "y_proba": y_proba,
                "confusion_matrix": cm
            }

            # Save models to cache
            joblib.dump(model, MODEL_PATH_LR if "Logistic" in name else MODEL_PATH_RF)

    return results


def plot_roc_curves(results, y_test):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))

    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result["y_proba"])
        roc_auc = result["metrics"]["ROC-AUC"]
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    return plt


def main():
    # Configure Streamlit app
    st.set_page_config(
        page_title="Sarcasm Detection",
        page_icon="🤖",
        layout="wide"
    )

    # Title and description
    st.title("📰 Sarcasm Detection in News Headlines")
    st.markdown("""
    ## Using ELMo Embeddings with Machine Learning Classifiers
    This application detects sarcasm in news headlines using:
    - **ELMo** for contextual word embeddings
    - **Text preprocessing** including cleaning, stopword removal, and lemmatization
    - **Logistic Regression** and **Random Forest** classifiers
    - Evaluation metrics: Precision, Recall, F1-Score, ROC-AUC
    """)

    # Show different message based on cache status
    if os.path.exists(ELMO_LOCAL_PATH) and os.listdir(ELMO_LOCAL_PATH):
        st.info("✅ ELMo model loaded from local cache")
    else:
        st.warning("First run will download ELMo model (~900MB) and may take several minutes")

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # Add retrain button to sidebar
    retrain = st.sidebar.button("Retrain Models", help="Clear cache and retrain models from scratch")

    # Clear cache button
    clear_cache = st.sidebar.button("Clear All Cache", help="Delete all cached models and data")

    # Data upload section
    st.sidebar.header("📂 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Dataset (JSON)",
        type="json",
        help="Upload JSON file in the format: {'headline': text, 'is_sarcastic': 0/1} per line"
    )

    # Initialize session state
    if "models_trained" not in st.session_state:
        st.session_state.models_trained = False
        st.session_state.results = None
        st.session_state.elmo_model = None
        st.session_state.df = None
        st.session_state.embeddings = None
        st.session_state.y_full = None

    # Handle clear cache request
    if clear_cache:
        for path in [CACHE_DIR]:
            if os.path.exists(path):
                for file in os.listdir(path):
                    file_path = os.path.join(path, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        st.error(f"Error deleting {file_path}: {e}")
                st.sidebar.success("All cache cleared")
        st.experimental_rerun()

    # Handle retrain request
    if retrain:
        st.session_state.models_trained = False
        st.session_state.results = None
        st.session_state.embeddings = None
        st.session_state.y_full = None

        # Clear model cache files
        for path in [MODEL_PATH_LR, MODEL_PATH_RF, EMBEDDINGS_PATH]:
            if os.path.exists(path):
                os.remove(path)
                st.sidebar.success(f"Cleared cache: {os.path.basename(path)}")

        st.sidebar.info("Models will be retrained with next action")

    # Data loading section
    st.header("📥 Data Loading")
    if uploaded_file is None:
        st.info("Please upload a JSON dataset using the sidebar")
        st.markdown("""
        ### Required Data Format:
        The JSON file should contain one JSON object per line with:
        ```json
        {"headline": "Your news headline text here", "is_sarcastic": 0}
        ```
        - `headline`: String containing the news headline
        - `is_sarcastic`: Binary label (0 = not sarcastic, 1 = sarcastic)

        Example:
        ```json
        {"headline": "Trump praises democrats for their unity", "is_sarcastic": 1}
        {"headline": "New study shows benefits of regular exercise", "is_sarcastic": 0}
        ```
        """)
        return

    # Load and prepare data
    if st.session_state.df is None:
        with st.spinner("Loading and preprocessing data..."):
            df = load_and_preprocess_data(uploaded_file)
            st.session_state.df = df

    df = st.session_state.df

    # Display dataset info
    st.success(f"✅ Dataset loaded: {len(df):,} records after preprocessing")

    # Display class distribution
    st.header("📊 Data Analysis")
    st.subheader("Class Distribution")

    col1, col2 = st.columns(2)
    with col1:
        plot_class_distribution(df, "After Preprocessing")
    with col2:
        st.markdown("""
        ### Class Balance Information
        - **Not Sarcastic (0):** Headlines with straightforward meaning
        - **Sarcastic (1):** Headlines with sarcastic content

        **Dataset Characteristics:**
        - The original dataset should be approximately balanced
        - Preprocessing may slightly affect distribution
        - Significant imbalance could affect model performance
        """)

    # Word clouds
    st.subheader("Word Frequency Analysis")
    tab1, tab2 = st.tabs(["Non-Sarcastic Headlines", "Sarcastic Headlines"])

    with tab1:
        plot_word_cloud(df, 0, "Frequent Words in Non-Sarcastic Headlines")
    with tab2:
        plot_word_cloud(df, 1, "Frequent Words in Sarcastic Headlines")

    # Display sample data
    st.subheader("Sample Headlines")
    st.dataframe(df[['headline', 'is_sarcastic']].head(10).reset_index(drop=True))

    # Train-test split (fixed test size)
    test_size = 0.2
    X = df["headline"].tolist()
    y = df["is_sarcastic"].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    st.info(f"Split data: {len(X_train)} training samples, {len(X_test)} test samples")

    # Load ELMo model
    st.header("🧠 ELMo Embeddings")
    if st.session_state.elmo_model is None:
        with st.spinner("Loading ELMo model..."):
            st.session_state.elmo_model = load_elmo_model()

    # Generate embeddings
    if not st.session_state.models_trained or st.session_state.embeddings is None:
        with st.spinner("Generating ELMo embeddings..."):
            # Check for cached embeddings
            if os.path.exists(EMBEDDINGS_PATH):
                embeddings = np.load(EMBEDDINGS_PATH)
                st.info("Loaded embeddings from cache")
            else:
                embeddings = elmo_embeddings(st.session_state.elmo_model, X_train + X_test)
                np.save(EMBEDDINGS_PATH, embeddings)
                st.info("Saved embeddings to cache")

            # Store embeddings in session state
            st.session_state.embeddings = embeddings
            st.session_state.y_full = np.concatenate([y_train, y_test])

            # Split embeddings
            train_size = len(X_train)
            X_train_emb = embeddings[:train_size]
            X_test_emb = embeddings[train_size:]

            # Train and evaluate models
            results = train_and_evaluate_models(
                X_train_emb, X_test_emb, y_train, y_test
            )

            st.session_state.results = results
            st.session_state.models_trained = True
            st.success("🎉 Model training completed!")

    # Advanced Data Analysis Section
    st.header("🔍 Advanced Data Analysis")
    analysis_tabs = st.tabs([
        "Text Length Analysis",
        "Sentiment Distribution",
        "Embedding Visualization"
    ])

    with analysis_tabs[0]:
        st.subheader("Text Length Analysis")
        plot_text_length_analysis(df)

    with analysis_tabs[1]:
        st.subheader("Sentiment Distribution")
        plot_sentiment_distribution(df)

    with analysis_tabs[2]:
        st.subheader("Embedding Visualization")
        if st.session_state.embeddings is not None and st.session_state.y_full is not None:
            fig = plot_embedding_visualization(
                st.session_state.embeddings,
                st.session_state.y_full,
                sample_size=2000  # Use a larger sample for better visualization
            )
            if fig:
                st.pyplot(fig)
                st.info("UMAP projection of ELMo embeddings shows how the model represents headlines in 2D space")
            else:
                st.warning("Could not generate embedding visualization")
        else:
            st.warning("Embeddings not available for visualization")

    # Display model evaluation results
    st.header("📈 Model Evaluation")
    if st.session_state.results:
        results = st.session_state.results

        # Metrics comparison table
        st.subheader("Performance Metrics")
        metrics_df = pd.DataFrame(
            {name: res["metrics"] for name, res in results.items()}
        ).T
        st.dataframe(metrics_df.style.format("{:.3f}").highlight_max(axis=0))

        # Confusion matrices
        st.subheader("Confusion Matrices")
        col1, col2 = st.columns(2)

        with col1:
            st.pyplot(plot_confusion_matrix(y_test,
                                            results["Logistic Regression"]["y_pred"],
                                            "Logistic Regression"))

        with col2:
            st.pyplot(plot_confusion_matrix(y_test,
                                            results["Random Forest"]["y_pred"],
                                            "Random Forest"))

        # ROC curve visualization
        st.subheader("ROC Curves")
        roc_plot = plot_roc_curves(results, y_test)
        st.pyplot(roc_plot)

    # Prediction interface
    st.header("🔮 Predict Sarcasm")
    user_input = st.text_area("Enter a news headline:", "Wow, what a surprising turn of events!")

    if st.button("Predict Sarcasm") and st.session_state.results:
        with st.spinner("Analyzing text..."):
            # Preprocess input
            cleaned_input = preprocess_text(user_input)

            # Generate ELMo embedding
            embedding = elmo_embeddings(st.session_state.elmo_model, [cleaned_input])

            # Make predictions
            predictions = {}
            for name, res in st.session_state.results.items():
                proba = res["model"].predict_proba(embedding)[0][1]
                predictions[name] = {
                    "probability": proba,
                    "prediction": "Sarcastic" if proba > 0.5 else "Not Sarcastic"
                }

            # Display results
            st.subheader("Prediction Results")
            st.markdown(f"**Original text:** `{user_input}`")
            st.markdown(f"**Preprocessed text:** `{cleaned_input}`")

            col1, col2 = st.columns(2)
            for name, pred in predictions.items():
                with col1 if "Logistic" in name else col2:
                    # Determine color based on prediction
                    color = "#FF4B4B" if pred["prediction"] == "Sarcastic" else "#0068C9"

                    st.metric(
                        label=name,
                        value=pred["prediction"],
                        delta=f"{pred['probability']:.2%} confidence",
                        delta_color="normal"
                    )

                    # Visual confidence indicator
                    confidence = pred["probability"] if pred["prediction"] == "Sarcastic" else 1 - pred["probability"]
                    st.progress(float(confidence))

    # Deployment information
    st.markdown("---")
    st.subheader("🚀 Deployment Information")
    st.info("""
    - **Cloud Infrastructure**: Streamlit Community Cloud
    - **Embedding**: ELMo (Contextual Word Embeddings)
    - **Text Preprocessing**: Lowercasing, special character removal, stopword removal, lemmatization
    - **Classifiers**: Logistic Regression, Random Forest
    - **Evaluation Metrics**: Precision, Recall, F1-Score, ROC-AUC
    """)
    st.caption("Group 9 Project | Sarcasm and Toxicity Detection in Reddit Comments Using ELMo")


if __name__ == "__main__":
    main()
