# ═══════════════════════════════════════════════════════════════
# SMS Spam Detection using NLP and Machine Learning
# ═══════════════════════════════════════════════════════════════
# College NLP & ML Project
# Model: Multinomial Naive Bayes with TF-IDF Vectorization
# Dataset: UCI SMS Spam Collection (5,572 messages)
# ═══════════════════════════════════════════════════════════════

import os
import re
import string
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

warnings.filterwarnings("ignore")

# ─── Download required NLTK data ───
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# ═══════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════
def load_dataset(filepath: str = "spam.csv") -> pd.DataFrame:
    """Load the SMS Spam Collection dataset."""
    # The UCI dataset is typically tab-separated or encoded in latin-1
    try:
        df = pd.read_csv(filepath, encoding="latin-1")
    except FileNotFoundError:
        print(f"[!] Dataset file '{filepath}' not found.")
        print("    Downloading UCI SMS Spam Collection...")
        import urllib.request

        url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
        urllib.request.urlretrieve(url, "sms.tsv")
        df = pd.read_csv("sms.tsv", sep="\t", header=None, names=["label", "message"])
        df.to_csv("spam.csv", index=False)
        print("    ✅ Dataset downloaded and saved as spam.csv")
        return df

    # Handle various column naming conventions
    if "v1" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "message"})
        df = df[["label", "message"]]
    elif "label" not in df.columns:
        df.columns = ["label", "message"] + list(df.columns[2:])
        df = df[["label", "message"]]

    df.dropna(subset=["message"], inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ═══════════════════════════════════════════
# 2. DATA EXPLORATION
# ═══════════════════════════════════════════
def explore_data(df: pd.DataFrame) -> None:
    """Print key statistics about the dataset."""
    print("=" * 55)
    print("  📊 DATASET OVERVIEW")
    print("=" * 55)
    print(f"  Total messages    : {len(df)}")
    print(f"  Ham messages      : {(df['label'] == 'ham').sum()}")
    print(f"  Spam messages     : {(df['label'] == 'spam').sum()}")
    print(f"  Spam percentage   : {(df['label'] == 'spam').mean() * 100:.1f}%")
    print(f"  Missing values    : {df.isnull().sum().sum()}")
    print(f"  Duplicate rows    : {df.duplicated().sum()}")
    print("=" * 55)
    print()


# ═══════════════════════════════════════════
# 3. TEXT PREPROCESSING (NLP Pipeline)
# ═══════════════════════════════════════════
class TextPreprocessor:
    """NLP text preprocessing pipeline."""

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))

    def clean_text(self, text: str) -> str:
        """
        Full NLP preprocessing pipeline:
        1. Convert to lowercase
        2. Remove special characters & numbers
        3. Remove punctuation
        4. Tokenize
        5. Remove stopwords
        6. Apply stemming
        """
        # Step 1: Lowercase
        text = text.lower()

        # Step 2: Remove special characters and numbers
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Step 3: Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Step 4: Tokenize
        tokens = text.split()

        # Step 5 & 6: Remove stopwords + Stemming
        tokens = [
            self.stemmer.stem(word)
            for word in tokens
            if word not in self.stop_words and len(word) > 2
        ]

        return " ".join(tokens)

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing to entire DataFrame."""
        print("   Cleaning text...")
        df["cleaned_message"] = df["message"].apply(self.clean_text)

        # Add feature columns
        df["msg_length"] = df["message"].apply(len)
        df["word_count"] = df["message"].apply(lambda x: len(x.split()))

        print("   Text preprocessing complete!")
        return df


# ═══════════════════════════════════════════
# 4. FEATURE EXTRACTION (TF-IDF)
# ═══════════════════════════════════════════
def extract_features(
    X_train: pd.Series, X_test: pd.Series, max_features: int = 5000
):
    """
    Convert text to TF-IDF feature vectors.

    TF-IDF = Term Frequency × Inverse Document Frequency
    - Measures how important a word is to a document in the corpus
    - Words that appear frequently in one document but rarely in others get higher scores
    """
    print("   Extracting TF-IDF features...")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=2,  # Minimum document frequency
        max_df=0.95,  # Maximum document frequency
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"   Feature matrix shape: {X_train_tfidf.shape}")
    print(f"     Vocabulary size: {len(vectorizer.vocabulary_)}")

    return vectorizer, X_train_tfidf, X_test_tfidf


# ═══════════════════════════════════════════
# 5. MODEL TRAINING
# ═══════════════════════════════════════════
def train_model(X_train_tfidf, y_train):
    """Train a Multinomial Naive Bayes classifier."""
    print("\n  🤖 Training Multinomial Naive Bayes model...")

    model = MultinomialNB(alpha=0.1)  # Laplace smoothing
    model.fit(X_train_tfidf, y_train)

    # Training accuracy
    train_pred = model.predict(X_train_tfidf)
    train_acc = accuracy_score(y_train, train_pred) * 100
    print(f"  ✅ Training Accuracy: {train_acc:.2f}%")

    return model


# ═══════════════════════════════════════════
# 6. MODEL EVALUATION
# ═══════════════════════════════════════════
def evaluate_model(model, X_test_tfidf, y_test):
    """Evaluate model performance on test set."""
    print("\n" + "=" * 55)
    print("  📋 MODEL EVALUATION RESULTS")
    print("=" * 55)

    y_pred = model.predict(X_test_tfidf)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, pos_label="spam") * 100
    recall = recall_score(y_test, y_pred, pos_label="spam") * 100
    f1 = f1_score(y_test, y_pred, pos_label="spam") * 100

    print(f"  Accuracy  : {accuracy:.2f}%")
    print(f"  Precision : {precision:.2f}%")
    print(f"  Recall    : {recall:.2f}%")
    print(f"  F1-Score  : {f1:.2f}%")
    print()

    # Classification Report
    print("  Classification Report:")
    print("  " + "-" * 50)
    report = classification_report(y_test, y_pred, target_names=["Ham", "Spam"])
    for line in report.split("\n"):
        print(f"  {line}")
    print()

    return y_pred, {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ═══════════════════════════════════════════
# 7. VISUALIZATION
# ═══════════════════════════════════════════
def plot_confusion_matrix(y_test, y_pred, save_path: str = "confusion_matrix.png"):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
        ax=ax,
        linewidths=1,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix — SMS Spam Detection", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Confusion matrix saved to: {save_path}")
    plt.close()


def plot_class_distribution(df: pd.DataFrame, save_path: str = "class_distribution.png"):
    """Plot class distribution of the dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    counts = df["label"].value_counts()
    colors = ["#00B894", "#E17055"]
    counts.plot(kind="bar", ax=axes[0], color=colors, edgecolor="white", linewidth=2)
    axes[0].set_title("Message Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Label")
    axes[0].set_ylabel("Count")
    axes[0].set_xticklabels(["Ham", "Spam"], rotation=0)

    # Pie chart
    counts.plot(
        kind="pie",
        ax=axes[1],
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        labels=["Ham", "Spam"],
        textprops={"fontsize": 11},
    )
    axes[1].set_title("Spam vs Ham Ratio", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"   Class distribution saved to: {save_path}")
    plt.close()


def plot_message_length(df: pd.DataFrame, save_path: str = "message_length.png"):
    """Plot message length distribution by label."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, color in [("ham", "#00B894"), ("spam", "#E17055")]:
        subset = df[df["label"] == label]
        ax.hist(
            subset["msg_length"],
            bins=50,
            alpha=0.6,
            color=color,
            label=label.capitalize(),
            edgecolor="white",
        )

    ax.set_title("Message Length Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Message Length (characters)")
    ax.set_ylabel("Frequency")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  📊 Message length plot saved to: {save_path}")
    plt.close()


# ═══════════════════════════════════════════
# 8. PREDICTION FUNCTION
# ═══════════════════════════════════════════
def predict_message(message: str, model, vectorizer, preprocessor) -> dict:
    """
    Predict whether a single SMS message is Spam or Ham.

    Returns:
        dict with 'prediction', 'confidence', 'probabilities'
    """
    # Preprocess
    cleaned = preprocessor.clean_text(message)

    # Vectorize
    features = vectorizer.transform([cleaned])

    # Predict
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    # Get class labels
    classes = model.classes_
    prob_dict = {cls: prob * 100 for cls, prob in zip(classes, probabilities)}

    confidence = max(probabilities) * 100

    return {
        "prediction": prediction.upper(),
        "confidence": confidence,
        "probabilities": prob_dict,
        "cleaned_text": cleaned,
    }


# ═══════════════════════════════════════════
# 9. SAVE / LOAD MODEL
# ═══════════════════════════════════════════
def save_model(model, vectorizer, preprocessor, directory: str = "model"):
    """Save trained model, vectorizer, and preprocessor to disk."""
    os.makedirs(directory, exist_ok=True)

    with open(os.path.join(directory, "spam_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(directory, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(directory, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)

    print(f"\n  💾 Model saved to '{directory}/' directory")


def load_saved_model(directory: str = "model"):
    """Load previously saved model artifacts."""
    with open(os.path.join(directory, "spam_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(directory, "tfidf_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(directory, "preprocessor.pkl"), "rb") as f:
        preprocessor = pickle.load(f)

    return model, vectorizer, preprocessor


# ═══════════════════════════════════════════
# 10. INTERACTIVE DEMO
# ═══════════════════════════════════════════
def interactive_demo(model, vectorizer, preprocessor):
    """Run an interactive SMS spam detection demo in the terminal."""
    print("\n" + "=" * 55)
    print("  🛡️  SMS SPAM DETECTION — Interactive Demo")
    print("=" * 55)
    print("  Type an SMS message to check if it's Spam or Ham.")
    print("  Type 'quit' or 'exit' to stop.\n")

    while True:
        message = input("  📩 Enter SMS: ").strip()
        if message.lower() in ("quit", "exit", "q"):
            print("\n  👋 Goodbye!")
            break
        if not message:
            print("  ⚠️  Please enter a message.\n")
            continue

        result = predict_message(message, model, vectorizer, preprocessor)

        print()
        if result["prediction"] == "SPAM":
            print(f"  ⚠️  Result   : 🚫 {result['prediction']}")
        else:
            print(f"  ✅  Result   : ✉️  {result['prediction']}")

        print(f"  📊  Confidence: {result['confidence']:.1f}%")
        print(f"  📈  Ham prob  : {result['probabilities'].get('ham', 0):.1f}%")
        print(f"  📉  Spam prob : {result['probabilities'].get('spam', 0):.1f}%")
        print()


# ═══════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════
def main():
    print()
    print("╔" + "═" * 53 + "╗")
    print("║  🛡️  SMS SPAM DETECTION                            ║")
    print("║  NLP & Machine Learning Project                    ║")
    print("╚" + "═" * 53 + "╝")
    print()

    # Step 1: Load data
    print("━" * 55)
    print("  STEP 1: Loading Dataset")
    print("━" * 55)
    df = load_dataset()
    explore_data(df)

    # Step 2: Preprocess text
    print("━" * 55)
    print("  STEP 2: Text Preprocessing (NLP Pipeline)")
    print("━" * 55)
    preprocessor = TextPreprocessor()
    df = preprocessor.preprocess_dataframe(df)
    print()

    # Step 3: Split data
    print("━" * 55)
    print("  STEP 3: Train/Test Split")
    print("━" * 55)
    X = df["cleaned_message"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Training set: {len(X_train)} messages")
    print(f"  Test set    : {len(X_test)} messages")
    print()

    # Step 4: Feature extraction
    print("━" * 55)
    print("  STEP 4: TF-IDF Feature Extraction")
    print("━" * 55)
    vectorizer, X_train_tfidf, X_test_tfidf = extract_features(X_train, X_test)
    print()

    # Step 5: Train model
    print("━" * 55)
    print("  STEP 5: Model Training")
    print("━" * 55)
    model = train_model(X_train_tfidf, y_train)

    # Step 6: Evaluate
    print("━" * 55)
    print("  STEP 6: Model Evaluation")
    print("━" * 55)
    y_pred, metrics = evaluate_model(model, X_test_tfidf, y_test)

    # Step 7: Visualizations
    print("━" * 55)
    print("  STEP 7: Generating Visualizations")
    print("━" * 55)
    plot_confusion_matrix(y_test, y_pred)
    plot_class_distribution(df)
    plot_message_length(df)
    print()

    # Step 8: Save model
    print("━" * 55)
    print("  STEP 8: Saving Model")
    print("━" * 55)
    save_model(model, vectorizer, preprocessor)

    # Step 9: Quick test predictions
    print("\n" + "━" * 55)
    print("  STEP 9: Sample Predictions")
    print("━" * 55)

    test_messages = [
        "Congratulations! You have won a free prize. Click here to claim now.",
        "Hi, are you coming to college today?",
        "WINNER!! You've been selected to receive a $900 prize reward!",
        "Hey, I'll be late for the meeting. Start without me.",
        "Free entry in a weekly comp for a £100 gift voucher. Text WIN to 80086.",
        "Can you send me the notes from yesterday's lecture?",
    ]

    for msg in test_messages:
        result = predict_message(msg, model, vectorizer, preprocessor)
        icon = "🚫" if result["prediction"] == "SPAM" else "✅"
        print(f"\n  {icon} [{result['prediction']:4s}] ({result['confidence']:.1f}%)")
        print(f"     \"{msg[:60]}{'...' if len(msg) > 60 else ''}\"")

    print("\n" + "=" * 55)
    print("  ✅ All steps completed successfully!")
    print("=" * 55)

    # Step 10: Interactive demo
    interactive_demo(model, vectorizer, preprocessor)


if __name__ == "__main__":
    main()
