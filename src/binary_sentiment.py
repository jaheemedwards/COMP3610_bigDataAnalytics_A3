from data_acquisition import *
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def add_sentiment_label(df):
    df["sentiment"] = df["rating"].apply(lambda x: 1 if x > 3 else 0)
    return df

def stratified_sample(df, frac=0.05):
    return df.groupby('sentiment', group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=42))

import pyarrow.dataset as ds

def load_and_sample_data():
    print("🔹 Loading and stratified sampling data...")
    base_path = r"D:\Jaheem\jaheemProjetToDelete\COMP3610_bigDataAnalytics_A3-master\src\data\cleaned"
    paths = [fr"{base_path}\cleaned_{category}.parquet" for category in VALID_CATEGORIES]

    all_samples = []
    for path, category in zip(paths, VALID_CATEGORIES):
        print(f"   - Reading and sampling 5% of data for category: {category}")

        # Use PyArrow scanner to read in a subset of rows
        dataset = ds.dataset(path, format="parquet")
        scanner = dataset.scanner(columns=["text", "rating"])
        table = scanner.head(100_000)  # Only load first 100,000 rows for memory safety
        df = table.to_pandas()

        df = add_sentiment_label(df)
        df_sample = stratified_sample(df, frac=0.05)
        all_samples.append(df_sample)

    combined_df = pd.concat(all_samples, ignore_index=True)
    return combined_df

def split_sentiment_data(df):
    return train_test_split(df["text"], df["sentiment"], test_size=0.2, stratify=df["sentiment"], random_state=42)

def vectorize_text(X_train, X_test):
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.8)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer

def train_logistic_regression(X_train_tfidf, y_train):
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    return model

def evaluate_model(model, X_test_tfidf, y_test):
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("✅ Evaluation Results")
    print("   - Accuracy:", acc)
    print("   - F1 Score:", f1)
    print("   - Confusion Matrix:\n", cm)

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix Heatmap')
    plt.tight_layout()
    plt.show()

    # Plot Accuracy and F1 Score
    plt.figure(figsize=(6, 4))
    plt.bar(["Accuracy", "F1 Score"], [acc, f1], color='skyblue')
    plt.ylim(0, 1)
    plt.title("Model Performance Metrics")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.show()

# Run sentiment classification pipeline
print("🔹 Starting sentiment classification pipeline...")
df = load_and_sample_data()
X_train, X_test, y_train, y_test = split_sentiment_data(df)
X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(X_train, X_test)
model = train_logistic_regression(X_train_tfidf, y_train)
evaluate_model(model, X_test_tfidf, y_test)

# Save model and vectorizer
model_dir = r"D:\Jaheem\jaheemProjetToDelete\COMP3610_bigDataAnalytics_A3-master\src\data\processed\models"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "sentiment_logistic_model.pkl")
vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print(f"📦 Sentiment model saved to: {model_path}")
print(f"📦 TF-IDF vectorizer saved to: {vectorizer_path}")
print("✅ Sentiment analysis completed.")