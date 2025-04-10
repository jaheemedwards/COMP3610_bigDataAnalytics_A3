# Sentiment Classification (Logistic Regression)
# Implement your sentiment classification model here
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def add_sentiment_label(df):
    df["sentiment"] = df["rating"].apply(lambda x: 1 if x > 3 else 0)
    return df

def split_sentiment_data(df):
    return train_test_split(df["text"], df["sentiment"], test_size=0.2, random_state=42)

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
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def run_sentiment_analysis(df):
    df = add_sentiment_label(df)
    X_train, X_test, y_train, y_test = split_sentiment_data(df)
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(X_train, X_test)
    model = train_logistic_regression(X_train_tfidf, y_train)
    evaluate_model(model, X_test_tfidf, y_test)


