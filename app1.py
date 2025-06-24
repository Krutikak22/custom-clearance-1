import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from joblib import dump

# 🚨 Replace these with your actual CSV file and column names
df = pd.read_csv("your_dataset.csv")
texts = df["text_column_name"]
labels = df["label_column_name"]

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)
y = labels

# Train/test split
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Save everything in this same environment
dump(vectorizer, "tfidf_vectorizer.pkl")
dump(rf_model, "random_forest.pkl")
dump(dt_model, "decision_tree.pkl")
dump(svm_model, "svm_model.pkl")

print("✅ Models trained and saved successfully.")
