import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from joblib import dump

# ✅ Load your dataset
df = pd.read_csv("your_dataset.csv")

# ✅ Replace with your column names
X_text = df["text"]
y = df["label"]

# ✅ Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X_text)

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Logistic Regression model (simple, fast, accurate for text)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ✅ Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Model Accuracy: {accuracy:.2f}")

# ✅ Save safely (no node dtype issue)
dump(model, "logreg_model.pkl")
dump(vectorizer, "vectorizer.pkl")
print("✅ Model and vectorizer saved.")
