import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from joblib import dump

# Load your dataset
df = pd.read_csv("your_dataset.csv")
texts = df["text_column_name"]        # Replace with your actual column
labels = df["label_column_name"]      # Replace with your actual column

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)
y = labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

svm = SVC(probability=True, random_state=42)
svm.fit(X_train, y_train)

# Save models and vectorizer
dump(vectorizer, "tfidf_vectorizer.pkl")
dump(rf, "random_forest.pkl")
dump(dt, "decision_tree.pkl")
dump(svm, "svm_model.pkl")
