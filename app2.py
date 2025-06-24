from joblib import load

# Load the vectorizer and a model
vectorizer = load('tfidf_vectorizer.pkl')
rf_model = load('random_forest.pkl')
dt_model = load('decision_tree.pkl')
svm_model = load('svm_model.pkl')

# Transform new data
new_texts = ["Your new input text here"]
new_X = vectorizer.transform(new_texts)

# Predict using any of the models
rf_predictions = rf_model.predict(new_X)
print("Random Forest predictions:", rf_predictions)
