import joblib

# Load saved files
model = joblib.load("complaint_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# New complaint
user_input = ["My credit card was charged twice for the same transaction"]

# Convert text to TF-IDF
input_tfidf = vectorizer.transform(user_input)

# Predict
prediction = model.predict(input_tfidf)

print("Predicted category:", prediction[0])