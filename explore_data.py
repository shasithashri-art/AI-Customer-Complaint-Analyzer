import pandas as pd

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("complaints.csv")

# Drop useless column
df = df.drop(columns=["Unnamed: 0"])

print("Before cleaning:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

# Remove missing rows
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()

print("\nAfter cleaning:")
print(df.shape)

print("\nUnique products:")
print(df["product"].unique())

print("\nNumber of categories:")
print(df["product"].nunique())

print("\nCategory counts:")
print(df["product"].value_counts())


# =========================
# TRAIN TEST SPLIT
# =========================
from sklearn.model_selection import train_test_split

X = df["narrative"]   # input text
y = df["product"]     # target labels

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining set size:", len(X_train))
print("Testing set size:", len(X_test))


# =========================
# TF-IDF VECTORIZATION
# =========================
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF shapes:")
print("Training:", X_train_tfidf.shape)
print("Testing:", X_test_tfidf.shape)


# =========================
# LOGISTIC REGRESSION
# =========================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train_tfidf, y_train)

lr_pred = lr_model.predict(X_test_tfidf)

lr_accuracy = accuracy_score(y_test, lr_pred)

print("\n=========================")
print("LOGISTIC REGRESSION")
print("=========================")
print("Accuracy:")
print(lr_accuracy)

print("\nClassification Report:")
print(classification_report(y_test, lr_pred))


# =========================
# LINEAR SVM
# =========================
from sklearn.svm import LinearSVC

svm_model = LinearSVC()

svm_model.fit(X_train_tfidf, y_train)

svm_pred = svm_model.predict(X_test_tfidf)

svm_accuracy = accuracy_score(y_test, svm_pred)

print("\n=========================")
print("LINEAR SVM")
print("=========================")
print("Accuracy:")
print(svm_accuracy)

print("\nClassification Report:")
print(classification_report(y_test, svm_pred))


# =========================
# COMPARE MODELS
# =========================
print("\n=========================")
print("MODEL COMPARISON")
print("=========================")
print("Logistic Regression Accuracy:", lr_accuracy)
print("Linear SVM Accuracy:", svm_accuracy)


# =========================
# CONFUSION MATRIX (SVM)
# =========================
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, svm_pred, labels=svm_model.classes_)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=svm_model.classes_,
    yticklabels=svm_model.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Confusion Matrix")
plt.show()

import joblib

joblib.dump(lr_model, "complaint_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully!")