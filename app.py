import streamlit as st
import joblib
from groq import Groq
import os
from dotenv import load_dotenv

# Page setup
st.set_page_config(page_title="AI Customer Support System", layout="wide")

# Sidebar
st.sidebar.title("System Details")
st.sidebar.write("ML Model: Logistic Regression")
st.sidebar.write("Vectorizer: TF-IDF")
st.sidebar.write("LLM: Groq (Llama 3.3)")
st.sidebar.write("Dataset: Consumer Complaints")

# Session state
if "complaint_text" not in st.session_state:
    st.session_state.complaint_text = ""

# Environment
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load model
model = joblib.load("complaint_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Main UI
st.title("AI-Powered Customer Complaint Analyzer")
st.write("Enter a customer complaint below to classify and analyze it.")

user_input = st.text_area(
    "Customer Complaint",
    value=st.session_state.complaint_text,
    key="complaint_box"
)

# Buttons
col1, col2 = st.columns(2)
analyze_clicked = False

with col1:
    if st.button("Analyze Complaint", use_container_width=True):
        analyze_clicked = True

with col2:
    if st.button("Clear Complaint", use_container_width=True):
        st.session_state.complaint_text = ""
        st.rerun()

# Prediction
if analyze_clicked:
    if user_input.strip() == "":
        st.warning("Please enter a complaint.")
    else:
        st.session_state.complaint_text = user_input
        input_vector = vectorizer.transform([user_input])

        prediction = model.predict(input_vector)[0]
        probabilities = model.predict_proba(input_vector)[0]
        confidence = max(probabilities) * 100

        st.subheader("Predicted Category")
        st.success(prediction)

        st.subheader("Confidence Score")
        st.info(f"{confidence:.2f}%")

        prompt = f"""
You are an AI customer support assistant.

A machine learning model classified this complaint as: {prediction}
Confidence score: {confidence:.2f}%

Customer complaint:
{user_input}

Tasks:
1. Identify severity level: Low / Medium / High / Critical
2. Explain issue briefly
3. Give support response
4. Suggest next steps

Output format:

Severity: <level>

Issue Summary:
<summary>

Response:
<response>

Next Steps:
<steps>
"""

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile"
        )

        reply = chat_completion.choices[0].message.content

        st.subheader("AI Analysis")
        st.write(reply)