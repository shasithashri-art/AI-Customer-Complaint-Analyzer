import streamlit as st
import joblib
from groq import Groq
import os
from dotenv import load_dotenv

# Page setup
st.set_page_config(
    page_title="AI Customer Support System",
    layout="wide"
)

# Sidebar
st.sidebar.title("System Details")
st.sidebar.write("ML Model: Logistic Regression")
st.sidebar.write("Vectorizer: TF-IDF")
st.sidebar.write("LLM: Groq (GPT-OSS 20B)")
st.sidebar.write("Dataset: Consumer Complaints")

# Session state
if "complaint_text" not in st.session_state:
    st.session_state.complaint_text = ""

# Environment
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY is not configured.")
    st.stop()

client = Groq(api_key=groq_api_key)

# Load ML model and vectorizer
@st.cache_resource
def load_models():
    model = joblib.load("complaint_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_models()

# Main UI
st.title("AI-Powered Customer Complaint Analyzer")
st.write("Enter a customer complaint below to classify and analyze it.")

# Complaint input
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

# Prediction and AI analysis
if analyze_clicked:

    if user_input.strip() == "":
        st.warning("Please enter a complaint.")

    else:
        st.session_state.complaint_text = user_input

        # -----------------------------
        # ML PREDICTION
        # -----------------------------

        input_vector = vectorizer.transform([user_input])

        prediction = model.predict(input_vector)[0]

        probabilities = model.predict_proba(input_vector)[0]

        confidence = max(probabilities) * 100

        # Clean label
        display_prediction = prediction.replace("_", " ").title()

        # Display prediction
        st.subheader("Predicted Category")
        st.success(display_prediction)

        # Display confidence
        st.subheader("Confidence Score")
        st.info(f"{confidence:.2f}%")

        # Low confidence warning
        if confidence < 70:
            st.warning(
                "⚠️ Low confidence prediction. "
                "This complaint may require manual review by a human agent."
            )

        # -----------------------------
        # GROQ PROMPT
        # -----------------------------

        prompt = f"""
You are an AI customer support assistant.

A machine learning model classified this customer complaint as:

Category: {display_prediction}
Confidence Score: {confidence:.2f}%

Customer Complaint:
{user_input}

Analyze the complaint and provide:

1. Severity level: Low, Medium, High, or Critical
2. A brief explanation of the issue
3. A professional and empathetic customer support response
4. Clear next steps for resolving the complaint

Use exactly this format:

Severity: <Low / Medium / High / Critical>

Issue Summary:
<brief summary>

Response:
<professional customer support response>

Next Steps:
<clear next steps>
"""

        # -----------------------------
        # GROQ LLM
        # -----------------------------

        try:

            chat_completion = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_completion_tokens=1000
            )

            reply = chat_completion.choices[0].message.content

            # Display AI analysis
            st.subheader("AI Analysis")
            st.write(reply)

        except Exception as e:

            st.error(
                "Unable to generate AI analysis. "
                "Please check the Groq API configuration."
            )

            st.exception(e)