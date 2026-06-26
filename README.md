# AI-Powered Customer Complaint Analyzer

An AI-powered web application that analyzes customer complaints, classifies them into relevant categories, and generates intelligent support responses.

This project combines **Machine Learning** and a **Large Language Model (LLM)** to help customer support teams handle complaints more efficiently. The ML model predicts the complaint category, while the LLM analyzes issue severity, summarizes the problem, and suggests appropriate next steps.

## Live Demo
Streamlit App: https://ai-customer-complaint-analyzer-ceclpyehsx5utsxn86whx4.streamlit.app/

## Main Features
- Classifies customer complaints into multiple financial service categories
- Displays prediction confidence score
- Detects complaint severity (Low / Medium / High / Critical)
- Generates AI-based issue summary and support response
- Suggests next steps for resolution
- Interactive web interface built with Streamlit

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Streamlit
- Groq API (LLM Integration)

## Dataset
The model was trained using a customer complaint dataset containing complaint narratives and their corresponding categories.

Supported categories include:
- Credit Card
- Retail Banking
- Credit Reporting
- Mortgages and Loans
- Debt Collection

## Model Performance
Multiple machine learning models were evaluated, including **Logistic Regression** and **Linear SVM**.  
Logistic Regression was selected as the final model based on overall performance.

**Final Accuracy: 84.89%**

## Project Workflow
1. Load and preprocess complaint data  
2. Convert complaint text into numerical features using TF-IDF  
3. Train the classification model  
4. Predict complaint category  
5. Send complaint to LLM for deeper analysis  
6. Generate severity, support response, and recommended actions  

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Future Improvements
- Improve classification accuracy using advanced NLP models (BERT / Transformers)
- Store complaint history using database integration
- Add analytics dashboard for support teams
