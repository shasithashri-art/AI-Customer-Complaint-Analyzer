# AI-Powered Customer Complaint Analyzer

This project focuses on analyzing customer complaints using Machine Learning and AI. The goal is to automatically identify the complaint category and generate an intelligent support response that helps customer service teams handle issues faster.

I built this project by combining a machine learning classifier with a Large Language Model (LLM). The machine learning model predicts the complaint category, while the LLM analyzes the complaint severity and generates an appropriate response with suggested next steps.

## Main Features
- Classifies customer complaints into different categories
- Displays prediction confidence score
- Analyzes complaint severity
- Generates AI-based support response
- Suggests next actions for resolution
- Interactive web application using Streamlit

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Streamlit
- Groq API (LLM)

## Dataset
The project uses a customer complaint dataset containing complaint narratives and their corresponding product categories.

Supported categories:
- Credit Card
- Retail Banking
- Credit Reporting
- Mortgages and Loans
- Debt Collection

## Model Performance
I tested multiple machine learning models and selected Logistic Regression as the final model because it performed better than Linear SVM.

Final Accuracy: **84.89%**

## Project Workflow
1. Load and clean complaint data  
2. Convert complaint text into numerical features using TF-IDF  
3. Train machine learning model  
4. Predict complaint category  
5. Send complaint to LLM for deeper analysis  
6. Generate severity, response, and next steps  

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
- Better UI design
- Database integration
- Complaint history tracking
- Cloud deployment
- 
## Project Link
[AI Customer Complaint Analyzer](https://github.com/shasithashri-art/AI-Customer-Complaint-Analyzer)
