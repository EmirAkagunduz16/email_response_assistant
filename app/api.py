# api.py
import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from pydantic import BaseModel
from model_and_nlp_staff.response import Response, get_svm_predicts

app = FastAPI()

## My API structure

# 1) Get email from user

# Giriş verisi için model  # Why this is needed?
class EmailInput(BaseModel):
    text: str
    
# 2) Use SVM model to predict the email categories

# API endpoint
@app.post("/predict/")
def predict_category(email: EmailInput):
    print("Received request:", email.text)  # Debugging output
    
    try:
        topic_name = get_svm_predicts(email.text)
        print(f"SVM predicted category: {topic_name}")
    except Exception as e:
        print(f"Error in get_svm_predicts: {e}")
        return {"error": str(e)}

    return {"category": topic_name}


# 3) Use LLaMa model to response the email to user

@app.post("/predict/response")
def predict_response(email: EmailInput, topic_name: str):
    user_email = f"""
        You must respond to the email strictly based on the given category. 

        Category: {topic_name}

        Email: {email}

        Do not change or reanalyze the category. Only use the provided category for your response.
    """
        
    client = Response()
    response = client.get_response(user_email)
    print(response)
    
    return {"response": response}









