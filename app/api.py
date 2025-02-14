import sys

sys.path.append('../')


from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from enum import Enum
from model_and_nlp_staff.response import Response, get_svm_predicts
from pprint import pprint as pp
from fastapi.responses import RedirectResponse
import numpy as np

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()


# Input model for receiving email content
class EmailInput(BaseModel):
    text: str

# Enum for the categories
# class CategoryName(str, Enum):
#     generalSupport = "General Support"
#     fylAirlinesIssues = "Fly / Airline Issues"
#     orderPaymentIssues = "Order / Payment Issues"
#     retailStoreComplaints = "Retail / Grocery Store Complaints"
#     techSupport = "Tech Support"
    

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


# 1) API Endpoint to classify email into a category using SVM model
# API endpoint


# @app.post("/predict/")
# def predict_category(email: EmailInput):
#     try:
#         logger.debug("Received email text: %s", email.text)
#         topic_name = get_svm_predicts(email.text) 
#         logger.debug("SVM predicted category: %s", topic_name)
#         logger.debug("Type of topic_name: %s", type(topic_name))
#         return {"category": topic_name}
#     except Exception as e:
#         logger.error("Error in predict_category: %s", str(e), exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))


# 2) API Endpoint to generate a response based on the email's category using LLaMa model
@app.post("/predict/response")
def predict_response(email: EmailInput):
    try:
        logger.debug("Received email text: %s", email.text)
        topic_name = get_svm_predicts(email.text) 
        logger.debug("SVM predicted category: %s", topic_name)
        logger.debug("Type of topic_name: %s", type(topic_name))
        
        user_email = f"""
            You must respond to the email strictly based on the given category. 

            Category: {topic_name}

            Email: {email.text}

            Do not change or reanalyze the category. Only use the provided category for your response.
        """
        
        print("Generating response for email:", user_email)
        client = Response()
        response = client.get_response(user_email)
        print("Generated response:", response)
        
        return {"response": response}
    except Exception as e:
        logger.error("Error in predict_response: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    
    
# # 3) API Endpoint to fetch category details (optional for documentation or user guidance)
# @app.get("/categories/{category_name}")
# async def get_category(category_name: CategoryName):
#     if category_name == CategoryName.generalSupport:
#         return {"category_name": category_name, "message": "General support inquiries"}
#     if category_name == CategoryName.fylAirlinesIssues:
#         return {"category_name": category_name, "message": "Issues related to flights or airlines"}
#     if category_name == CategoryName.orderPaymentIssues:
#         return {"category_name": category_name, "message": "Problems with orders or payments"}
#     if category_name == CategoryName.retailStoreComplaints:
#         return {"category_name": category_name, "message": "Complaints about retail or grocery stores"}
#     if category_name == CategoryName.techSupport:
#         return {"category_name": category_name, "message": "Technical support requests"}
#     return {"category_name": category_name, "message": "Unknown category"}
