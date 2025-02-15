import streamlit as st
import requests
from model_and_nlp_staff.response import Response

# Fastapi endpoint
API_URL = "http://localhost:8000/predict/response/"


def main():
    st.title("Email Response Generator")
    st.write("This app generates a response based on the email's category using the LLaMa model.")

    email_text = st.text_area("Enter email text here", height=200)
    if st.button("Generate Response"):
        response = Response.get_response(email_text)
        st.write("Response:", response)



