import joblib
from response import Response
from groq import Groq

class System:
  
  def __init__(self):
    """
    Initializes the system with a default prompt and a Groq client.

    Attributes:
      prompt (list): A list containing a dictionary with the role and content of the user email.
      client (Groq): An instance of the Groq client initialized with an API key.
    """
    self.prompt = [{"role": "user", "content": self.user_email}]
    self.client = Groq(api_key='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')
  
  
  def get_classification_model(self):
    return joblib.load(r'C:\Users\Victus\Desktop\AI Email Assistant\models\xgboost_email_classifier.pkl')
  
  
  @property
  def user_email(self):
    """
    Property that returns the user's email address.

    Returns:
      str: The user's email address.
    """
    return """
            first you must categorize the email into one of the following categories:
            1) General Support 
            2) Fly / Airline Issues
            3) Order / Payment Issues
            4) Retail / Grocery Store Complaints 
            5) Tech Support

            Email: I have an issue with my pc and I need help. Issue is my RAM is broken.

            After you have categorized the email, you can then proceed to respond to the email.
          """


  def get_response(self):
    """
    Generates a response from the chat model based on the provided prompt.
    This method sends a request to the chat model with the specified parameters
    and streams the response in chunks. The response is printed to the console
    as it is received.
    Parameters:
    None
    Returns:
    None
    """
    
    completion = self.client.chat.completions.create(
      model="llama-3.1-8b-instant",
      messages=self.prompt,
      temperature=1,
      max_completion_tokens=1024,
      top_p=1,
      stream=True,
      stop=None,
    )
    for chunk in completion:
      print(chunk.choices[0].delta.content or "", end="")







if __name__ == "__main__":
  System().get_response()

