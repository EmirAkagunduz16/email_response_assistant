from groq import Groq
import joblib
from preprocessing import preprocess_email

   
class Response:
    
    def __init__(self):
        self.client = Groq(api_key='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')
    
    def get_response(self, user_email):
        completion = self.client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": user_email}],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
        )
        for chunk in completion:
            print(chunk.choices[0].delta.content or "", end="")
            
    
def get_xgb_predicts(user_email):
    label_encoder = joblib.load(r'C:\Users\Victus\Desktop\AI Email Assistant\models\label_encoder.pkl')
    xgb = joblib.load(r'C:\Users\Victus\Desktop\AI Email Assistant\models\xgboost_email_classifier.pkl')
    
    # Kullanıcının e-postasını işle
    cleaned_user_email = preprocess_email(user_email)
    
    # Model tahmini yap
    predicted_class = xgb.predict(cleaned_user_email)

    # Sayısal tahmini etikete çevir
    predicted_label = label_encoder.inverse_transform(predicted_class)

    return predicted_label


if __name__ == "__main__":
    email = str(input('Emailinizi giriniz:'))
    topic_name = get_xgb_predicts(email)
    print(f'XGB nin tahmin ettigi kategori: {topic_name}')
    
    user_email = f"""
    You must respond to the email strictly based on the given category. 

    Category: {topic_name}

    Email: {email}

    Do not change or reanalyze the category. Only use the provided category for your response.
    """

    
    client = Response()
    response = client.get_response(user_email)
    print(response)
    