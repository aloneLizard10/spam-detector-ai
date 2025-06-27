import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
import nltk

# Ensure NLTK stopwords are available
nltk.download('stopwords')

# Load saved model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocessing function
def clean_text(msg):
    msg = msg.lower()
    msg = ''.join([c for c in msg if c not in string.punctuation])
    words = msg.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

# Streamlit UI
st.title("📩 Spam Message Classifier")
st.write("Enter a message and find out whether it's SPAM or HAM.")

# User input
user_input = st.text_area("Your message", "")

if st.button("Check"):
    cleaned = clean_text(user_input)
    vect_input = vectorizer.transform([cleaned])
    prediction = model.predict(vect_input)[0]

    if prediction == 1:
        st.error("🚫 This message is SPAM!")
    else:
        st.success("✅ This message is HAM (Not Spam)")
