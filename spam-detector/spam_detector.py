import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import nltk
import string
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# Preprocessing function
def clean_text(msg):
    msg = msg.lower()
    msg = ''.join([char for char in msg if char not in string.punctuation])
    words = msg.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

# Apply cleaning
df['cleaned'] = df['message'].apply(clean_text)

# Vectorize the text
cv = CountVectorizer()
X = cv.fit_transform(df['cleaned'])
y = df['label'].map({'ham': 0, 'spam': 1})

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

# Save model and vectorizer
joblib.dump(model, 'spam_model.pkl')
joblib.dump(cv, 'vectorizer.pkl')

# Test function
def predict_spam(message):
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    cleaned = clean_text(message)
    vect_msg = vectorizer.transform([cleaned])
    result = model.predict(vect_msg)
    return "SPAM 🚫" if result[0] == 1 else "HAM ✅"

# Interactive testing loop
while True:
    user_input = input("\nEnter a message to check (or type 'exit'): ")
    if user_input.lower() == 'exit':
        print("Goodbye! 👋")
        break
    print("Prediction:", predict_spam(user_input))
