# 📩 Spam Detector AI

A smart and simple **Spam Detection System** built using Machine Learning, with both a **web interface** (Streamlit) and a **desktop app** (Tkinter).

> 🔍 Built with 💡 NLP, 🔥 Scikit-learn, 🎨 Streamlit, and 🖥️ Tkinter

---

## 🚀 Features

- 🔍 **Real-time spam prediction** using Naive Bayes
- 🌐 **Streamlit web app** — clean, responsive UI
- 💻 **Tkinter desktop GUI** — works offline
- 🧠 Text preprocessing with NLTK
- 🎨 Dark mode UI for desktop
- 🎵 Optional spam alert sound
- 🖼️ Custom app icon support
- 📦 Easy to package as `.exe`

---

## 🧠 How It Works

1. Model trained using the SMS Spam Collection Dataset
2. Cleaned using NLTK stopwords and punctuation removal
3. Vectorized using `CountVectorizer`
4. Trained with `MultinomialNB` (Naive Bayes)

---

## 🗂 Project Structure

spam-detector/
├── app.py # Streamlit web app
├── gui_app.py # Tkinter desktop app
├── spam_detector.py # Core model training script
├── spam_model.pkl # Saved ML model
├── vectorizer.pkl # Saved text vectorizer
├── spam.csv # Dataset (SMS messages)
├── icon.ico # (Optional) App icon
├── requirements.txt # Python dependencies
└── README.md # You're reading it!

📚 Dataset Used
SMS Spam Collection Dataset
UCI / Kaggle: View Dataset

👨‍💻 Author
Soham Ray
Built with ❤️ using Python and ML.


# spam-detector-ai
