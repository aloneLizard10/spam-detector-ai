import tkinter as tk
from tkinter import messagebox
import joblib
import string
from nltk.corpus import stopwords
import nltk
import os
import winsound  # For sound alerts (Windows only)

# Download stopwords if not already
nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Optional alert sound
def play_alert():
    # Use your own WAV file if desired
    frequency = 1000  # Hz
    duration = 300    # ms
    winsound.Beep(frequency, duration)

# Clean text function
def clean_text(msg):
    msg = msg.lower()
    msg = ''.join([c for c in msg if c not in string.punctuation])
    words = msg.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

# Prediction function
def predict():
    msg = text_entry.get("1.0", tk.END).strip()
    if not msg:
        messagebox.showwarning("Input Error", "Please enter a message.")
        return

    cleaned = clean_text(msg)
    vect = vectorizer.transform([cleaned])
    result = model.predict(vect)[0]

    if result == 1:
        result_label.config(text="🚫 SPAM", fg="#ff4c4c")
        # play_alert()  # Uncomment to enable sound
    else:
        result_label.config(text="✅ HAM", fg="#90ee90")

# Window setup
root = tk.Tk()
root.title("Spam Detector - Desktop App")
root.geometry("500x350")
root.resizable(False, False)

# Set dark mode colors
bg_color = "#2c2c2c"
fg_color = "#ffffff"
btn_color = "#4CAF50"

root.configure(bg=bg_color)

# Set icon (optional)
icon_path = "icon.ico"
if os.path.exists(icon_path):
    root.iconbitmap(icon_path)

# Widgets
tk.Label(root, text="📩 Spam Detector", font=("Helvetica", 18, "bold"),
         bg=bg_color, fg=fg_color).pack(pady=10)

tk.Label(root, text="Enter your message:", font=("Helvetica", 12),
         bg=bg_color, fg=fg_color).pack()

text_entry = tk.Text(root, height=6, width=55, wrap="word", font=("Helvetica", 11),
                     bg="#444", fg=fg_color, insertbackground=fg_color)
text_entry.pack(pady=5)

tk.Button(root, text="Check", command=predict,
          font=("Helvetica", 12, "bold"), bg=btn_color, fg="white").pack(pady=10)

result_label = tk.Label(root, text="", font=("Helvetica", 16, "bold"),
                        bg=bg_color)
result_label.pack(pady=5)

# Run app
root.mainloop()
