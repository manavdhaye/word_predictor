# 🏏 Next Word Predictor – Cricket Edition (Streamlit + Keras)

This is a simple Streamlit web application that predicts the **next word(s)** based on a user’s input, using a **trained LSTM model**. The model is trained on a short cricket-themed paragraph focused on the Indian men's cricket team.

---

## 🚀 Features

- Predict the **next N words** (customizable with a slider)
- Uses **Keras LSTM model**
- Tokenizer trained on a **custom cricket paragraph**
- 80% accuracy on short paragraph dataset
- Lightweight and runs on **Streamlit**
- Clean user interface with scrollable reference paragraph

---

## 📋 Tech Stack

- **Frontend**: Streamlit
- **Backend**: TensorFlow / Keras (LSTM model)
- **Language**: Python
- **Tokenizer**: Keras Tokenizer
- **Model**: Sequential LSTM trained on a short dataset

---

## 🧠 Model Info

- **Architecture**: Embedding → LSTM → Dense (Softmax)
- **Trained On**: A paragraph about Indian cricket players and history
- **Accuracy**: ~80% on validation (short dataset)
- **Input Shape**: Sequences padded to length 77

---

## 🔧 Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/your-username/word-predictor-app.git
cd word-predictor-app
