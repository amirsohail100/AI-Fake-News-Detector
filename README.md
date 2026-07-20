# AI Fake News Detector

An advanced web application powered by a Deep Learning Artificial Neural Network (ANN) model that analyzes news text and classifies whether the information is Real or Fake with an outstanding **98% accuracy**.

## 🖥️ User Interface

![AI Fake News Detector UI](image_4b8d26.png)

## ✨ Features

- **High Accuracy:** Achieves a 98% success rate using a robust, multi-layered deep learning model.
- **Instant Verification:** Paste any news text or paragraph for immediate real-time analysis.
- **Clean Interface:** Intuitive web UI designed for straightforward and efficient user interaction.

## 🛠️ Model Architecture

The model is built using a Keras `Sequential` pipeline, combining Text Embeddings, Dense layers, Bidirectional LSTMs, and Dropout layers for optimal feature extraction and regularization:

```python
model_ann = Sequential([
    Embedding(input_dim=Max_words, output_dim=embedding_dim, input_length=Max_len),
    Dense(128, activation="relu"),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.4),
    Dense(128, activation="relu"),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Bidirectional(LSTM(16, return_sequences=False)),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dropout(0.1),
    Dense(1, activation="sigmoid")
])
```

AI Fake News Detector: An advanced web application powered by a Deep Learning Artificial Neural Network (ANN) that analyzes news text and classifies it as Real or Fake with an outstanding 98% accuracy. Paste text for instant verification.
