from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import pickle
import joblib

app = Flask(__name__)

# Load model and vectorizer

model = joblib.load("SVM.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    review = request.form["review"]
    vector = vectorizer.transform([review])
    prediction = model.predict(vector)
    confidence = round(float(prediction) * 100, 2) if prediction > 0.5 else round(float(1 - prediction) * 100, 2)
    sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😞"
    return jsonify({
        "sentiment": sentiment,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)
