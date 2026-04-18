from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
import io
from PIL import Image

app = Flask(__name__)

model = tf.keras.models.load_model("my_model.keras")

class_names = [
    "Conditioner", "Pipes", "Surveillance cameras", "Tile",
    "Water Tank", "aluminum", "crack", "moisture",
    "plants", "satellite dish", "solar energy", "water leak",
    "wood", "حديد", "حديد مصدي", "عوازل وقرميد", "كهرباء"
]

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "لا يوجد صورة"}), 400

    file = request.files["image"]
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    predicted_class = class_names[np.argmax(pred)]
    confidence = float(np.max(pred)) * 100

    return jsonify({
        "class": predicted_class,
        "confidence": round(confidence, 2)
    })

@app.route("/", methods=["GET"])
def home():
    return "Wall Defect API is running!"

if __name__ == "__main__":
    app.run(debug=False)