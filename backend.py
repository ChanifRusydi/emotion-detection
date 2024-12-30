from flask import Flask, request, jsonify, send_file
from flask_pymongo import PyMongo
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import base64
import datetime
from bson.objectid import ObjectId
import os

app = Flask(__name__)
CORS(app)

app.config["MONGO_URI"] = "mongodb://<USERNAME>:<PASSWORD>@localhost:27017/<DB_NAME>?authSource=admin"
mongo = PyMongo(app)

emotion_model = tf.keras.models.load_model("emotion_model.h5")
emotion_labels = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']

@app.route('/submit', methods=['POST'])
def submit():
    data = request.json
    user_id = mongo.db.users.insert_one({
        "name": data["name"],
        "age": int(data["age"]),
        "gender": data["gender"]
    }).inserted_id
    return jsonify({"user_id": str(user_id)})

@app.route('/video/<int:video_id>')
def get_video(video_id):
    video_path = f"static/video{video_id}.mp4"
    if os.path.exists(video_path):
        return send_file(video_path, mimetype="video/mp4")
    else:
        return jsonify({"error": "Video not found"}), 404

@app.route('/webcam_frame', methods=['POST'])
def webcam_frame():
    data = request.json
    user_id = data["user_id"]
    frame_data = data["frame"]

    # Decode frame
    header, encoded = frame_data.split(",", 1)
    decoded = base64.b64decode(encoded)
    np_data = np.frombuffer(decoded, np.uint8)
    frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

    # Preprocess frame for emotion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48)) / 255.0
    frame_input = np.expand_dims(resized, axis=-1)
    frame_input = np.expand_dims(frame_input, axis=0)

    # Predict emotion
    probabilities = emotion_model.predict(frame_input)[0]
    emotion_label = emotion_labels[np.argmax(probabilities)]

    # Store result
    mongo.db.emotion_data.insert_one({
        "user_id": ObjectId(user_id),
        "timestamp": datetime.datetime.utcnow(),
        "emotion_label": emotion_label,
        "emotion_probabilities": probabilities.tolist()
    })

    return jsonify({"emotion_label": emotion_label})

@app.route('/aggregation/<user_id>', methods=['GET'])
def aggregate_emotions(user_id):
    data = list(mongo.db.emotion_data.find({"user_id": ObjectId(user_id)}))
    if not data:
        return jsonify({"message": "No data found"}), 404

    emotion_counts = {label: 0 for label in emotion_labels}
    for entry in data:
        emotion_counts[entry["emotion_label"]] += 1

    dominant_emotion = max(emotion_counts, key=emotion_counts.get)
    return jsonify({"emotion_counts": emotion_counts, "dominant_emotion": dominant_emotion})

if __name__ == '__main__':
    app.run(debug=True)
