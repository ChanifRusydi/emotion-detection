# import numpy as np
# # import cv2
# from tensorflow.keras.models import load_model
# from flask import Flask, request, jsonify
# # from flask_pymongo import PyMongo
# from pymongo import MongoClient
# from flask_cors import CORS



# app = Flask(__name__)
# # CORS(app)
# app.config["DEBUG"] = True

# model = load_model("test_model1.h5")
# print("Loaded emotion model")
# # print(model.summary())
# labels = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
# db_pass = os.getenv("MONGODB_PASS")
# uri = f"mongodb+srv://user1:{db_pass}@cluster0.kexyy.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# print("URI:",uri)
# client = MongoClient(uri)
# db = client["test-database"]
# print(db)
# collection = db["test-collection"]
# print(collection)
# post = {

#     "author": "Mike",

#     "text": "My first blog post!",

#     "tags": ["mongodb", "python", "pymongo"],

#     "date": datetime.datetime.now(tz=datetime.timezone.utc),
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

app = Flask(__name__)

# Load the model
# model = load_model('model.h5')
model = load_model('test_model1.h5')

# Load the tokenizer
with open('tokenizer.json', 'r') as f:
    tokenizer_data = json.load(f)
tokenizer = tokenizer_from_json(tokenizer_data)

# Global variables
MAX_SEQUENCE_LENGTH = 256
CLASSES = ['negative', 'positive']
MODEL_NAME = "sentiment_model"

@app.route(f'/v1/models/{MODEL_NAME}:predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Check for required keys and extract the text
        if 'instances' not in data:
            return jsonify({'error': 'Missing instances in the request'}), 400
        text = data['instances']

        if not isinstance(text, str):
              return jsonify({'error': 'The instances value has to be a string'}), 400
        
        signature_name = data.get('signature_name','serving_default') # If not found, default to 'serving_default'
        if signature_name != "serving_default":
            return jsonify({'error': f'Unsupported signature_name:{signature_name}'}), 400
    

        # Preprocess the input
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)

        # Make prediction
        prediction = model.predict(padded_sequence)[0]
        predicted_class_index = np.argmax(prediction)
        predicted_class = CLASSES[predicted_class_index]


        return jsonify({
            'predictions': [predicted_class]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)





# app.config["MONGO_URI"] = "mongodb://<USERNAME>:<PASSWORD>@localhost:27017/<DB_NAME>?authSource=admin"
# db_pass = os.getenv("MONGODB_PASS")
# app.config["MONGO_URI"] = f"mongodb+srv://user1:{db_pass}@cluster0.kexyy.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# mongo = PyMongo(app)

# @app.route('/submit', methods=['POST'])
# def submit():
#     data = request.json
#     user_id = mongo.db.users.insert_one({
#         "name": data["name"],
#         "age": int(data["age"]),
#         "gender": data["gender"]
#     }).inserted_id
#     return jsonify({"user_id": str(user_id)})

# app.run(host='0.0.0.0',port=5000)