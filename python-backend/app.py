from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def decode_image(image_data):

    try:
        image_bytes = base64.b64decode(image_data.split(',')[1])
        img = Image.open(BytesIO(image_bytes))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return None


@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():

    try:
        data = request.json
        image_data = data.get("image")
        frame = decode_image(image_data)

        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400

        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        emotions = analysis[0]['emotion']
        dominant_emotion = max(emotions, key=emotions.get)

        return jsonify({
            "emotions": emotions,
            "dominant_emotion": dominant_emotion
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
