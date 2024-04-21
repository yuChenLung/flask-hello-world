from flask import Flask, request
from skimage import io
import cv2, sys, json
from keras.saving import load_model
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# @app.route('/inference')
# def hello_world():
#     return 'Hello, World!'

# app = Flask(__name__)
# CORS(app)

@app.route('/inference', methods=['POST'])
def runModel():
    try:
        # print("test--")
        data = request.get_json()
        uri = data['input_data']
        
        model = load_model('./parkinson_disease_probability.h5')

        image = io.imread(uri)
        image = cv2.resize(image, (128, 128))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.array(image)
        image = np.expand_dims(image, axis=(0,-1))

        logits = model(image)
        scaled_logits = logits - 2 * tf.reduce_min(logits, axis=-1, keepdims=True)
        score = scaled_logits / tf.reduce_sum(scaled_logits, axis=-1, keepdims=True)

        output = {
            "healthy": float(score[0][0]),
            "parkinsons": float(score[0][1])
        }

        with open("inference.json", "w") as outfile: 
            json.dump(output, outfile)

        return "test"
    except Exception as e:
        print("test--")
        return str(e), 500

# if __name__ == '__main__':
#     app.run(debug=True)