from __future__ import division, print_function

import os

import numpy as np
from flask import Flask, request, render_template
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = "../model.h5"

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()  # Necessary

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x, mode='caffe')

    return model.predict(x)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(
            base_path, 'uploads', secure_filename(f.filename)
        )
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        # return result
        for x in preds:
            return str(np.round(x))

    return None


if __name__ == '__main__':
    app.run(debug=True)

# https://stackoverflow.com/questions/72315003/valueerror-decode-predictions-expects-a-batch-of-predictions-i-e-a-2d-array
