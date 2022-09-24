from __future__ import division, print_function

import os

import numpy as np
from flask import Flask, request, render_template
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from werkzeug.utils import secure_filename

DEFAULT_PORT = 5000
TARGET_SIZE = (224, 224)
MODEL_PATH = "../model/model_resnet.h5"

app = Flask(__name__)

print("Loading model")

model = load_model(MODEL_PATH)
model.make_predict_function()  # Necessary

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, _model):
    img = keras_image.load_img(img_path, target_size=TARGET_SIZE)

    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    return _model.predict(x)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method != 'POST':
        return

    f = request.files['file']

    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)

    prediction = model_predict(file_path, model)

    # Process your result for human
    pred_class = prediction.argmax(axis=-1)  # Simple argmax
    # pred_class = decode_predictions(prediction, top=1)   # ImageNet Decode
    # result = str(pred_class[0][0][1])               # Convert to string
    # return result
    result = ""
    for x in prediction:
        result += str(np.round(x))

    return result + ", Class: " + pred_class


if __name__ == '__main__':
    port = int(os.environ.get('PORT', DEFAULT_PORT))
    app.run(debug=True, host='0.0.0.0', port=port)

# https://stackoverflow.com/questions/72315003/valueerror-decode-predictions-expects-a-batch-of-predictions-i-e-a-2d-array
