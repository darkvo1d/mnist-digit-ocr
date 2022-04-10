# -*- coding: utf-8 -*-
"""
Created on Sat Mar 05 11:45:03 2022

@author: noopa
"""
import numpy as np
from keras.models import load_model
from flask import Flask, request, render_template
from utils import preprocess, value_invert, data_uri_to_cv2_img

app = Flask(__name__)
model = load_model("classifier.h5")

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # Read the image data from a base64 data URL
    imgstring = request.form.get('data')

    # Convert to OpenCV image
    img = preprocess(data_uri_to_cv2_img(imgstring))
    # cv2.imwrite('static/user_drawn/temp.png', img)

    data = (img / 255).reshape((1, 28, 28, 1))

    prediction = model.predict(data)
    classes_x = np.argmax(prediction, axis=1)
    predicted_class = classes_x

    return f"The digit drawn looks like {predicted_class}"


if __name__ == '__main__':
    app.run()