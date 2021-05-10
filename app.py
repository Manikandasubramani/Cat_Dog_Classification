import os
import numpy as np
import tensorflow
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

UPLOAD_FOLDER = '../Cat_Dog_Classifier/Uploaded_Images/'
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

classifier = load_model('../Cat_Dog_Classifier/models/cat_dog_classification.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    img = request.files['image']
    path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
    img.save(path)
    
    test_image = image.load_img(path, target_size=(64,64))
    test_image = image.img_to_array(test_image)
    test_image = test_image.reshape(1,64,64,3)

    output = classifier.predict(test_image)

    {'dog': 1, 'cats': 0}

    if output[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'

    return render_template('index.html', prediction_text='The uploaded pic is a {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
