from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('model/cnn_model.h5')

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img) / 255.0
    img_tensor = np.expand_dims(img_tensor, axis=0)
    prediction = model.predict(img_tensor)
    return "Malignant" if prediction[0][0] > 0.5 else "Benign"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"
    file = request.files['file']
    filepath = os.path.join('static', file.filename)
    file.save(filepath)
    result = predict_image(filepath)
    return render_template('result.html', prediction=result, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
