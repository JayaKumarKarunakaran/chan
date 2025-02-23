from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import joblib
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ✅ Load the trained model correctly using joblib
model = joblib.load('lung_cancer_rf_model.pkl')

# ✅ Class labels
class_labels = {1: "Malignant", 2: "Normal"}

# ✅ Function to preprocess the uploaded image for Random Forest
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img.flatten().reshape(1, -1)  # Flatten for RF model input
    return img

# ✅ Home route to upload the image
@app.route('/')
def home():
    return '''
    <div style="text-align:center; padding:50px; font-family: Arial, sans-serif;">
        <h1 style="color:#333;">Lung Cancer Classification</h1>
        <p>Upload a lung X-ray image to get a prediction.</p>
        <form action="/predict" method="post" enctype="multipart/form-data" style="margin-top:20px;">
            <input type="file" name="file" accept="image/*" required style="padding:10px; margin:10px;">
            <br>
            <input type="submit" value="Predict" style="padding:10px 20px; background-color:#28a745; color:white; border:none; cursor:pointer;">
        </form>
    </div>
    '''

# ✅ Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # ✅ Preprocess the image
    image_data = preprocess_image(file_path)

    # ✅ Make prediction using Random Forest
    prediction = model.predict(image_data)  # Random Forest outputs a class directly
    predicted_label = class_labels[int(prediction[0])]  # Convert prediction to label

    return f'''
    <div style="text-align:center; padding:50px; font-family: Arial, sans-serif;">
        <h2 style="color:#333;">Predicted Diagnosis: {predicted_label}</h2>
        <img src="/uploads/{filename}" alt="Uploaded Image" style="max-width:300px; border:2px solid #ddd; padding:10px; margin-top:20px;">
        <br><a href="/" style="margin-top:20px; display:inline-block; padding:10px 20px; background-color:#007bff; color:white; text-decoration:none; border-radius:5px;">Go Back</a>
    </div>
    '''

# ✅ Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)