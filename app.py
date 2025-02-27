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

    
    
@app.route('/about')
def about():
    return '''
    <head >
        <title>Lung Cancer Classification</title>
        <i><center><h1 style="margin: 0.5em;font-size: 3rem ">Lung Cancer Detection Using Machine Learning</h1></center></i>
    </head>
     <body style="background-color: Black; color: white;  align-items: center; background-image: url('https://statnano.com/resource/news/files/images/21/2/thumbnail_98475abde06cc48db710e1b98f130ddc.jpg'); background-repeat: no-repeat; background-attachment: fixed; background-size: 105% 100%;">
          <a href="/home" style="color: white; text-decoration: none;">Back to Home</a>
          <div style="color:white; width: 100%; background-color: rgba(148, 185, 255, 0.4); padding:0.25em; margin:1em 0;">
                <ol style="display: flex; justify-content: center; list-style-type: none; padding: 0; gap:2em;">
                    <li><a href="/home" style="text-decoration:none; color:white">Home</a></li>
                     <li><a href="/home" style="text-decoration:none; color:white">|</a></li>
                    <li><a href="/about" style="text-decoration:none; color:white">About Us</a></li>
                </ol>
            </div>
        <center><h1 style="margin: 0.5em;font-size: 3rem; color:white; align-content:center">About Lung Cancer Detection</h1></center>
      
        <div style="margin-left:30%;">
        
        <p>Objective: Use deep learning to detect signs of lung cancer in CT scan images.</p>
        <p>Dataset: Lung Cancer CT Scan Dataset.</p>
        <p>Approach: Convolutional Neural Networks (CNNs) are widely used for image-based lung cancer detection.</p>
        <p>Implementation Overview:</p>
        <div style="margin-left: 40px;">
            <li>Data Collection: Use CT scan image datasets for lung cancer detection.</li>
            <li>Preprocessing: Resize, augment, and normalize CT scan images.</li>
            <li>Model: Train a CNN or a pre-trained model (e.g., ResNet or Inception) for classification.</li>
            <li>Evaluation: Use classification metrics to evaluate the model's accuracy and reliability.</li>
        </div>
        <p>Tech Stack: Python, TensorFlow/Keras, OpenCV.</p>
        <br>
        </div>
    </body>
    '''


@app.route('/')
def inter():
    return '''

    <body style="background-color: Black; color: white; margin: 20px; align-items: center; background-image: url('https://repository-images.githubusercontent.com/474572546/d2b783f4-a08f-4b2a-b26b-4989404f9304'); background-repeat: no-repeat; background-attachment: fixed; background-size: 100% 100%;">
        <div class="image-map-container" style="position: relative; width: 90%; height: 101vh;" onclick="window.location.href='/home'">
            <a href="/home" style="position: absolute; left: 20%; top: 50%; width: 100px; height: 100px; border: 2px solid rgba(255, 255, 255, 0.5,0); display: block;" title="Go Home"></a>
        </div>
    </body>
    '''
    


@app.route('/home')
def home():
    return '''
    <head>
        <title>Lung Cancer Classification</title>
        <i><center><h1 style="margin: 0.5em;font-size: 3rem ">Lung Cancer Detection Using Machine Learning</h1></center></i>
    </head>
    <body style="background-color: Black; color: black; margin: 20px; align-items: center; background-image: url('https://i.pinimg.com/736x/75/d1/ab/75d1abe32122ad678cdb9b5e91becc65.jpg'); background-repeat: no-repeat; background-attachment: fixed; background-size: 105% 100%;">
        <center>
            <div style="color:black; width: 100%; background-color: rgba(148, 185, 255, 0.4); padding:0.25em; margin:1em 0;">
                <ol style="display: flex; justify-content: center; list-style-type: none; padding: 0; gap:2em;">
                    <li><a href="/home" style="text-decoration:none; color:black;">Home</a></li>
                    <li><a href="/home" style="text-decoration:none; color:black">|</a></li>
                    <li><a href="/about" style="text-decoration:none; color:black">About Us</a></li>
                </ol>
            </div>
            <div style="text-align:center; padding:50px; font-family: Arial, sans-serif; width:fit-content; background-color: rgba(255, 255, 255, 0.2); border-radius: 10px; color: black; font-size: 20px; font-weight: bold;">
                <p>Upload a lung X-ray image to get a prediction.</p>
                <form action="/predict" method="post" enctype="multipart/form-data" style="margin-top:20px; ">
                    <input type="file" name="file" accept="image/*" required style="padding:10px; margin:10px;">
                    <br>
                    <input type="submit" value="Predict" style="padding:10px 20px; background-color:#28a745; color:black; border:none; cursor:pointer;">
                </form>
            </div>
        </center>
    </body>
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
    <head>
        <title>Lung Cancer Detection</title>
        <i><center><h1 style="margin: 0.5em;font-size: 3rem ">Lung Cancer Detection Using Machine Learning</h1></center></i>
    </head>
    
    
     <body style="background-color: Black; color: white; margin: 20px; align-items: center; background-image: url('https://statnano.com/resource/news/files/images/21/2/thumbnail_98475abde06cc48db710e1b98f130ddc.jpg'); background-repeat: no-repeat; background-attachment: fixed; background-size: 105% 100%;">
       
    
    <div style="text-align:center; padding:50px; font-family: Arial, sans-serif;">
        <h2 style="color:white;">Predicted Diagnosis: {predicted_label}</h2>
        
        '''+('''<h2 style="color: red;">Diagnosis: Malignant</h2>

<h3>DON’T BE STRESSED; FOLLOW OUR TREATMENT TIPS…</h3>

<h4>Dos:</h4>
<ul>
    <li>Follow Your Doctor’s Instructions: Always adhere to your oncologist’s advice, whether it's about treatments, medications, or follow-up appointments.</li>
    <li>Maintain a Healthy Diet: Focus on a balanced diet rich in fruits, vegetables, lean proteins, and whole grains.</li>
    <li>Stay Hydrated: Drink plenty of fluids, such as water, herbal teas, and clear soups. Staying hydrated helps in managing side effects like dry mouth or constipation.</li>
    <li>Get Rest: Prioritize good sleep and rest. Fatigue is a common symptom during treatment, so listen to your body and avoid overexertion.</li>
    <li>Manage Stress: Consider activities like meditation, deep breathing, or mindfulness exercises. Mental health is important, and managing stress can improve overall well-being.</li>
</ul>

<h4>Don'ts:</h4>
<ul>
    <li>Don’t Smoke: Smoking should be completely avoided. It’s crucial to quit, as smoking can worsen lung cancer and impede treatment effectiveness.</li>
    <li>Avoid Exposure to Pollutants: Stay away from second-hand smoke, industrial chemicals, or other environmental pollutants.</li>
    <li>Don’t Skip Appointments: Missing treatment sessions, scans, or follow-up appointments can delay treatment progress and increase the risk of recurrence.</li>
    <li>Don’t Overexert Yourself: Avoid physically demanding activities if you’re feeling fatigued. Too much strain can result in exhaustion or injury.</li>
    <li>Don’t Self-Medicate: Never take any medications or supplements without consulting your doctor first. Some over-the-counter medications or natural remedies can interfere with cancer treatment.</li>
</ul>

<h4>General Diet Plans for Lung Cancer Patients:</h4>
<ul>
    <li><strong>High-Protein Diet:</strong> Chicken, turkey, fish, eggs, beans, legumes, tofu, low-fat dairy, and lean meats.</li>
    <li><strong>Balanced Carbohydrates:</strong> Whole grain bread, brown rice, quinoa, oats, sweet potatoes, and whole-wheat pasta.</li>
    <li><strong>Healthy Fats:</strong> Avocados, olive oil, nuts, seeds, fatty fish (salmon, mackerel), and flaxseeds.</li>
    <li><strong>Fruits and Vegetables:</strong> Spinach, kale, carrots, berries, oranges, apples, tomatoes, broccoli, cauliflower, and Brussels sprouts.</li>
    <li><strong>Hydration:</strong> Drink water, herbal teas, and clear soups. Smoothies with fruits, vegetables, and yogurt are also beneficial.</li>
</ul>

<h4>Naturopathy and Ayurvedic Remedies:</h4>
<ul>
    <li><strong>Ashwagandha:</strong> Reduces stress and supports immunity.</li>
    <li><strong>Turmeric:</strong> Has strong anti-inflammatory properties, helps ease pain, and reduce swelling.</li>
    <li><strong>Tulsi (Holy Basil):</strong> Clears the lungs and improves breathing.</li>
    <li><strong>Amla (Indian Gooseberry):</strong> High in Vitamin C and boosts immunity.</li>
    <li><strong>Ginger:</strong> Helps remove toxins and supports digestion.</li>
</ul>

<h4>Ayurvedic Therapies:</h4>
<ul>
    <li><strong>Panchakarma:</strong> Cleanses toxins from the body, helping patients feel lighter and more energetic.</li>
    <li><strong>Nasya:</strong> Clears the respiratory system through medicinal oils administered through the nose.</li>
    <li><strong>Rasayana Therapy:</strong> Boosts immunity and energy, aiding the body's natural defenses.</li>
</ul>

<h4>Recommended Hospitals in South India:</h4>
<p>
    <strong>MGM Cancer Institute</strong>, Chennai - Dr. Dhanasekar Padmanabhan & Dr. Balaji Ramani<br>
    <strong>VS Hospitals</strong>, Chennai - Prof. Dr. S. Subramanian<br>
    <strong>Apollo Cancer Institutes</strong>, Chennai - Dr. Vishnu Ramanujan<br>
    <strong>Fuda Hospital</strong>, Chennai - Dr. Anup Aboti<br>
    <strong>DCodeCare</strong>, Bengaluru - Dr. Sandeep Nayak's Clinic
</p>

<p style="font-weight: bold; color: lightgreen;">HOPE YOU FEEL BETTER SOON…</p>
''' if predicted_label == "Malignant" else '''<head>
    <title>Lung Cancer Detection Result</title>
    <style>
        body {
            background-color: black;
            color: white;
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
        }
        .content {
            text-align: center;
            max-width: 600px;
            padding: 20px;
            background-color: rgba(0, 0, 0, 0.7);
            border-radius: 10px;
        }
        h2 {
            color: lightgreen;
        }
        a {
            color: white;
            text-decoration: none;
            padding: 10px 20px;
            background-color: #007bff;
            border-radius: 5px;
            display: inline-block;
            margin-top: 20px;
        }
    </style>
</head>

<body>
    <div class="content">
        <h2>Diagnosis: Normal</h2>
        <p>Maintain good health </p>
        <br>
        <a href="/">Go Back</a>
    </div>
</body>
''')+'''
        <br><a href="/" style="margin-top:20px; display:inline-block; padding:10px 20px; background-color:#007bff; color:white; text-decoration:none; border-radius:5px;">Go Back</a>
    </div>
    
      </body>
    '''

# ✅ Run Flask app
if __name__ == '__main__':
    app.run()