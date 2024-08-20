import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained model
model_path = r'E:\Brain Tumor Analyzation\model.h5'
model = load_model(model_path)

# Define categories (class labels)
categories = ['negative', 'positive']

# Prediction function
def predict_image(model, img_path, target_size=(75, 75)):
    try:
        img = Image.open(img_path)
        img = img.resize(target_size).convert('L')  # Resize and convert to grayscale
        img_array = np.array(img) / 255.0  # Normalize
        
        # Ensure the input shape matches what the model expects
        # Reshape the image array to (1, 75, 75, 1)
        img_array = img_array.reshape(1, target_size[0], target_size[1], 1)
        
        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction[0])
        
        return categories[predicted_class]
    
    except Exception as e:
        return str(e)

# Route to upload image and get prediction
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        try:
            # Save the uploaded file to a temporary location
            upload_folder = 'uploads'
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
            
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)

            # Get prediction
            predicted_class = predict_image(model, file_path)
            os.remove(file_path)  # Remove the uploaded file

            return jsonify({'result': predicted_class})

        except Exception as e:
            return jsonify({'error': str(e)})

    return render_template('Index.html')

if __name__ == '__main__':
    import sys
    if sys.argv[0].endswith('debugpy'):
        app.run(debug=True, use_reloader=False)
    else:
        app.run(debug=True)
