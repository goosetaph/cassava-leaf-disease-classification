import os
from flask import Flask, request, jsonify, render_template # Import render_template
import numpy as np
from PIL import Image
import io
from tensorflow import keras

app = Flask(__name__)

model = keras.models.load_model('cassava_model_b7.keras')

@app.route('/', methods=['GET']) # Add this route for the root URL
def index():
    return render_template('index.html') # Render the HTML template

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']

    try:
        # Open the image from the bytes
        image = Image.open(image_file.stream).convert('RGB')
        image = image.resize((600, 600))
        image = np.array(image) / 255.0  # Normalize HERE!
        print(image) #DEBUGGING
        image = np.array(image) # divide by 255.0 if model used normalization
        image = np.expand_dims(image, axis=0)
        print(image.shape) #DEBUGGING
        print(image) #Print the image data #DEBUGGING
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 400

    try:
        print(image.shape) #Print the image shape #DEBUGGING
        prediction = model.predict(image)
        print(prediction) #Print the prediction #DEBUGGING
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]
        class_names = ['Bacterial Blight', 'Brown Streak Disease', 'Green Mottle', 'Mosaic Disease', 'Healthy']
        predicted_disease = class_names[predicted_class]
        print(f"Image Shape: {image.shape}")  # More descriptive
        prediction = model.predict(image)
        print(f"Raw Predictions: {prediction}")  # More descriptive
        print(f"Prediction Shape: {prediction.shape}") #CHECK THE SHAPE
        predicted_class = np.argmax(prediction)
        print(f"Predicted Class Index: {predicted_class}") #Print the index
        print(f"Class Names List: {class_names}") #Print the class names
        predicted_disease = class_names[predicted_class]
        print(f"Predicted Disease: {predicted_disease}") #Print the disease name
        confidence = prediction[0][predicted_class]
        print(f"Confidence: {confidence}")
        return jsonify({'disease': predicted_disease, 'confidence': float(confidence)*100.0}), 200
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)