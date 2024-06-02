from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io

# Initialize the Flask app
app = Flask(__name__)

# Load the trained MNIST model
model = load_model('mnist_model.h5')

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        # Read the image file and convert it to an array
        image = Image.open(io.BytesIO(file.read()))
        image = image.convert('L').resize((28, 28), Image.ANTIALIAS)
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # Predict the class
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Return the result as JSON
        return jsonify({'prediction': int(predicted_class)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
