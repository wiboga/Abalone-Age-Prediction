from flask import Flask, render_template, request
import numpy as np
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model (ensure that 'model.pkl' is in the correct path)
with open('abalone_age_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    sex = float(request.form['sex'])
    length = float(request.form['length'])
    diameter = float(request.form['diameter'])
    height = float(request.form['height'])
    whole_weight = float(request.form['whole_weight'])
    shucked_weight = float(request.form['shucked_weight'])
    viscera_weight = float(request.form['viscera_weight'])
    shell_weight = float(request.form['shell_weight'])

    # Convert input data to numpy array
    input_data = np.array([[sex, length, diameter, height, whole_weight, shucked_weight, viscera_weight, shell_weight]])

    # Predict the age
    prediction = model.predict(input_data)

    # Return the prediction as a response
    return f'Predicted Age: {prediction[0]}'

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
