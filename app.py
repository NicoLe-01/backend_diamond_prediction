from flask import Flask, request, jsonify
import joblib  # or any other library to load your model
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load your trained model here
model = joblib.load('xgb_model.pkl')  # Update with the actual path to your model file

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Assuming you have a JSON input like {"carat": 0.5, "cut": 2, "x": 3.0, "y": 4.0, "z": 5.0}
        carat = data['carat']
        cut = data['cut']
        x = data['x']
        y = data['y']
        z = data['z']

        input_data = np.array([[carat, cut, x, y, z]])
        # Make predictions using the loaded model
        prediction = model.predict(input_data)
        print(prediction)

        return jsonify({'prediction': round(float(prediction[0]), 2)})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/hello', methods=['GET'])
def hello_world():
    return "Hello, World!"

if __name__ == '__main__':
    app.run(debug=True)
