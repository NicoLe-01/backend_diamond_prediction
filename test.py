from flask import Flask, request, jsonify
import joblib  # or any other library to load your model
import numpy as np

app = Flask(__name__)

# Load your trained model here
model = joblib.load('xgb_model.pkl')  # Update with the actual path to your model file

data = {
    "carat": 0.5,
    "cut": 2.0,
    "x": 3.0,
    "y": 4.0,
    "z": 5.0
}

carat = data['carat']
cut = data['cut']
x = data['x']
y = data['y']
z = data['z']


input_data = np.array([[carat, cut, x, y, z]])

prediction = model.predict(input_data)


print(round(prediction[0], 2))
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
#         # Assuming you have a JSON input like {"carat": 0.5, "cut": 2, "x": 3.0, "y": 4.0, "z": 5.0}
#         carat = data['carat']
#         cut = data['cut']
#         x = data['x']
#         y = data['y']
#         z = data['z']
#         print(carat)

#         # Make predictions using the loaded model
#         prediction = model.predict([[carat, cut, x, y, z]])
#         print(prediction)

#         return jsonify({'prediction': prediction[0]})

#     except Exception as e:
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)
