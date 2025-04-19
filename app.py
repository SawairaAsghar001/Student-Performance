from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
model = joblib.load('student_performance_model.pkl')

@app.route('/')
def home():
    return "🎓 Student Grade Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Convert to DataFrame
    input_data = pd.DataFrame([data])

    # Predict grade
    prediction = model.predict(input_data)

    return jsonify({'predicted_grade': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
