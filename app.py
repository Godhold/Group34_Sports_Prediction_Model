from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np

# Load the pre-trained model and scaler
model = joblib.load('/Users/godholdalomenu/Desktop/Deploy/sports_prediction_model.pkl ')
scaler = joblib.load('/Users/godholdalomenu/Desktop/Deploy/scaler.pkl ')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve input data from the web form
        input_data = request.form.to_dict()

        # Convert input data to a DataFrame
        input_data = pd.DataFrame(input_data, index=[0])

        # Define the order of features based on your model's input
        feature_order = [
            'age', 'value_eur', 'movement_reactions', 'passing', 'dribbling', 'shooting', 'defending', 'physic',
            'wage_eur', 'potential', 'release_clause_eur', 'pace', 'height_cm', 'weight_kg', 'club_contract_valid_until',
            'weak_foot', 'skill_moves', 'international_reputation', 'attacking_short_passing', 'skill_curve',
            'skill_fk_accuracy', 'skill_long_passing', 'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
            'movement_reactions', 'power_jumping', 'power_stamina', 'power_strength', 'defending_marking_awareness',
            'defending_standing_tackle', 'defending_sliding_tackle', 'shooting_skills', 'goalkeeping_ability', 'mentality'
        ]

        # Reorder the DataFrame columns according to the feature order
        input_data = input_data[feature_order]

        # Preprocess and scale the input data using the scaler
        scaled_input = scaler.transform(input_data)

        # Use the model to make predictions
        prediction = model.predict(scaled_input)[0]

        return render_template('index.html', prediction=f'Predicted Overall Rating: {prediction:.2f}')
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
