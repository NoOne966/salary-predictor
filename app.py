from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load your trained model, scaler, and encoders
model = joblib.load('model/tuned_salary_prediction_model.pkl')
scaler = joblib.load('model/scaler.pkl')
label_encoders = joblib.load('model/label_encoders.pkl')

# For dropdown generation:
countries = ['United-States', 'India', 'Mexico', 'Philippines', 'Germany', 'Canada', 'England', 'Other']
marital_statuses = ['Single', 'Married', 'Divorced']
sexes = ['Male', 'Female']

@app.route('/')
def home():
    return render_template(
        'index.html',
        countries=countries,
        marital_statuses=marital_statuses,
        sexes=sexes
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        sex = request.form['sex']
        country = request.form['country']
        marital_status = request.form['marital_status']
        education_num = int(request.form['education_num'])

        # Manually encode sex:
        sex_encoded = 1 if sex.lower() == 'male' else 0

        # Encode using label_encoders if available, else default to 0
        def encode_label(col, value):
            try:
                le = label_encoders[col]
                return le.transform([value])[0]
            except:
                return 0

        country_encoded = encode_label('native-country', country)
        marital_status_encoded = encode_label('marital-status', marital_status)
        workclass_encoded = encode_label('workclass', 'Private')
        occupation_encoded = encode_label('occupation', 'Prof-specialty')
        relationship_encoded = encode_label('relationship', 'Not-in-family')
        race_encoded = encode_label('race', 'White')

        X_input = np.array([
            age,
            workclass_encoded,
            200000,  # fnlwgt
            education_num,
            marital_status_encoded,
            occupation_encoded,
            relationship_encoded,
            race_encoded,
            sex_encoded,
            0,       # capital-gain
            0,       # capital-loss
            40,      # hours-per-week
            country_encoded
        ]).reshape(1, -1)

        X_input_scaled = scaler.transform(X_input)

        prediction = model.predict(X_input_scaled)[0]
        result = ">50K" if prediction == 1 else "<=50K"
        return render_template('result.html', prediction=result)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
