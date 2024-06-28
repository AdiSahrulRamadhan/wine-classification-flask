from flask import Flask, render_template, request
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import pickle
import os

app = Flask(__name__)

MODEL_PATH = 'gnb_model.pkl'

if not os.path.exists(MODEL_PATH):
    data_wine = pd.read_csv("wine.csv")
    X = data_wine.drop(columns=['class'])
    y = data_wine['class']

    # Inisiasi Model Gaussian Naive Bayes dan data train 
    gnb_model = GaussianNB()
    gnb_model.fit(X, y)

    # Save Model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(gnb_model, f)

    # Output to confirm training completion
    print("Model Tersimpan.")
else:
    # Load the model
    with open(MODEL_PATH, 'rb') as f:
        gnb_model = pickle.load(f)
    print("Model Sudah Tersimpan.")


@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_class = None
    input_data = {}
    error_message = None

    if request.method == 'POST':
        try:
            # Get user input from form and validate
            input_data['alcohol'] = float(request.form['alcohol'].replace(',', '.'))
            input_data['malic_acid'] = float(request.form['malic_acid'].replace(',', '.'))
            input_data['ash'] = float(request.form['ash'].replace(',', '.'))
            input_data['alcalinity_of_ash'] = float(request.form['alcalinity_of_ash'].replace(',', '.'))
            input_data['magnesium'] = float(request.form['magnesium'].replace(',', '.'))
            input_data['total_phenols'] = float(request.form['total_phenols'].replace(',', '.'))
            input_data['flavanoids'] = float(request.form['flavanoids'].replace(',', '.'))
            input_data['nonflavanoid_phenols'] = float(request.form['nonflavanoid_phenols'].replace(',', '.'))
            input_data['proanthocyanins'] = float(request.form['proanthocyanins'].replace(',', '.'))
            input_data['color_intensity'] = float(request.form['color_intensity'].replace(',', '.'))
            input_data['hue'] = float(request.form['hue'].replace(',', '.'))
            input_data['od280_od315'] = float(request.form['od280_od315'].replace(',', '.'))
            input_data['proline'] = float(request.form['proline'].replace(',', '.'))

            # Make prediction for the user input
            new_data_point = [[
                input_data['alcohol'],
                input_data['malic_acid'],
                input_data['ash'],
                input_data['alcalinity_of_ash'],
                input_data['magnesium'],
                input_data['total_phenols'],
                input_data['flavanoids'],
                input_data['nonflavanoid_phenols'],
                input_data['proanthocyanins'],
                input_data['color_intensity'],
                input_data['hue'],
                input_data['od280_od315'],
                input_data['proline']
            ]]
            predicted_class = gnb_model.predict(new_data_point)[0]
        except ValueError:
            error_message = "Invalid input: Please enter valid numbers."

    return render_template('index.html', predicted_class=predicted_class, input_data=input_data, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
