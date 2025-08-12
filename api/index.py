from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

app = Flask(__name__, template_folder="../templates")

# Path ke warehouse.csv
csv_path = os.path.join(os.path.dirname(__file__), "../warehouse.csv")
data = pd.read_csv(csv_path)

selected_columns = ['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']
missing_columns = set(selected_columns) - set(data.columns)
if missing_columns:
    raise Exception(f"Kolom berikut tidak ada dalam dataset: {', '.join(missing_columns)}")

X = data[selected_columns]
y = data['Species']

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error_message = None

    if request.method == 'POST':
        try:
            weight = float(request.form['weight'])
            length1 = float(request.form['length1'])
            length2 = float(request.form['length2'])
            length3 = float(request.form['length3'])
            height = float(request.form['height'])
            width = float(request.form['width'])

            if (
                weight < X['Weight'].min() or weight > X['Weight'].max() or
                length1 < X['Length1'].min() or length1 > X['Length1'].max() or
                length2 < X['Length2'].min() or length2 > X['Length2'].max() or
                length3 < X['Length3'].min() or length3 > X['Length3'].max() or
                height < X['Height'].min() or height > X['Height'].max() or
                width < X['Width'].min() or width > X['Width'].max()
            ):
                error_message = "Kesalahan : Input morfometrik melebihi rentang yang diharapkan."
            else:
                input_data = [[weight, length1, length2, length3, height, width]]
                input_data_imputed = imputer.transform(input_data)
                input_data_scaled = scaler.transform(input_data_imputed)
                prediction = rf.predict(input_data_scaled)
                prediction = prediction[0]
                prediction = prediction.strip("'").strip("[]")
        except ValueError:
            error_message = "Kesalahan: Pastikan input morfometrik berupa angka yang valid."

    return render_template('index.html', prediction=prediction, error_message=error_message)

# Handler untuk Vercel
handler = app

