from app import app as handler
from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Membaca dataset dari file CSV
data = pd.read_csv('warehouse.csv')

# Memeriksa apakah atribut yang dipilih ada dalam dataset
selected_columns = ['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']
missing_columns = set(selected_columns) - set(data.columns)
if missing_columns:
    print(f"Kesalahan: Kolom berikut tidak ada dalam dataset: {', '.join(missing_columns)}")
    exit(1)

# Memisahkan atribut dan target
X = data[selected_columns]
y = data['Species']

# Mengisi nilai yang hilang dengan nilai rata-rata
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Normalisasi atribut menggunakan StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Membangun model Random Forest
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

            # Check if input values are within the range of the dataset
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
                # Menggunakan imputer yang sama untuk mengisi nilai yang hilang pada input
                input_data = [[weight, length1, length2, length3, height, width]]
                input_data_imputed = imputer.transform(input_data)

                # Normalisasi input menggunakan StandardScaler yang sama
                input_data_scaled = scaler.transform(input_data_imputed)

                # Memprediksi spesies ikan berdasarkan inputan pengguna
                prediction = rf.predict(input_data_scaled)
                prediction = prediction[0]  # Ambil elemen pertama dari array prediksi

                # Menghilangkan tanda ' ' dan []
                prediction = prediction.strip("'")
                prediction = prediction.strip("[]")

        except ValueError:
            error_message = "Kesalahan: Pastikan input morfometrik berupa angka yang valid."

    return render_template('index.html', prediction=prediction, error_message=error_message)

