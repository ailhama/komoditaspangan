import numpy as np
import pickle
from flask import Flask, request, render_template

# Create Flask app
app = Flask(__name__)

#load model
model_file = open('randomforest_regression_model.pkl', 'rb')
multioutput_regressor = pickle.load(model_file)

# Dataset columns
komoditas_columns = ['Beras Premium', 'Beras Medium', 'Minyak Goreng Curah', 'Gula Pasir', 'Terigu Segitiga Biru',
                     'Bawang Merah', 'Bawang Putih', 'Cabe Rawit Merah', 'Cabe Merah Keriting', 'Cabe Besar',
                     'Daging Sapi', 'Daging Ayam Ras', 'Telur Ayam Ras', 'Ikan', 'Kedelai Biji Kering',
                     'Jagung Pipilan Kering', 'Jagung Manis']

# Route
@app.route("/")
def index():
    return render_template("index.html", hasil={})

@app.route("/predict", methods=["POST"])
def predict():
    bulan = float(request.form["bulan"])
    minggu = float(request.form["minggu"])
    komoditas = request.form["komoditas"]

    # Membuat input array untuk model
    input_array = np.array([[bulan, minggu]])

    # Melakukan prediksi
    predictions = multioutput_regressor.predict(input_array)
    
    # Mendapatkan harga untuk komoditas yang dipilih
    indeks_komoditas = komoditas_columns.index(komoditas)
    harga_komoditas_dipilih = predictions[0][indeks_komoditas]

    # Menyiapkan hasil untuk ditampilkan
    hasil = {harga_komoditas_dipilih}

    return render_template('index.html', hasil=hasil)

if __name__ == "__main__":
    app.run(debug=True)