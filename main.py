from flask import Flask, request, jsonify, send_file
import os

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from urllib.parse import quote_plus
import io

from database_model import Prediksi, Dataperencanaan, Perencanaan, Assetlab, db

app = Flask(__name__)

username = "managerlab"
password = "l@bJu@r4"  # Password dengan karakter @
host = "localhost"
database = "simalab"

encoded_password = quote_plus(password)  # Encode karakter spesial
database_uri = f"mysql+pymysql://{username}:{encoded_password}@{host}/{database}"
app.config['SQLALCHEMY_DATABASE_URI'] = database_uri

# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/labsaintek'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

scaler = MinMaxScaler(feature_range=(0, 1))

# Ambil tahun dan bulan saat ini
current_year = datetime.now().year
current_month = datetime.now().month
semester = 1 if current_month <= 6 else 2
nama_perencanaan = f"{current_year}-{semester}"

def transform_semester(semester):
    semester = int(semester)  # Konversi ke integer
    year = str(semester)[:4]  # Tahun
    sem = str(semester)[4]    # Semester
    return f"{year} Semester {sem}"

@app.route('/predict', methods=['POST'])
def predict():
    if 'csv_file' not in request.files:
        return jsonify({"status": "No file received"}), 400

    user_id = request.form.get('user_id')
    if not user_id:
        return jsonify({"status": "User ID is required"}), 400

    file = request.files['csv_file']
    if file.filename == '':
        return jsonify({"status": "No selected file"}), 400

    try:
        # Get Data
        model = load_model('Model/new_model_2.2.h5')
        df = pd.read_csv(file, delimiter=';')
        data_bhp = pd.read_csv('Dataset/DatasetAgro.csv', delimiter=';')
        data_bhp['created_at'] = data_bhp['created_at'].apply(transform_semester)
        data_bhp_pivot = data_bhp.pivot(index='created_at', columns='product_code', values='kebutuhan').fillna(
            0).astype(
            int)
        data_matkul = df[['Nama Matkul', 'Semester akademik', 'jml mahasiswa']]
        data_matkul['Semester akademik'] = data_matkul['Semester akademik'].apply(transform_semester)
        data_matkul_pivot = data_matkul.pivot(index='Semester akademik', columns='Nama Matkul',
                                              values='jml mahasiswa').fillna(0).astype(int)
        scaler_matkul = MinMaxScaler(feature_range=(0, 1))
        scaler_bhp = MinMaxScaler(feature_range=(0, 1))
        data_bhp_scaled = scaler_bhp.fit_transform(data_bhp_pivot)
        data_matkul_scaled = scaler_matkul.fit_transform(data_matkul_pivot)
        input_data = np.reshape(data_matkul_scaled, (1, data_matkul_scaled.shape[1], 1))
        prediksi = model.predict(input_data)
        prediksi_original = scaler_bhp.inverse_transform(prediksi)

        new_data_bhp = data_bhp_pivot.drop(data_bhp_pivot.index, inplace=True)
        prediksi_df = pd.DataFrame(prediksi_original, columns=data_bhp_pivot.columns)
        prediksi_df['updated_at'] = datetime.today().strftime('%Y-%m-%d')
        new_data_bhp = pd.concat([data_bhp_pivot, prediksi_df], ignore_index=True)
        new_data_bhp.set_index('updated_at', inplace=True)
        df_original = new_data_bhp.reset_index().melt(id_vars='updated_at', var_name='product_code',
                                                      value_name='kebutuhan')
        # Mengubah tipe data kolom 'kebutuhan' menjadi integer dan mengganti nilai negatif menjadi nol.
        df_original['kebutuhan'] = df_original['kebutuhan'].astype('int', errors='ignore')
        df_original['kebutuhan'] = df_original['kebutuhan'].apply(lambda x: 0 if x < 0 else x)
        df_original = df_original[df_original['kebutuhan'] > 0]
        df_original = df_original.sort_values(by='kebutuhan', ascending=False)
        df_original['location_code'] = df_original['product_code'].str[:3]
        print(df_original.head())

        # Delete data from DB
        db.session.query(Prediksi).delete()
        db.session.commit()

        # Save to Database
        for _, row in df_original.iterrows():
            # Tentukan nama lokasi berdasarkan location_code
            location_mapping = {
                '701': "Matematika",
                '702': "Biologi",
                '703': "Fisika",
                '704': "Kimia",
                '705': "Teknik Informatika",
                '706': "Agroteknologi",
            }

            location = location_mapping.get(row['location_code'], "Teknik Elektro")  # Default jika tidak ada di mapping

            prediksi_entry = Prediksi(
                tahun_perencanaan=nama_perencanaan,
                product_code=row['product_code'],
                kebutuhan=row['kebutuhan'],
                location=location,  # Tambahkan field location
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            db.session.add(prediksi_entry)
        db.session.commit()

        unique_locations = df_original['location_code'].unique()

        # Buat entri `Dataperencanaan` dan `Perencanaan`
        for location in unique_locations:
            if location == '701':
                prodi = "Matematika"
            elif location == '702':
                prodi = "Biologi"
            elif location == '703':
                prodi = "Fisika"
            elif location == '704':
                prodi = "Kimia"
            elif location == '705':
                prodi = "Teknik Informatika"
            elif location == '706':
                prodi = "Agroteknologi"
            else:
                prodi = "Teknik Elektro"

            data_perencanaan_entry = Dataperencanaan(
                nama_perencanaan=f"{nama_perencanaan}",
                prodi=prodi,
                type="bhp",
                status="belum",
                created_by=user_id,
                updated_by=user_id,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            db.session.add(data_perencanaan_entry)
            db.session.commit()

            # Ambil id dari `data_perencanaan_entry` yang baru dibuat
            rencana_id = data_perencanaan_entry.id

            # Filter data berdasarkan lokasi dan masukkan ke tabel `Perencanaan`
            location_data = df_original[df_original['location_code'] == location]
            for _, row in location_data.iterrows():
                asset = Assetlab.query.filter_by(product_code=row['product_code']).first()
                stock = asset.stock if asset else 0

                perencanaan_entry = Perencanaan(
                    rencana_id=rencana_id,
                    product_code=row['product_code'],
                    stock=stock,
                    jumlah_kebutuhan=row['kebutuhan'],
                    created_by=user_id,
                    updated_by=user_id,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                db.session.add(perencanaan_entry)
            db.session.commit()

        return jsonify({"status": "Prediction successful", "nama_perencanaan": nama_perencanaan}), 200

    except Exception as e:
        print(f"Error processing file: {e}")
        return jsonify({"status": "Error processing file", "error": str(e)}), 500

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    # app.run(port=5001)
    app.run(host='0.0.0.0' ,port=5001)