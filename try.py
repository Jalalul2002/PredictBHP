from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import io

from database_model import Prediksi, Dataperencanaan, Perencanaan, Assetlab, db

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/labsaintek'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

model = load_model('Model/my_lstm_model.h5')
scaler = MinMaxScaler(feature_range=(0, 1))

# Ambil tahun dan bulan saat ini
current_year = datetime.now().year
current_month = datetime.now().month
semester = 1 if current_month <= 6 else 2
nama_perencanaan = f"{current_year}-{semester}"

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
        df = pd.read_csv(file, delimiter=';')
        df['updated_at'] = pd.to_datetime(df['updated_at'], format='%d/%m/%Y')
        df_train_test = df[['product_code', 'kebutuhan', 'updated_at']]
        df_transposed = df_train_test.pivot(index='updated_at', columns='product_code', values='kebutuhan').fillna(0)
        df_pivot_scaled = pd.DataFrame(scaler.fit_transform(df_transposed), columns=df_transposed.columns,
                                       index=df_transposed.index)

        # Convert Data Array
        # input_data = np.array([df_pivot_scaled.values])
        input_data = np.array([df_pivot_scaled.iloc[-4:]])
        prediksi = model.predict(input_data)

        # Convert predictions to original scale
        prediksi_original = scaler.inverse_transform(prediksi)
        df_transposed.drop(df_transposed.index, inplace=True)
        prediksi_df = pd.DataFrame(prediksi_original, columns=df_transposed.columns)
        prediksi_df['updated_at'] = datetime.today().strftime('%Y-%m-%d')
        df_transposed = pd.concat([df_transposed, prediksi_df], ignore_index=True)
        df_transposed.set_index('updated_at', inplace=True)
        df_original = df_transposed.reset_index().melt(id_vars='updated_at', var_name='product_code',
                                                       value_name='kebutuhan')
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
            prediksi_entry = Prediksi(
                tahun_perencanaan=nama_perencanaan,
                product_code=row['product_code'],
                kebutuhan=row['kebutuhan'],
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

        # # Save to Excel
        # output = io.BytesIO()
        # with pd.ExcelWriter(output, engine='openpyxl') as writer:
        #     df_original.to_excel(writer, index=False, sheet_name='Prediksi')
        # output.seek(0)
        #
        # # Send file to API
        # return send_file(output, as_attachment=True, download_name='prediksi.xlsx',
        #                  mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        return jsonify({"status": "Prediction successful", "nama_perencanaan": nama_perencanaan}), 200

    except Exception as e:
        print(f"Error processing file: {e}")
        return jsonify({"status": "Error processing file", "error": str(e)}), 500

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(port=5001)