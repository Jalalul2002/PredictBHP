import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def transform_semester(semester):
    semester = int(semester)  # Konversi ke integer
    year = str(semester)[:4]  # Tahun
    sem = str(semester)[4]    # Semester
    return f"{year} Semester {sem}"

def model_lstm():
    # data_train_test = pd.read_csv('../Dataset/DataTrainTest-update.csv', delimiter=';')
    # data_train_test['updated_at'] = pd.to_datetime(data_train_test['updated_at'], format='%d/%m/%Y')
    # df_train_test = data_train_test[['product_code', 'kebutuhan', 'updated_at']]
    # df_transposed = df_train_test.pivot(index='updated_at', columns='product_code', values='kebutuhan').fillna(0)
    # df_transposed.fillna(0, inplace=True)
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # df_pivot_scaled = pd.DataFrame(scaler.fit_transform(df_transposed), columns=df_transposed.columns, index=df_transposed.index)
    # print(df_pivot_scaled)
    #
    # # Konversi Array
    # X = []
    # y = []
    # for i in range(len(df_pivot_scaled) - 1):
    #     X.append(df_pivot_scaled.iloc[i])
    #     y.append(df_pivot_scaled.iloc[i + 1])
    #
    # X = np.array(X)
    # y = np.array(y)
    # #
    # # Reshape data untuk LSTM
    # X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    data_bhp = pd.read_csv('../Dataset/DatasetAgro.csv', delimiter=';')
    data_matkul = pd.read_csv('../Dataset/Matkul_Agro.csv', delimiter=';')
    data_matkul = data_matkul[['Nama Matkul', 'Semester akademik', 'jml mahasiswa']]
    data_matkul['Semester akademik'] = data_matkul['Semester akademik'].apply(transform_semester)
    data_matkul_pivot = data_matkul.pivot(index='Semester akademik', columns='Nama Matkul',
                                          values='jml mahasiswa').fillna(0).astype(int)
    data_bhp['created_at'] = data_bhp['created_at'].apply(transform_semester)
    data_bhp_pivot = data_bhp.pivot(index='created_at', columns='product_code', values='kebutuhan').fillna(0).astype(
        int)
    scaler_bhp = MinMaxScaler(feature_range=(0, 1))
    scaler_matkul = MinMaxScaler(feature_range=(0, 1))
    data_bhp_scaled = scaler_bhp.fit_transform(data_bhp_pivot)
    data_matkul_scaled = scaler_matkul.fit_transform(data_matkul_pivot)
    data_train_y = data_bhp_scaled[:7]
    data_test_y = data_bhp_scaled[7]
    data_train_X = data_matkul_scaled[:7]
    data_test_X = data_matkul_scaled[7]
    X_data = np.reshape(data_train_X, (data_train_X.shape[0], data_train_X.shape[1], 1))
    X_test = np.reshape(data_test_X, (1, data_test_X.shape[0], 1))
    y_data = data_train_y
    y_test = data_test_y

    # Buat model LSTM
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(X_data.shape[1], X_data.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(y_data.shape[1]))

    model.compile(loss='mean_squared_error', optimizer="adam", metrics=['mean_absolute_error'])

    # Training model
    model.fit(X_data, y_data, epochs=100, batch_size=1)

    return model

if __name__ == '__main__':
    model = model_lstm()
    model.save("my_lstm_model.h5")