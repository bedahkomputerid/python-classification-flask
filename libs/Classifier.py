# import library pandas untuk membuat dataframe
import pandas as pd
# import library copy untuk copy object
import copy
# import library json
import json

# import library StandardScaler untuk normalisasi
from sklearn.preprocessing import StandardScaler
# import library LogisticRegression untuk prediksi
from sklearn.linear_model import LogisticRegression
# import train test split untuk split dataset
from sklearn.model_selection import train_test_split
# import clasification report untuk cek hasil prediksi
from sklearn.metrics import classification_report

# buat class baru
class LRClassifier:

    # buat constructor
    def __init__(self, dataset, delimiter, column, na_values, target):
        # simpan dataset
        self.__dataset = dataset
        # simpan delimiter
        self.__delimiter = delimiter
        # simpan nama kolom
        self.__column = column
        # simpan na value(penanda value kosong)
        self.__na_values = na_values
        # simpan nama kolom target
        self.__target = target
        # buat model StandardScaler untuk normalisasi data
        self.__scaler = StandardScaler()
        # baca dataset
        self.__load()

    # buat fungsi untuk load data
    def __load(self):
        # load data dari csv sesuai parameter yang sudah disimpan dari constructor
        self.__df = pd.read_csv(self.__dataset, na_values=self.__na_values, delimiter=self.__delimiter, names=self.__column)

    # buat fungsi untuk preprocessing data
    def __preprocessing(self, data, train=True):
        # copy object dataframe agar saat manipulasi data
        # data asli tidak berubah dan bisa digunakan lagi
        df = copy.deepcopy(data)
        # hapus na value atau value yang tidak valid
        df.dropna(axis=1, inplace=True)
        # deklarasikan nilai y
        y = None
        # jika fungsi preprocessing dipanggil untuk data train,
        if (train):
            # maka simpan isi dari kolom target
            # untuk digunakan sebagai testing
            y = df[self.__target]
            # kemudian hapus kolom target
            df.drop(self.__target, axis=1, inplace=True)
        # ubah data kategorikal menjadi numerik
        # karena model pada machine learning tidak bisa memproses data kategorikal
        df = pd.get_dummies(df, dtype=float)
        # simpan nama kolomnya jika train set sebagai acuan prediksi
        columns = df.columns
        # jika data training
        if (train):
            # simpan nama kolomnya ke attribute class
            self.__columns = columns
        # jika fungsi preprocessing dipanggil bukan untuk data train
        if not train:
            # ambil semua kolom
            for column in columns:
                # hapus kolom yang tidak sesuai
                if column not in self.__columns:
                    df.drop(column, axis=1, inplace=True)
            # buat kolom baru hasil one hot encoding
            missing = set(self.__columns) - set(columns)
            # isi kolom baru dengan nilai 0
            for column in missing:
                df[column] = float(0)
            # urutkan kolomnya seperti data training
            df = df.reindex(columns=self.__columns)
        # definisikan ulang kolomnya jika terdapat perubahan
        columns = df.columns
        # jika fungsi preprocessing dipanggil untuk data train
        if (train):
            # maka lakukan penghitungan scaling atau normalisasi data
            # dari data training
            self.__scaler.fit(df)
        # masukkan data untuk dilakukan normalisasi
        # simpan ke variabel X
        X = self.__scaler.transform(df)
        result = pd.DataFrame(data=X, columns=columns)
        # kembalikan data hasil preprocessing dan data target
        return result, y

    # buat fungsi read untuk mendapatkan contoh data dari dataframe
    def read(self, total = None):
        # jika diisi berapa total data yang di akan tampilkan,
        result = None
        if (total != None):
            # maka kembalikan data sebanyak yang dipassing pengguna 
            result = self.__df.sample(total)
        else:
            # selain itu dapatkan sebanyak default(1)
            result = self.__df.sample()
        # kembalikan data sample dalam format json dan tabel(1 field 1 value)
        return result.to_json(orient='records')

    # buat fungsi format untuk melakukan formatting data
    # sebelum melakukan prediksi
    def format(self, data):
        # kembalikan data dalam bentuk dataframe
        return pd.DataFrame([data])

    # buat fungsi predict untuk membuat prediksi dari data yang diberikan
    def predict(self, data):
        # lakukan preprocessing untuk dataset
        df, ydf = self.__preprocessing(self.__df)
        # lakukan preprocessing untuk data prediksi atau yang diinput
        dt, ydt = self.__preprocessing(data, False)
        # buat model LogisticRegression
        model = LogisticRegression(random_state=43)
        # isi dan hitung model dari dataset untuk training
        model.fit(df, ydf)
        # prediksi hasil
        result = model.predict(dt)
        # ubah hasil prediksi ke list
        list = result.tolist()

        # kembalikan hasil prediksi dalam bentuk json
        return json.dumps({'result': list})

    # buat fungsi getScore untuk mendapatkan nilai akurasi
    # dari model yang digunakan
    def getScore(self, size=0.2):
        # lakukan preprocessing untuk dataset
        df, y = self.__preprocessing(self.__df)
        # pecah dataset menjadi 2 bagian, yaitu train dan test
        # memecah dataset dilakukan sesuai proporsi yang diberikan user
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=size, random_state=42)
        # buat model logistic regression
        model = LogisticRegression(random_state=43)
        # isi dan hitung model dari dataset untuk training
        model.fit(X_train, y_train)
        # lakukan prediksi dari semua data testing
        y_pred = model.predict(X_test)
        # ukur hasil prediksi dengan data aktual
        # tambahkan parameter output_dict untuk menghilangkan spasi
        # karena outputnya akan dipassing ke format json
        report = classification_report(y_test, y_pred, output_dict=True)
        # kembalikan hasil prediksi dalam bentuk json
        return json.dumps({'result': report})