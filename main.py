# import library yang diperlukan
import os
import random
import json

from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, session
from libs.Classifier import LRClassifier

# buat instance flask
app = Flask(__name__)
# deklarasikan konstanta agar mudah dalam mengubah valuenya
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['EXTENSION'] = 'csv'
# buat secret key
# ini hanya contoh saja, agar lebih mudah
app.secret_key = '12345'

# buat default atau root url
@app.route('/')
# buat method index untuk url root
def index():
    # kembalikan string untuk ditampilkan
    return 'Hello, this is BedahKomputerID website.'

# buat route baru dengan url /upload dan method POST
# untuk menyimpan dataset
@app.route('/upload', methods=['POST'])
# buat method upload
def upload():
    # dapatkan parameter "data"
    params = json.loads(request.form.get('data'))
    # cek apakah ada file dataset atau tidak
    if 'file' not in request.files:
        # jika tidak ada, kembalikan pesan error berupa JSON
        return jsonify({'message': 'File csv tidak ada.'})
    # dapatkan file dataset yang dikirim
    file = request.files['file']
    # buat angka random
    random_number = random.randrange(100000000, 999999999)
    # konversi menjadi tipe data string
    key = str(random_number)
    # buat nama file dari angka yang sudah dibuat dengan ekstensi .csv
    new_filename = f"{key}.{app.config['EXTENSION']}"
    # simpan file ke folder uploads
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(new_filename)))
    # simpan data parameter dataset ke dalam session
    session[key] = {
        'delimiter': params['delimiter'],
        'columns': params['columns'],
        'na': params['na'],
        'target': params['target']
    }
    # kembalikan response berisi pesan dan key dari dataset yang tersimpan
    return jsonify({
        'message': f'File berhasil disimpan. ID: "{key}"',
        'data': key
    })

# buat route baru dengan url /get
# untuk mendapatkan beberapa data dari dataset
@app.route('/get/<key>', methods=['GET'])
# buat method untuk mendapatkan data
def get(key):
    # ambil data dari session sesuai key yang diberikan
    sess = session.get(key, None)
    # buat path untuk mengambil file dari penyimpanan
    path = f"{app.config['UPLOAD_FOLDER']}/{key}.{app.config['EXTENSION']}"
    # buat instance dari class Classifier
    model = LRClassifier(path, ',', sess['columns'], sess['na'], sess['target'])
    # jika terdapat parameter "row" pada request
    if request.args.get('row'):
        # kembalikan data sebanyak yang diminta
        return model.read(int(request.args.get('row')))
    # jika tidak, kembalikan sebanyak nilai default dari classnya
    return model.read()

# buat route baru dengan url /predict 
# untuk mendapatkan hasil prediksi dari data/parameter yang diberikan
@app.route('/predict/<key>', methods=['POST'])
# buat method untuk mendapatkan hasil prediksi
def predict(key):
    # dapatkan parameter "data"
    params = json.loads(request.form.get('data'))
    # dapatkan data dari session sesuai key yang dikirim
    sess = session.get(key, None)
    # buat path menuju file dataset sesuai key
    path = f"{app.config['UPLOAD_FOLDER']}/{key}.{app.config['EXTENSION']}"
    # buat instance class Classifier
    model = LRClassifier(path, ',', sess['columns'], sess['na'], sess['target'])
    # deklarasikan variabel untuk menampung parameter prediksi
    predict_data = {}
    # loop untuk semua kolom dari csv
    for element in sess['columns']:
        # hanya ambil selain kolom target
        if element not in [sess['target']]:
            # masukkan data yang dipassing ke dalam variabel penampungan
            predict_data[element] = params[element]
    # kembalikan hasil predict dengan library Classifier
    return model.predict(model.format(predict_data))

# buat route baru dengan url /score
# untuk mendapatkan nilai akurasi dari dataset
@app.route('/score/<key>', methods=['GET'])
# buat method baru untuk route score
def score(key):
    # dapatkan data dari session sesuai key yang dikirim
    sess = session.get(key, None)
    # buat path menuju file dataset sesuai key
    path = f"{app.config['UPLOAD_FOLDER']}/{key}.{app.config['EXTENSION']}"
    # buat instance class Classifier
    model = LRClassifier(path, ',', sess['columns'], sess['na'], sess['target'])
    # jika terdapat parameter "size" pada request
    if request.args.get('size'):
        # kembalikan score dari prediksi dataset sebanyak ukuran data test yang dikirim
        return model.getScore(float(request.args.get('size')))
    # jika tidak, kembalikan sebanyak nilai default dari classnya
    return model.getScore()

# jalankan program yang dibuat
if __name__ == '__main__':
    # buat dan jalankan instance flask
    app.run(debug=True)