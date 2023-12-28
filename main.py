import os
import random
import json

from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, session
from libs.Classifier import LRClassifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['EXTENSION'] = 'csv'
app.secret_key = '12345'

# Define a route for the root URL ("/")
@app.route('/')
def index():
    return 'Hello, this is BedahKomputerID website.'

@app.route('/upload', methods=['POST'])
def upload():
    params = json.loads(request.form.get('data'))

    if 'file' not in request.files:
        return jsonify({'message': 'File csv tidak ada.'})

    file = request.files['file']

    random_number = random.randrange(100000000, 999999999)
    key = str(random_number)

    new_filename = f"{key}.{app.config['EXTENSION']}"

    file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(new_filename)))

    session[key] = {
        'delimiter': params['delimiter'],
        'columns': params['columns'],
        'na': params['na'],
        'target': params['target']
    }

    return jsonify({
        'message': f'File berhasil disimpan. ID: "{key}"',
        'data': key
    })

@app.route('/get/<key>', methods=['GET'])
def score(key):
    sess = session.get(key, None)

    path = f"{app.config['UPLOAD_FOLDER']}/{key}.{app.config['EXTENSION']}"
    model = LRClassifier(path, ',', sess['columns'], sess['na'], sess['target'])

    app.logger.info(request.args.get('row'))

    if request.args.get('row'):
        return model.read(int(request.args.get('row')))

    return model.read()

@app.route('/predict/<key>', methods=['POST'])
def predict(key):
    params = json.loads(request.form.get('data'))

    sess = session.get(key, None)

    path = f"{app.config['UPLOAD_FOLDER']}/{key}.{app.config['EXTENSION']}"
    model = LRClassifier(path, ',', sess['columns'], sess['na'], sess['target'])

    predict_data = {}
    for element in sess['columns']:
        if element not in [sess['target']]:
            predict_data[element] = params[element]

    return model.predict(model.format(predict_data))

@app.route('/score/<key>', methods=['GET'])
def getScore(key):
    sess = session.get(key, None)

    path = f"{app.config['UPLOAD_FOLDER']}/{key}.{app.config['EXTENSION']}"
    model = LRClassifier(path, ',', sess['columns'], sess['na'], sess['target'])

    if request.args.get('size'):
        return model.getScore(float(request.args.get('size')))
    
    return model.getScore()

# Run the application if this script is executed
if __name__ == '__main__':
    app.run(debug=True)