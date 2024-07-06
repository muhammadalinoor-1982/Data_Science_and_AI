# Flask API, main.py
from flask import Flask,jsonify,request
import os
from lrmodel import LRModel

LR = LRModel()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])

def predict_flowers():
    
    SepalLength = float(request.form.get('SepalLength')) 
    SepalWidth  = float(request.form.get('SepalWidth'))
    PetalLength = float(request.form.get('PetalLength')) 
    PetalWidth  = float(request.form.get('PetalWidth'))
    
    if os.path.exists('File/lr_model.pickle'):
        y_pred = LR.test(SepalLength, SepalWidth, PetalLength, PetalWidth)
        return jsonify({'result' : y_pred})
    else:
        LR.train()
        y_pred = LR.test(SepalLength, SepalWidth, PetalLength, PetalWidth)
        return jsonify({'result' : y_pred})

if __name__ == '__main__':
    app.run(debug=True)