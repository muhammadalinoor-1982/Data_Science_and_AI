from flask import Flask,jsonify,request
import os
from lrmodel import LRModel

LRM = LRModel()

app = Flask(__name__)

@app.route('/predict', methods = ['POST'])

def predict_flowers():
    
    SepalLength = request.form.get('SepalLength') 
    SepalWidth  = request.form.get('SepalWidth')
    PetalLength = request.form.get('PetalLength') 
    PetalWidth  = request.form.get('PetalWidth')
    
    if os.path.exists('File/lr_model.pickle'):
        y_pred = LRM.test(SepalLength, SepalWidth, PetalLength, PetalWidth)
        return jsonify({'result' : y_pred})
    else:
        LRM.train()
        y_pred = test(SepalLength, SepalWidth, PetalLength, PetalWidth)
        return jsonify({'result' : y_pred})