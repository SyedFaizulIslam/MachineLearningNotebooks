from flask import Flask, request, redirect, url_for, flash, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import tensorflow as tf 
import json
import tensorflow.keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras.models import model_from_json
import pandas as pd

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def init():
    global model
    modelfile = 'tensorflowmodel.json'
    json_file = open(modelfile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    # load model and weights
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model = model_from_json(loaded_model_json)
        model.load_weights("tensorflowmodel.h5")
        print("Model Loaded...")
        print (model.summary())


@app.route('/api/', methods=['POST'])
@cross_origin()
def predictiptrend():
    data = request.get_json()
    ContractType=data['ContractType']
    df=PrepInput(ContractType)
    prediction = model.predict(df).tolist()
    res=np.round(prediction[0],2)
    return str(res)

def PrepInput(ContractType):
   #Prep pandas dataframe to hold independent values required to predict.
   IndependentValue1=234
   IndependentValue2=245
   IndependentValue3=210
   df= pd.DataFrame({'IndependentVariable1': IndependentValue1, 'IndependentVariable2': IndependentValue2, 'IndependentVariable1':IndependentValue3}, index=[0])
   return df
if __name__ == '__main__':
    init()
    df9=PrepInput('18')
    print (df9.head(1))
    result = model.predict(df9.round(2)).tolist()
    print('Prediction on sample dataframe: '+ str(np.round(result[0],2)))
    app.run(host='0.0.0.0')
    #app.run()
