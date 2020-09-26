from flask import Flask,request,redirect,url_for,flash,jsonify
from flask_cors import CORS,cross_origin
import joblib
import numpy as np
import pandas as pd 

app= Flask(__name__)
cors=CORS(app)
app.config['CORS_HEADERS']='Content-Type'

def init():
    global model
    modelfile='HousePricingModel.pkl'
    model=joblib.load(modelfile)

@app.route('/api/',methods=['GET'])
def helloworld():
    return "Hello World!!"

@app.route('/api/PredictHousePrice',methods=['POST'])
def PredictHousePrice():
    data=request.get_json()
    YearBuilt=data['YearBuilt']
    YearRemodAdd=data['YearRemodAdd']
    TotalBsmtSF=data['TotalBsmtSF']
    FirstFlrSF=data['1stFlrSF']
    GrLivArea=data['GrLivArea']
    GarageArea=data['GarageArea']
    df=pd.DataFrame({'YearBuilt':YearBuilt,'YearRemodAdd':YearRemodAdd,'TotalBsmtSF':TotalBsmtSF,'1stFlrSF':FirstFlrSF,'GrLivArea':GrLivArea,'GarageArea':GarageArea})
    prediction=model.predict(df).tolist()
    result=np.round(prediction[0],2)
    return str(result)

if __name__=='__main__':
    init()
    print('Model initialize')
    app.run()
