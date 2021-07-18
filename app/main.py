import joblib
import uvicorn
import numpy as np
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI
import joblib

app = FastAPI(title='spectral class prediction', version='1.0',
              description='spectral class prediction using machine learning')

le  = joblib.load('../model/label_encoder.joblib')
clf = joblib.load('../model/lgb_model_1.0.joblib')
features = joblib.load('../model/features.joblib')
categorical_features = joblib.load('../model/categorical_features.joblib')
scaler = joblib.load('../model/minmax_scaler.joblib')

class schema(BaseModel):
    temperature: float
    luminosity: float
    radius: float
    absolute_magnitude: float
    star_color: str
    spectral_class: str

@app.get('/')
@app.get('/home')
def read_home():
    
    """
     Home endpoint which can be used to test the availability of the application
    """

    return {'message': 'system up'}

@app.post("/predict")
def predict(data: schema):
    data_dict = data.dict()
    data_df = pd.DataFrame.from_dict([data_dict])
    data_df = data_df[features]
    data_df[categorical_features] = le.transform(data_df[categorical_features])
    
    data_df = scaler.transform(data_df)
    print(data_df, flush = True)
    
    prediction = clf.predict(data_df)
    print(prediction, flush=True)
    ##Red Dwarf, Brown Dwarf, White Dwarf, Main Sequence , SuperGiants, HyperGiants
    if prediction == 0:
        prediction_label = "Red Dwarf"
    if prediction == 1:
        prediction_label = "Brown Dwarf"
    if prediction == 2:
        prediction_label = "White Dwarf"
    if prediction == 3:
        prediction_label = "Main Sequence"
    if prediction == 4:
        prediction_label = "Super Giants"
    if prediction == 5:
        prediction_label = "Hyper Giants"

    return {"prediction": prediction_label}

if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    