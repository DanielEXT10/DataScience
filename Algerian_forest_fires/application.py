import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app = application



## import ridge regressor and standard scaler pickle
ridge_model = pickle.load(open('models/ridge.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata", methods = ['GET','POST'])
def predict_datapoint():
    if request.method == "POST":
        try:
            #Extract features from form
            Temperature = float(request.form['Temperature'])
            RH = float(request.form['RH'])
            WS = float(request.form['WS'])
            Rain = float(request.form['Rain'])
            FFMC = float(request.form['FFMC'])
            DMC = float(request.form['DMC'])
            ISI = float(request.form['ISI'])
            Classes = float(request.form['Classes'])
            Region = float(request.form['Region'])

            new_data_scaled = standard_scaler.transform([[Temperature,RH,WS,Rain,FFMC,DMC,ISI,Classes,Region]])
            result= ridge_model.predict(new_data_scaled)
            
            return render_template('home.html', result=result[0])
        
        except Exception as e:
            return render_template('home.html', result=f"Error: {str(e)}")

    else:
        return render_template('home.html')
def index():
    return render_template('index.html')

if __name__== "__main__":
    app.run(host="0.0.0.0")