import numpy as np
import os
from flask import Flask, request, jsonify, render_template
import pickle
from werkzeug.utils import secure_filename
from pathlib import Path

flask_app = Flask(__name__)

abspath = Path('model.pkl')


with open(abspath, 'rb') as savefile:
    model = pickle.load(savefile)

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    features1= request.form["Maximum Wind"]
    features2= request.form["Minimum Pressure"]
    features3= request.form["Latitude"]
    features4= request.form["Longitude"]
    features5= request.form["Low Wind NE"]
    features6= request.form["Low Wind NW"]
    
    features = np.array([[features1,features2,features3,features4,features5,features6]])
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "Forecasted event is {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)
