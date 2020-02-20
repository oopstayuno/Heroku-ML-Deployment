from flask import render_template, request, jsonify
from flask import Flask
import flask
import numpy as np
import traceback
import pickle
import pandas as pd
 
 
# App definition
app = Flask(__name__)
 
# importing models
with open('model/model.pkl', 'rb') as f:
	classifier = pickle.load (f)
 
with open('model/model_columns.pkl', 'rb') as f:
	model_columns = pickle.load (f)
 
 
@app.route('/')
def home():
	return render_template("index.html")
 
@app.route('/predict', methods=['POST','GET'])
def predict():

	if flask.request.method == 'GET':
		return render_template("predict.html")

	if flask.request.method == 'POST':

		int_features = [int(x) for x in request.form.values()]

		price_ = classifier.predict(np.array(int_features).reshape(-1,1).T)

		return render_template("predicted.html", prediction_text="%.2f" % price_[0])



if __name__ == "__main__":
	app.run()