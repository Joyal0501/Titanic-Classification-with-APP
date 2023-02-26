from flask import Flask, request, jsonify
import pickle
import numpy as np
import sklearn
import pandas as pd

model = pickle.load(open('Titanic.pkl','rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return "Hello World"


@app.route('/predict', methods=['POST'])
def predict():
    Sex = request.form.get('Sex')
    Age = request.form.get('Age')
    Pclass = request.form.get('Pclass')
    Sibsp = request.form.get('Sibsp')
    Parch = request.form.get('Parch')
    Embarked = request.form.get('Embarked')
    Fare = request.form.get('Fare')

    input_query = pd.DataFrame([[Sex,Age,Pclass,Sibsp,Parch,Embarked,Fare]])
    #print(input_query)
    #result = {'Sex':Sex,'Age':Age,'Class': Pclass,'Siblings':Sibsp,'Adults':Parch,'Embark': Embarked,'Fare':Fare}
    result = model.predict(input_query)[0]

    #print(result)

    return jsonify({'Prediction':str(result)})

if __name__ == '__main__':
    app.run(debug=True)
