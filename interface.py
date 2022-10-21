from flask import Flask,render_template,jsonify,request
app=Flask(__name__)
from util import Iris
@app.route('/')
def home():
    return jsonify({"Hello":'We are at home page'})

@app.route('/predict')
def predict():
    data=request.form
    SepalLengthCm=data['SepalLengthCm']
    SepalWidthCm=data['SepalWidthCm']
    PetalLengthCm=data['PetalLengthCm']
    PetalWidthCm=data['PetalWidthCm']
    pred=Iris(SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm)
    prediction=Iris.predict(pred)
    return jsonify({"Output":f"Predicted class will be {prediction}"})
if __name__=="__main__":
    app.run(port='5555',debug=True)
