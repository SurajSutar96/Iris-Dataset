import pandas as pd
import numpy as np
import pickle
import json
class Iris():
    def __init__(self,a,b,c,d):
        self.SepalLengthCm=a
        self.SepalWidthCm=b
        self.PetalLengthCm=c
        self.PetalWidthCm=d
    def data(self):
        with open('Iris_model.pkl','rb')as f:
            self.model=pickle.load(f)
        with open('Iris_data.json','r')as f:
            self.data=json.load(f)
    def predict(self):
        self.data()
        array=np.zeros(len(self.data['columns']))
        array[0]=self.SepalLengthCm
        array[0]=self.SepalWidthCm
        array[0]=self.PetalLengthCm
        array[0]=self.PetalWidthCm
        pred=self.model.predict([array])[0]
        print("Predicted Class will be:-",pred)
        return pred
if __name__=="__main__":
    SepalLengthCm=7.8
    SepalWidthCm=9.8
    PetalLengthCm=6.6
    PetalWidthCm=3.2
    obj=Iris(SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm)
    obj.predict()

