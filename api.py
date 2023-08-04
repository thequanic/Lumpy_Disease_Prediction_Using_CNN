from flask import Flask, jsonify, request
from flask_restful import Resource, Api,reqparse
import pickle
import pandas as pd
import numpy as np
from flask_cors import CORS
import tensorflow as tf
import os
import cv2

app = Flask(__name__)
api=Api(app)

CORS(app)

model=tf.keras.models.load_model("E://vsc2.0//GitHub//Cutaneous_disease_classification_using_CNN//CutaneousModel")

    
class prediction(Resource):
    
    def post(self):
        images=[]
        for i in range(len(request.files)):
                image_file = request.files['img'+str(i)]
                image_path = 'img'+str(i)+'.jpg'
                image_file.save(os.path.join("E://vsc2.0//GitHub//Cutaneous_disease_classification_using_CNN//backendApi",image_path))

                img=cv2.imread(os.path.join("E://vsc2.0//GitHub//Cutaneous_disease_classification_using_CNN//backendApi",image_path))
                
                img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)                 # Read Image as numbers
                img = cv2.resize(img,(300,300),interpolation=cv2.INTER_CUBIC)
                img=np.asarray(img,dtype='float32') / (255.0) 
                images.append(img)

        images=np.array(images)
        y=model.predict(images).flatten()
        for i in range(0,len(y)):
            if(y[i]>=0.5):
                y[i]=1.0
            else:
                y[i]=0.0
        ypred=y.astype(np.int64)
        ypred=ypred.tolist()

        for i in range(len(request.files)):
            image_path = 'img'+str(i)+'.jpg'
            os.remove(os.path.join("E://vsc2.0//GitHub//Cutaneous_disease_classification_using_CNN//backendApi",image_path))
        print(ypred)
        return jsonify({'result':ypred})
            


api.add_resource(prediction, '/predict')
  

if __name__ == '__main__':
  
    app.run(port=3000,debug = True)