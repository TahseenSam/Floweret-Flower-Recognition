from flask import Flask,render_template,request
import numpy as np
# import tensorflow as tf
import tensorflow as tf
# Keras
from keras.models import load_model
import cv2,os
from werkzeug.utils import secure_filename



app=Flask(__name__)


# Model saved with Keras model.save()
# MODEL_PATH ='InceptionV3-98-90.h5'

# Load your trained model
model = load_model("adamMODEL95-91.h5")
#model = load_model(MODEL_PATH)

classes=['Bluebell',
    'Buttercup',
    'BishopOfIlandaff',
    'Crocus',
    'Daffodils',
    'Daisy',
    'Dandelion',
    'Fritillary',
    'Iris',
    'Rose',
    'Snowdrop',
    'Tigerlily',
    'Gazania',
    'Sunflower',
    'Jasmine',
    'Periwinkle',
    'CrownOfThorns',
    'BlanketFlower']



def model_predict(img_path, model):
    print(img_path)
    img = cv2.imread(img_path)

    # Preprocessing the image
    x=cv2.resize(img,(256,256))
    x=cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
    x=np.expand_dims(x,axis=0)
    preds=model.predict(x)
    preds="The Flower Is "+classes[np.argmax(preds)]    
    desc="Dummy Flower Description"
    
    # preds = "Your Image Shape is "+str(x.shape)
    return preds

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/testModel')
def test():
    return render_template('testModelBackup.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
        # return render_template('result.html',result=preds)
    return None



if __name__ == '__main__':
    app.run(debug=True)