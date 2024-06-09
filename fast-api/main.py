import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from fastapi import FastAPI, UploadFile,File 
import uvicorn 
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf  

from typing import Union
import uuid 

app = FastAPI()

model_path = "C:/Users/Arnav Singh/OneDrive/Desktop/Potato Disease/models/new_model.keras"

DL_model = tf.keras.models.load_model(model_path)

class_names = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]

@app.get("/wtf/{name}")

async def wtf (name):
    return {f"fuckererrrr chal jaaaaaaaaaaa {name}"}


def read_file_as_image(bytes) -> np.ndarray:
    image = np.array(Image.open(BytesIO(bytes)))
    return image



@app.post("/prediction")
async def prediction(file: UploadFile = File(...) ):
    
    content = read_file_as_image(await file.read())
    
    image_batch = np.expand_dims(content , 0)
    
    predict = DL_model.predict(image_batch)
    class_predicted = class_names[np.argmax(predict[0])]
    most_accurate = np.max(predict[0])
    
    return {
        'class': class_predicted,
        'confidence': float(most_accurate)
    }

    
     

   
    
    

    
    

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)