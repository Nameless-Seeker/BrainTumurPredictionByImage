import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
import numpy as np
from PIL import Image
import io

# Brain tumour detection
disease = FastAPI()

# Standard, clean load
model = tf.keras.models.load_model("brain.keras")

def PredictBrainTumour(image):
    # Preprocess
    image = image.resize((64,64))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model.predict(image_array)[0]
    return prediction

@disease.post("/BrainPrediction")
async def Brain(file: UploadFile = File(...)):
    contents = await file.read()

    # Convert image into model readable image
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Getting the prediction confidence
    prediction = PredictBrainTumour(image)

    # Converting the confidence to integer %
    tumour = int(prediction[1]*100)
    no_tumour = int(prediction[0]*100)

    result = {}

    # Comparing the results
    if(tumour > no_tumour):
        result['result'] = "You have tumour"
        result['chances'] = tumour
    else:
        result['result'] = "You do not have tumour"
        result['chances'] = no_tumour

    print(result)

    return result