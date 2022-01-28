import io
import os
from pickle import load
from typing import Dict, List

from flask import Flask, Response, request
from google.cloud import storage
from keras import models
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
from PIL import Image

app = Flask(__name__)
model = None

def download_blob(bucket_name):
    blobs = storage.Client(project="lenswear-service").bucket(bucket_name).list_blobs()

    if not os.path.exists('assets'):
        os.makedirs('assets')

    for blob in blobs:
        print(f'Downloading {blob.name}...')
        destination_uri = f'assets/{blob.name}'
        blob.download_to_filename(destination_uri)
        print(f'Finish downloading {blob.name}.')

def load_models():
    global model
    model = models.load_model("assets/apparel_list.h5")
    print('Model loaded.')

def load_classes():
    global apparel_ids
    with open("assets/apparel_list.pkl", 'rb') as f:
        apparel_ids = load(f)
    print('Classes loaded.')

def cloth_matching(classification_results: List, img: Image) -> Dict[str, float]:
    """
    WRITE YOUR FUNCTION HERE
    WRITE YOUR FUNCTION HERE
    WRITE YOUR FUNCTION HERE
    WRITE YOUR FUNCTION HERE
    WRITE YOUR FUNCTION HERE
    WRITE YOUR FUNCTION HERE
    WRITE YOUR FUNCTION HERE
    params:
        results -> classification result from your model
    return:
        List[str] of possible apparel ids in database
    """
    return {"apparel_shirt_1": 0.6, "apparel_shirt_2": 0.4}

def preprocessing_image(img, target_size):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img = img_to_array(img)
    img = expand_dims(img, axis=0)
    return img

def init_model_server():
    # download_blob("lenswear-service-model")
    load_models()
    load_classes()
    pass


@app.route("/health", methods=["GET"])
def index():
    return Response(status=200)

@app.route("/predict", methods=["POST"])
def predict():
    response = {'success': False}
    image_file = request.files['image'].read()
    img = Image.open(io.BytesIO(image_file))
    process_image = preprocessing_image(img, target_size=(256, 256))
    classification_results = model.predict(process_image)
    matching_result = cloth_matching(classification_results, img)
    response['predictions'] = matching_result
    response['success'] = True
    return response, 200

if __name__ == '__main__':
    init_model_server()
    app.run(debug=False, port=9000)
