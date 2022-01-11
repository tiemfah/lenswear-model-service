import io
import os
from pickle import load

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
    model = models.load_model("assets/shirt.h5")
    print('Model loaded.')

def load_classes():
    global apparel_ids
    with open("assets/shirt_classes.pickle", 'rb') as f:
        apparel_ids = load(f)
    print('Classes loaded.')

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
    results = model.predict(process_image)
    response['predictions'] = []
    for i in range(len(results[0])):
        row = {'label': apparel_ids[i],
               'probability': f"{float(results[0][i])*100:.2f}%"}
        response['predictions'].append(row)
        response['success'] = True
    return response, 200

if __name__ == '__main__':
    init_model_server()
    app.run(debug=False, port=9000)
