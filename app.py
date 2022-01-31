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
from skimage import metrics
from scipy.linalg import norm
from scipy import sum, average
import cv2

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


def normalize(arr):
    rng = arr.max() - arr.min()
    amin = arr.min()
    return (arr - amin) * 255 / rng


def compare_images(img1, img2):
    # normalize to compensate for exposure difference, this may be unnecessary
    img1 = normalize(img1)
    img2 = normalize(img2)
    # calculate the difference and its norms
    diff = img1 - img2  # elementwise for scipy arrays
    m_norm = sum(abs(diff))  # Manhattan norm
    z_norm = norm(diff.ravel(), 0)
    return m_norm, z_norm


def to_grayscale(arr):
    """
    If arr is a color image (3D array), convert it to grayscale (2D array).
    """
    if len(arr.shape) == 3:
        return average(arr, -1)  # average over the last axis (color channels)
    else:
        return arr


def cloth_matching(image, filenames: List) -> Dict[str, float]:
    """
    params:
        image -> 3D numpy array of query image
    return:
        filenames -> List[str] of possible apparel paths in database
    """
    image = to_grayscale(image.astype(float))
    image = cv2.resize(image, (600, 600))
    matching_dict = {}
    img_list = [imread(filename) for filename in filenames]
    for i, img in enumerate(img_list):
        img = to_grayscale(img.astype(float))
        img = cv2.resize(img, (600, 600))
        score, diff = metrics.structural_similarity(image, img, full=True)
        matching_dict[filenames[i]] = score
    return matching_dict


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
