import os
import pickle
from typing import Dict

import numpy as np
import skimage.io
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, Response, request
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

apparels = ["tshirt", "shirt", "dress", "pants", "shorts", "skirt"]
IMG_SIZE = 224
BATCH_SIZE = 32

def process_image(image_path, img_size=IMG_SIZE):
    """
    Take an image file path and turn image into a Tensor.
    """
    image = tf.io.read_file(image_path)  # Read image file
    image = tf.image.decode_jpeg(image, channels=3)  # Turn the image into 3 channels RGB
    image = tf.image.convert_image_dtype(image, tf.float32)  # Turn the value 0-255 to 0-1
    image = tf.image.resize(image, size=[img_size, img_size])  # Resize the image to 224x224
    return image


def get_image_label(image_path, label):
    """
    Take an image file path name and the associated label,
    process the image and return a tuple of (image,label)
    """
    image = process_image(image_path)
    return image, label


def create_data_batches(X, batch_size=BATCH_SIZE):
    """
    Create batches of data out of image (X) and lebel (y) pairs.
    Shuffles the data if it's training data but doesn't shuffle if it's validation data.
    Also accepts test data as input (no labels).
    """
    print("Creating testing data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))  # Only file path (no label)
    data_batch = data.map(process_image).batch(batch_size)
    return data_batch


def cloth_matching(image, apparel_histogram):
    """
    params:
        image -> 3D numpy array of query image
    return:
        filenames -> List[str] of possible apparel paths in database
    """
    image = np.histogram(image, bins=256, range=(0, 1))[0]
    matching_dict = {}
    for name in apparel_histogram:
        score = cosine_similarity([image], [apparel_histogram[name]])
        matching_dict[name] = score[0][0]
    matching_dict = dict(sorted(matching_dict.items(), key=lambda x: x[1], reverse=True))
    return matching_dict


def init_model():
    global model
    model = tf.keras.models.load_model("models/apparel-classification.h5", custom_objects={'KerasLayer': hub.KerasLayer})


def get_results(filepath: str) -> Dict:
    image_to_predict = create_data_batches([filepath])
    prediction = model.predict(image_to_predict)
    output_from_classifier_model = apparels[np.argmax(prediction)]

    image = skimage.io.imread(filepath, as_gray=True)
    with open(f'histogram/apparel_histogram_dict.pkl', 'rb') as f:
        apparel_histogram_dict = pickle.load(f)
    return cloth_matching(image, apparel_histogram_dict[output_from_classifier_model])


def process_result(results: Dict) -> Dict:
    new_result = {}
    max_result = 5
    if len(results) < max_result:
        max_result = len(results)
    sub_result = sorted(results, key=results.get, reverse=True)[:max_result]
    for key in sub_result:
        new_result[key.split("/")[1]] = results[key]
    return new_result


@app.route("/health", methods=["GET"])
def index():
    return Response(status=200)


@app.route("/predict", methods=["POST"])
def predict():
    response = {}
    
    file = request.files['image']
    filename = file.filename
    file.save(filename)

    response['predictions'] = process_result(get_results(filename))
    if os.path.isfile(filename):
        os.remove(filename)
    return response, 200


if __name__ == '__main__':
    init_model()
    app.run(debug=False, port=9000)
