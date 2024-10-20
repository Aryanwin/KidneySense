from DataOrganization import Rescale, RandomCrop, ToTensor, transforms
from skimage import io
import time
import base64
from Eval import predictionout
from Maintest import gradout
from flask import Flask, request, Response, send_file, jsonify, abort
from PIL import Image
from matplotlib import pyplot as plt

app = Flask(__name__)

@app.route('/processing', methods=['POST'])
def process():
    file = request.files['image']
    #image = Image.open(file.stream)

    image = io.imread(file.stream)
    imgarr = {'image': image, 'label': -1}

    transform = transforms.Compose([Rescale(32), RandomCrop(32), ToTensor()])
    imagefin = transform(imgarr)

    returnpath = gradout(imagefin['image'])
    predictioncode = predictionout(imagefin['image'])

    response = send_file(returnpath, mimetype='image/jpeg')
    response.headers['X-Prediction-Value'] = predictioncode
    return response

if __name__ == "__main__":
    app.run()