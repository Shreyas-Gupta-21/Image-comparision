#from skimage.measure import structural_similarity as ssim
from skimage.metrics import structural_similarity as ssim
import numpy as np
#pip install --upgrade skimage
#from skimage.measure import compare_ssim
import skimage.util
from skimage import measure
#import argparse
#import imutils
import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path
from PIL import Image
from skimage import io
from flask import Flask,request,jsonify
import pyrebase

config = {
  "apiKey": "AIzaSyBCTULvyrph54v4daplZogA-wAXc5Wf5gI",
  "authDomain": "projectId.firebaseapp.com",
  "databaseURL": "https://karunafightersdoctors.firebaseio.com/",
  "storageBucket": "karunafightersdoctors.appspot.com",
  "serviceAccount": "karunafightersdoctors-firebase-adminsdk-bbos4-3f66d9ae94.json"
}
firebase = pyrebase.initialize_app(config)
db = firebase.database()
storage = firebase.storage()

app = Flask(__name__)
app.config['DEBUG'] = True



@app.route('/get_results',methods = ['POST'])
def get_results():
    patient_id = request.args['id']
    image = get_image(patient_id)
    return 

def get_image(patient_id):
    image_name = str(patient_id)
    print(image_name)
    storage.child('images/'+ image_name).download(image_name)
    kaam_karo(image_name)
    # image = cv2.imread(image_name)
    return preprocess_image(image)

app.run()








def kaam_karo(input_name):
    # before = plt.imread(input_name, cv2.IMREAD_GRAYSCALE)
    # after = plt.imread("C:\\Users\\AAVAIG\\Documents\\webd\\hack-the-crisis\\api\\med2.jpeg", cv2.IMREAD_GRAYSCALE)
    before_gray = io.imread(input_name, as_gray=True)
    # after_gray = io.imread("C:\\Users\\AAVAIG\\Documents\\webd\\hack-the-crisis\\api\\med2.jpeg", as_gray=True)

    print(before_gray.shape)
    print(after_gray.shape)
    # Compute SSIM between two images
    (score, diff) = skimage.metrics.structural_similarity(before_gray, after_gray, full=True)
    print("Image similarity", score)
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    mask = np.zeros(before.shape, dtype='uint8')
    filled_after = after.copy()

    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.rectangle(after, (x, y), (x + w, y + h), (0,0,255), 5)
            cv2.drawContours(mask, [c], 0, (0,255,0), -1)
            cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

    cv2.imwrite('after.jpeg',after)
