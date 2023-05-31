from flask import Flask
from flask import request
from flask import send_file
import os
import sys
from subprocess import Popen
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt 
from flask import jsonify
import json
from pathlib import Path
# from flask_ngrok import run_with_ngrok
# pip install Flask-Caching
# from flask_caching import Cache

app = Flask(__name__)
# run_with_ngrok(app)
# cache = Cache(app)S
file_path = "E://Remove_Object_Bhsoft"

@app.route("/")
def home():
    return('Hello')

@app.route("/index", methods=["POST"])
def index():
    index = request.form["index"]
    path = file_path + "/u7/seg/runs/predict-seg/index.txt"
    with open(path, "w") as f:
        f.write(str(index))
        f.close()
    return("OK")

@app.route("/post", methods=["POST"])
def post():
    index = file_path + "/u7/seg/runs/predict-seg/index.txt"
    with open(index, "w") as f:
        f.write("0")
        f.close()
    image = request.files['file'] 
    filepath = os.path.join(file_path + "/u7/image_test", image.filename)
    image.save(filepath)

    process = Popen(["python",file_path +  "/u7/seg/segment/predict.py", "--weights",file_path + "/u7/pretrain/yolov7-seg.pt","--source",filepath], shell=True)
    process.wait()

    num_obj = open(file_path + "/u7/seg/runs/predict-seg/num_obj.txt", "r")
    num = num_obj.read()
    num_obj.close()

    return(num)

@app.route("/sendfile")
def sendfile():
    return send_file(file_path + '/u7/seg/runs/predict-seg/delete_object/delete.jpg')

@app.route("/segment",methods=["POST"])
def segment():
    index = request.form["index"]
    file_name = request.form["file_name"]
    path = file_path + "/u7/seg/runs/predict-seg/index.txt"
    with open(path, "w") as f:
        f.write(str(index))
        f.close()
    filepath = file_path + "/u7/image_test/" + file_name
    process = Popen(["python",file_path + "/u7/seg/segment/predict.py", "--weights",file_path + "/u7/pretrain/yolov7-seg.pt","--source",filepath], shell=True)
    process.wait()
    return ("OK")


@app.route("/send")
def send():
    process = Popen(["python",file_path + "/edge-connect/test.py", "--checkpoints",file_path + "/edge-connect/scripts/checkpoints/places2","--input",file_path + "/u7/seg/runs/predict-seg/delete_object/delete.jpg","--mask",file_path + "/u7/seg/runs/predict-seg/mask/mask.jpg","--output","E:/Remove_Object_Bhsoft/edge-connect/result"], shell=True)
    process.wait()
    image = np.array(Image.open(file_path + '/edge-connect/result/delete.jpg'))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imsave(file_path + '/edge-connect/result/delete.jpg',image)
    return send_file(file_path + '/edge-connect/result/delete.jpg')

@app.route("/blur",methods=["POST"])
def blur():
    file_name = request.form["file_name"]
    img = cv2.imread(file_path + "/u7/image_test/" + file_name)
    blurred_img = cv2.GaussianBlur(img, (21, 21), 0)
    mask = cv2.imread(file_path + "/u7/seg/runs/predict-seg/mask/mask.jpg")

    output = np.where(mask!=np.array([255, 255, 255]), blurred_img, img)
    cv2.imwrite("C://Users/84852/Downloads/output.png", output)

    return send_file("C://Users/84852/Downloads/output.png")



if __name__ == "__main__":
    app.run(host="0.0.0.0")     