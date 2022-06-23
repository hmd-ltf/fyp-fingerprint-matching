from flask import Flask
from flask import request
from flask import jsonify
import urllib
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, World!"

@app.route("/match" , methods=["GET"])
def matchImages():
    data = request.get_json()
    orignal_image_url = data.get("orignal")
    match_image_url = data.get("match")

    orignal_image_url =  urllib.request.urlopen(orignal_image_url)
    match_image_url =  urllib.request.urlopen(match_image_url)

    orignal_img = np.array(bytearray(orignal_image_url.read()), dtype=np.uint8)
    match_img = np.array(bytearray(match_image_url.read()), dtype=np.uint8)

    orignal_img = cv2.imdecode(orignal_img , 0)
    match_img = cv2.imdecode(match_img , 0)

    orignal_img = cv2.medianBlur(orignal_img , 3)
    match_img = cv2.medianBlur(match_img , 3)


    orignal_edges = cv2.Canny(orignal_img , 100 , 200)
    match_edges = cv2.Canny(match_img , 100 , 200)

    
    #SIFT Matching
    sift = cv2.SIFT_create()

    kp1 , des1 = sift.detectAndCompute(orignal_edges , None) 
    kp2 , des2 = sift.detectAndCompute(match_edges , None) 


    matches = cv2.FlannBasedMatcher({"algorithm":1, "trees":10}, 
                {}).knnMatch(des1, des2, k=2)

    match_points = []

    did_match = False

    for p, q in matches:
        if p.distance < 0.8*q.distance:
            match_points.append(p)
    keypoints = 0
    if len(kp1) <= len(kp2):
        keypoints = len(kp1)            
    else:
        keypoints = len(kp2)

    if (len(match_points) / keypoints) > 0.15:
        print((len(match_points) / keypoints))
        did_match = True
    
    print(len(match_points) , len(kp1) , len(kp2))


    return jsonify(did_match)

if __name__ == '__main__':
    app.run(debug=False)