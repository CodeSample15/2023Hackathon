from flask import Flask, render_template, url_for, Response, request
import cv2
import requests
import datetime, time
import os, sys
import numpy as np
import json
from tensorflow.keras.models import load_model
import shutil
import os
from pydub import AudioSegment
from PIL import Image
import threading
from ModelRunner import Recorder, Runner
import time

app = Flask(__name__)

recorder = Recorder()
recorder.start_recording(hidden_preview=True)
#runner = Runner(model_path='../../Machine Learning/Models/v3.keras', input_frame_count=20)
recording = False

def gen_frames():  # generate frame by frame from camera
    global recorder
    
    while True:
        if not recording:
            recorder.clear_frames()
        temp = recorder.preview
        if not np.array_equal(temp, np.zeros((1, 1, 1))):
            temp = cv2.resize(temp, (500, 500), interpolation=cv2.INTER_AREA)
            ret,buffer=cv2.imencode('.jpg', temp)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/about')
def about():
    return render_template('about.html')
            
@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera,recorder, recording
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
        elif  request.form.get('grey') == 'Grey':
            global grey
            grey=not grey
        elif  request.form.get('neg') == 'Negative':
            global neg
            neg=not neg
        elif  request.form.get('face') == 'Face Only':
            global face
            face=not face 
            if(face):
                time.sleep(4)   
        elif  request.form.get('stop') == 'Stop/Start':
            recording = not recording
            if not recording:
                try:
                    pass
                    #x = threading.Thread(target = runner.run, args=(recorder.export_frames(),))
                    #x.start()
                except:
                    print("oops")

        elif  request.form.get('rec') == 'Start/Stop Recording':
            recording = not recording
            if not recording:
                try:
                    pass
                    #x = threading.Thread(target = runner.run, args=(recorder.export_frames(),))
                    #x.start()
                except:
                    print("oops")         
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
