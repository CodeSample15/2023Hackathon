from flask import Flask, render_template, Response
from ModelRunner import Recorder
import cv2

app = Flask(__name__)
recorder = Recorder()

def generate_frames():
    while True:
        recorder.clear_frames()
        frame = recorder.preview
        ret,buffer=cv2.imencode('.jpg',frame)
        frame = buffer.tobytes()

        yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n0' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('Simple Front End/index.html')

@app.route('/video')
def video():
    return Response(generate_frames, mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    recorder.start_recording(hidden_preview=True)
    app.run(debug=True)