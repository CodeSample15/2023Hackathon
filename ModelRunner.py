from tensorflow.keras.models import load_model

import cv2
import numpy as np
import shutil
import os

from pydub import AudioSegment
from PIL import Image
import DataProcessing
from DataProcessing import segmentMouth
import threading
import time

from tqdm import tqdm

class Recorder:
    def __init__(self, framerate=30, image_size=128):
        self.fps = framerate
        self.recording = False
        self.frames = []
        self.r_thread = threading.Thread(target=self.recording_thread)
        self.im_size = image_size

    def clear_frames(self):
        self.frames = []

    def export_frames(self):
        temp = np.array(self.frames)
        temp = np.reshape(temp, (temp.shape[0], temp.shape[1], temp.shape[2], 1))
        return temp

    def recording_thread(self, preview=False):
        vid = cv2.VideoCapture(0)
        vid.set(cv2.CAP_PROP_FPS, self.fps)

        while self.recording:
            ret, frame = vid.read()
            if ret:
                segmentMouth(frame, img_width=self.im_size, img_height=self.im_size)
                processed_frame = DataProcessing.processed
                if not np.array_equal(processed_frame, np.zeros((1, 1, 1))):
                    gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                    if preview:
                        cv2.imshow('preview', gray)

                    self.frames.append(gray)
            cv2.waitKey(1)

        vid.release()
        cv2.destroyAllWindows()
                

    def start_recording(self, preview=False):
        if not self.recording:
            self.frames = []
            self.recording = True
            self.r_thread = threading.Thread(target=self.recording_thread, args=(preview,))
            self.r_thread.start()

    def stop_recording(self):
        self.recording = False
        self.r_thread.join()


class Runner:
    def __init__(self, model_path='model.keras', output_folder='', frame_size=128, spec_size=128, fps=30, input_frame_count=60):
        self.frame_size = frame_size
        self.spec_size = spec_size
        self.fps = fps
        self.input_frame_count = input_frame_count
        self.model = load_model(model_path)
        self.output_folder = output_folder

    def run(self, frames):
        if len(frames) < self.input_frame_count:
            raise Exception(f"Not enough input frames! Expected {self.input_frame_count} frames, got {len(frames)}.")
        if frames.shape[1] != self.frame_size or frames.shape[2] != self.frame_size:
            raise Exception(f"Frames incorrect size! Expected {self.frame_size}x{self.frame_size}, got {frames.shape[1]}/{frames.shape[2]}.")
        if frames.shape[3] != 1:
            raise Exception(f"Frames must be 1 channel only, got {frames.shape[3]} channels.")
        temp_file = 'temp/'

        frames /= 255

        if os.path.isdir(temp_file):
            shutil.rmtree(temp_file)
        os.mkdir(temp_file)

        divs = int(len(frames) / self.input_frame_count)
        spectrograms = []

        for i in tqdm(range(divs)):
            index = i*self.input_frame_count
            x = frames[index:index+self.input_frame_count]
            x = np.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3]))

            pred = self.model.predict(x, verbose=0)[0] * 255
            #pred = np.reshape(pred, (pred.shape[0], pred.shape[1]))
            spectrograms.append(pred)

        #run arss to reverse the spectrograms
        for i, s in enumerate(spectrograms):
            path = f'{temp_file}{i}.png'
            s = s.astype('int32')
            cv2.imwrite(path, s)

            img = Image.open(path).convert('RGB')
            path = f'{temp_file}{i}.bmp'
            img.save(path)

            command = f"arss {temp_file}{i}.bmp {temp_file}temp{i}.wav -s -q -r 44100 -min 27 -max 19912 -p {int(self.spec_size/(self.input_frame_count/self.fps))}"
            os.system(command)

        combined = AudioSegment.empty()
        for i in range(len(spectrograms)):
            next_sound = AudioSegment.from_file(f'{temp_file}temp{i}.wav', format='wave')
            combined = combined + next_sound

        combined.export(self.output_folder+'output.mp3', format='mp3')

        shutil.rmtree(temp_file)

if __name__ == '__main__':
    test = Recorder()
    test.start_recording(preview=True)

    time.sleep(5)

    test.stop_recording()

    print(test.export_frames().shape)