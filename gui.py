# add the following 2 lines to solve OpenGL 2.0 bug
from kivy import Config

Config.set("graphics", "multisamples", "0")

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.lang.builder import Builder

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread

# import playsound
import argparse
import imutils
import time
import dlib
import cv2

# image variable
image = None

KV = """
BoxLayout:
    orientation: 'vertical'
    spacing: dp(20)
    padding: dp(20)

    GridLayout:
        cols: 3

        Label:
            text: '[b]Settings[b]'
            markup: True

        Label:
            text: '[b]Current Values[b]'
            markup: True

        Label:
            text: '[b]Set Values[b]'
            markup: True

        Label:
            text: 'EAR Threshold'

        Label:
            id: ear_label
            text: '{:.2f}'.format(app.ear)

        TextInput:
            id: ear_in
            multiline: False
            text: str(app.EYE_AR_THRESH)

        Label:
            text: 'Blink Time'

        Label:
            id: frames_label
            text: '{:.2f}'.format(app.blink_time_counter)

        TextInput:
            id: frames_in
            multiline: False
            text: str(app.microsleep)

        Label:
            text: 'FPS'

        Label:
            id: fps_label
            text: '{:.2f}'.format(app.fps)

        Widget:


        Button:
            text: 'Save'
            on_release: app.save(ear_in.text, frames_in.text)
"""


class KivyCamera(Image):
    def __init__(self, capture, fps, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        global image

        if image is not None:
            frame = image
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tobytes()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt="rgb"
            )
            image_texture.blit_buffer(buf, colorfmt="rgb", bufferfmt="ubyte")
            # display image from the texture
            self.texture = image_texture


class MyApp(App):
    """docstring for MyApp"""

    def __init__(self, **kwargs):
        super(MyApp, self).__init__()
        self.title = "Driver's Drowsiness Detector"

        self.stop_thread = False

        # define two constants, one for the eye aspect ratio to indicate
        # blink and then a second constant for the number of consecutive
        # frames the eye must be below the threshold for to set off the
        # alarm
        self.EYE_AR_THRESH = 0.27
        self.EYE_AR_CONSEC_FRAMES = 20
        self.ear = 0

        # initialize the frame counter as well as a boolean used to
        # indicate if the alarm is going off
        self.COUNTER = 0
        self.ALARM_ON = False

        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        print("[INFO] loading facial landmark predictor...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(r"predictor.dat")

        # grab the indexes of the facial landmarks for the left and
        # right eye, respectively
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        # start the video stream thread
        print("[INFO] starting video stream thread...")
        self.vs = VideoStream(0).start()
        time.sleep(1.0)

        # define the image to be shown on the kivy app
        self.capture = None
        self.ret = None

        self.fps = 0.0

        # define the threashold for duration of a blink
        # blink time average is 100–150 milliseconds
        # or 0.150 secends
        # Closures in excess of 1000 ms or 1 second were
        # defined as microsleeps.
        self.blink_time_counter: float = 0.0
        self.microsleep: float = 1.0
        self.blink_threshhold: float = 0.15

    def sound_alarm(self):
        # play an alarm sound
        # playsound.playsound(r'alarm.wav')
        pass

    def eye_aspect_ratio(self, eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])

        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        # return the eye aspect ratio
        return ear

    def say_hello(self, *arg):
        print("hello world!")
        pass

    def save(self, a, b):
        print("[INFO] Saving: EAR: ", a, " Frames: ", b)
        try:
            self.EYE_AR_THRESH = float(a)
            self.microsleep = float(b)

            # end the predict thread and start it again
            print("[INFO] Stopping predict thread")
            self.stop_thread = True
            self.stop_thread = False
            print("[INFO] Starting predict thread")
            detection_thread = Thread(target=self.detection)
            detection_thread.deamon = True
            detection_thread.start()

        except Exception as e:
            raise e

    def build(self):
        # build the kv
        root = Builder.load_string(KV)
        root.add_widget(KivyCamera(capture=self.capture, fps=25))

        # run the detection thread
        detection_thread = Thread(target=self.detection)
        detection_thread.deamon = True
        detection_thread.start()

        return root

    def on_stop(self):
        self.stop_thread = True

    def detection(self):
        # loop over frames from the video stream
        while True:

            # start time for fps
            start_time = time.time()

            # grab the frame from the threaded video file stream, resize
            # it, and convert it to grayscale
            # channels)
            frame = self.vs.read()
            frame = cv2.flip(frame, 1)
            frame = imutils.resize(frame, width=400)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale frame
            rects = self.detector(gray, 0)

            # loop over the face detections
            for rect in rects:
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # extract the left and right eye coordinates, then use the
                # coordinates to compute the eye aspect ratio for both eyes
                leftEye = shape[self.lStart : self.lEnd]
                rightEye = shape[self.rStart : self.rEnd]
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)

                # average the eye aspect ratio together for both eyes
                self.ear = (leftEAR + rightEAR) / 2.0

                self.root.ids.ear_label.text = "{:.2f}".format(self.ear)
                self.root.ids.frames_label.text = "{:.2f}".format(
                    self.blink_time_counter
                )

                # compute the convex hull for the left and right eye, then
                # visualize each of the eyes
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                # check to see if the eye aspect ratio is below the blink
                # threshold, and if so, increment the blink frame counter
                if self.ear < self.EYE_AR_THRESH:
                    self.COUNTER += 1

                    # start counting for time
                    self.blink_time_counter += time.time() - start_time

                    # if the eyes were closed for a sufficient number of
                    # then sound the alarm
                    if self.blink_time_counter >= self.microsleep:
                        # if the alarm is not on, turn it on
                        if not self.ALARM_ON:
                            self.ALARM_ON = True

                            # sound played in the background
                            sound_thread = Thread(target=self.sound_alarm)
                            sound_thread.deamon = True
                            sound_thread.start()

                        # draw an alarm on the frame
                        cv2.putText(
                            frame,
                            "DROWSINESS ALERT!",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )

                # otherwise, the eye aspect ratio is not below the blink
                # threshold, so reset the counter and alarm
                else:
                    self.COUNTER = 0
                    self.ALARM_ON = False
                    self.blink_time_counter = 0

                # draw the computed eye aspect ratio on the frame to help
                # with debugging and setting the correct eye aspect ratio
                # thresholds and frame counters
                cv2.putText(
                    frame,
                    "EAR: {:.2f}".format(self.ear),
                    (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            if self.stop_thread:
                break

            global image
            image = frame
            self.fps = 1.0 / (time.time() - start_time)
            self.root.ids.fps_label.text = "{:.2f}".format(self.fps)


MyApp().run()
