from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.core.window import Window
import cv2
import mediapipe as mp
import numpy as np

class MultiDetectionApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        self.img = Image()
        self.layout.add_widget(self.img)

        self.btn_layout = BoxLayout(size_hint_y=0.2)
        self.btn_face = Button(text='Detect Face')
        self.btn_hand = Button(text='Detect Hands')
        self.btn_body = Button(text='Detect Body')

        self.btn_face.bind(on_press=self.toggle_face_detection)
        self.btn_hand.bind(on_press=self.toggle_hand_detection)
        self.btn_body.bind(on_press=self.toggle_body_detection)

        self.btn_layout.add_widget(self.btn_face)
        self.btn_layout.add_widget(self.btn_hand)
        self.btn_layout.add_widget(self.btn_body)
        self.layout.add_widget(self.btn_layout)

        # Initialize video capture
        self.camera_index = 1  # Default to back camera
        self.capture = cv2.VideoCapture(self.camera_index)
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=4)
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=2)
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()

        self.detect_face = False
        self.detect_hand = False
        self.detect_body = False

        self.last_touch_time = 0
        self.double_tap_threshold = 0.5  # seconds

        Window.bind(on_touch_down=self.on_touch_down)
        Clock.schedule_interval(self.update, 1.0/30.0)
        return self.layout

    def toggle_face_detection(self, instance):
        self.detect_face = not self.detect_face

    def toggle_hand_detection(self, instance):
        self.detect_hand = not self.detect_hand

    def toggle_body_detection(self, instance):
        self.detect_body = not self.detect_body

    def on_touch_down(self, instance, touch):
        current_time = touch.time_update
        if current_time - self.last_touch_time < self.double_tap_threshold:
            self.switch_camera()
        self.last_touch_time = current_time

    def switch_camera(self):
        # Release the current capture
        self.capture.release()

        # Toggle camera index between 0 and 1 (back and front)
        self.camera_index = 0 if self.camera_index == 1 else 1
        self.capture = cv2.VideoCapture(self.camera_index)

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            imRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.detect_face:
                face_results = self.faceMesh.process(imRGB)
                if face_results.multi_face_landmarks:
                    for faceLms in face_results.multi_face_landmarks:
                        self.draw_face_box_and_mood(faceLms, frame)

            if self.detect_hand:
                hand_results = self.hands.process(imRGB)
                if hand_results.multi_hand_landmarks:
                    for handLms in hand_results.multi_hand_landmarks:
                        self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)

            if self.detect_body:
                pose_results = self.pose.process(imRGB)
                if pose_results.pose_landmarks:
                    self.mpDraw.draw_landmarks(frame, pose_results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img.texture = texture

    def draw_face_box_and_mood(self, faceLms, frame):
        h, w, _ = frame.shape
        coords = [(int(lm.x * w), int(lm.y * h)) for lm in faceLms.landmark]

        x_min = min([coord[0] for coord in coords])
        y_min = min([coord[1] for coord in coords])
        x_max = max([coord[0] for coord in coords])
        y_max = max([coord[1] for coord in coords])

        left_eye = np.mean([coords[33], coords[133], coords[160], coords[158], coords[153], coords[144]], axis=0)
        right_eye = np.mean([coords[263], coords[362], coords[387], coords[385], coords[380], coords[373]], axis=0)
        mouth_top = coords[13]
        mouth_bottom = coords[14]
        left_eyebrow = coords[105]
        right_eyebrow = coords[334]

        eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
        mouth_distance = np.linalg.norm(np.array(mouth_top) - np.array(mouth_bottom))
        eyebrow_distance = np.linalg.norm(np.array(left_eyebrow) - np.array(right_eyebrow))

        ratio = mouth_distance / eye_distance
        eyebrow_ratio = eyebrow_distance / eye_distance

        if ratio > 0.05:
            mood = 'Happy'
            color = (0, 255, 0)
        elif ratio < 0.02 and eyebrow_ratio < 0.2:
            mood = 'Sad'
            color = (255, 0, 0)
        elif eyebrow_ratio < 0.1:
            mood = 'Angry'
            color = (0, 0, 255)
        else:
            mood = 'Neutral'
            color = (0, 255, 255)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(frame, f'Mood: {mood}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    def on_stop(self):
        self.capture.release()

if __name__ == '__main__':
    MultiDetectionApp().run()
