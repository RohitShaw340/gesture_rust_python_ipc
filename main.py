import cv2
import mediapipe as mp
import numpy as np
import struct
import socket
import json
from PIL import Image
import io
import tensorflow as tf

# tf.get_logger().setLevel("ERROR")

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="lite_gesture_model.tflite")
label_map = np.load("lable_map.npy", allow_pickle=True).item()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
mp_pose = mp.solutions.pose
UPPER_BODY_PARTS = [0, 7, 8, 11, 12, 13, 14, 15, 16]


def preprocess_image(image_bytes):
    # Convert image bytes to OpenCV image
    img = np.array(Image.open(io.BytesIO(image_bytes)).convert(mode="RGB"))
    # img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW

    pose = mp_pose.Pose()

    results = pose.process(img)
    pose.close()
    del pose

    # Draw pose landmarks on the frame for upper body parts only
    if results.pose_landmarks:
        temp = []
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if idx in UPPER_BODY_PARTS:
                h, w, c = img.shape
                temp.append([landmark.x, landmark.y, landmark.z])
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)

        data = np.array(temp)
        center_x = data[:, 0].mean()
        center_y = data[:, 1].mean()
        center_z = data[:, 2].mean()

        data[:, 0] = (data[:, 0] - center_x) * 500  # X coordinates
        data[:, 1] = (data[:, 1] - center_y) * 500  # Y coordinates
        data[:, 2] = (data[:, 2] - center_z) * 500  # Z coordinates

        return data
    return None


def predict_gesture(data):
    # Allocate tensors
    interpreter.allocate_tensors()

    # Process the image and predict gesture
    interpreter.set_tensor(
        input_details[0]["index"], data.reshape(1, 9, 3).astype(np.float32)
    )

    # run the inference
    interpreter.invoke()

    # output_details[0]['index'] = the index which provides the input
    prediction = interpreter.get_tensor(output_details[0]["index"])
    output = np.argmax(prediction)
    label = label_map[output]

    if prediction[0][output] < 0.9:
        display_text = "None"
        label = "None"

    # Convert prediction to JSON
    if label == "One_Hand_Up":
        label = "Toggle"
    else:
        label = "None"

    prediction_json = json.dumps({"prediction": [{"gesture": label}]})

    return prediction_json


config = {
    "process_id": "gesture",
    "server_address": "/tmp/gesurease.sock",
}


def run():
    data_len_bytes = sock.recv(4)
    if len(data_len_bytes) == 0:
        print("Connection closed, exiting...")
        exit(1)

    data_len = struct.unpack("!I", data_len_bytes)[0]

    img = sock.recv(data_len)
    while len(img) < data_len:
        img += sock.recv(data_len - len(img))

    # print(img)

    data = preprocess_image(img)
    gesture_prediction = (
        predict_gesture(data)
        if data is not None
        else json.dumps({"prediction": [{"gesture": "None"}]})
    )

    sock.sendall(struct.pack("!I", len(gesture_prediction)))
    sock.sendall(gesture_prediction.encode())


if __name__ == "__main__":
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(config["server_address"])
    sock.setblocking(True)

    # Send the process identifier to the Rust server
    sock.sendall(config["process_id"].encode())

    while True:
        run()
