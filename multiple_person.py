import time
import socket
import mediapipe as mp
import numpy as np
import io
from PIL import Image
import struct
import json

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

import tensorflow as tf


# tf.get_logger().setLevel("ERROR")

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="lite_gesture_model.tflite")
label_map = np.load("lable_map.npy", allow_pickle=True).item()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

UPPER_BODY_PARTS = [0, 7, 8, 11, 12, 13, 14, 15, 16]
interpreter.allocate_tensors()


def save_result(
    result: vision.PoseLandmarkerResult,
    unused_output_image: mp.Image,
    timestamp_ms: int,
):
    global DETECTION_RESULT
    DETECTION_RESULT = result


# Initialize the pose landmarker model
base_options = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_poses=5,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_segmentation_masks=False,
    result_callback=save_result,
)
detector = vision.PoseLandmarker.create_from_options(options)


def preprocess_image(image_bytes):
    # Convert image bytes to OpenCV image
    img = np.array(Image.open(io.BytesIO(image_bytes)).convert(mode="RGB"))
    # img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    detector.detect_async(mp_image, time.time_ns() // 1_000_000)
    nose_coords = []
    # Draw pose landmarks on the frame for upper body parts only
    if DETECTION_RESULT is not None:

        keypoints = []
        for pose_landmarks in DETECTION_RESULT.pose_landmarks:
            temp = []
            for idx, landmark in enumerate(pose_landmarks):
                if idx in UPPER_BODY_PARTS:
                    temp.append([landmark.x, landmark.y, landmark.z])
                    # cx, cy = int(landmark.x * w), int(landmark.y * h)
                    # cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            nose_coords.append(temp[0])
            data = np.array(temp)
            center_x = data[:, 0].mean()
            center_y = data[:, 1].mean()
            center_z = data[:, 2].mean()

            data[:, 0] = (data[:, 0] - center_x) * 500  # X coordinates
            data[:, 1] = (data[:, 1] - center_y) * 500  # Y coordinates
            data[:, 2] = (data[:, 2] - center_z) * 500  # Z coordinates
            keypoints.append(data)
        keypoints = np.array(keypoints)
        return keypoints, nose_coords

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

    prediction = interpreter.get_tensor(output_details[0]["index"])
    output = np.argmax(prediction)
    label = label_map[output]

    if prediction[0][output] < 0.9:
        label = "None"

    if label == "One_Hand_Up":
        label = "Toggle"
    else:
        label = "None"

    return label


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
    key_points_multiple_person, nose_coords = preprocess_image(img)

    if key_points_multiple_person is not None:
        gesture_prediction = []
        for idx, key_points in enumerate(key_points_multiple_person):
            gesture_prediction.append([predict_gesture(key_points), nose_coords[idx]])
        json_data = []
        for i in gesture_prediction:
            dict = {"gesture": i[0], "x": i[1][0], "y": i[1][1]}
            json_data.append(dict)

        json_response = {"prediction": json_data}
        sock.sendall(struct.pack("!I", len(json_response)))
        sock.sendall(json_response.encode())
    else:
        gesture_prediction = json.dumps(
            {"prediction": [{"gesture": "None", "x": "None", "y": "None"}]}
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
