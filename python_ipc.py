import cv2
import mediapipe as mp
import numpy as np
import socket
import json
from PIL import Image
import io
import tensorflow as tf

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="lite_gesture_model.tflite")
label_map = np.load("lable_map.npy", allow_pickle=True).item()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
mp_pose = mp.solutions.pose
UPPER_BODY_PARTS = [0, 7, 8, 11, 12, 13, 14, 15, 16]


def preprocess_image(image_bytes):
    # Convert image bytes to OpenCV image
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    pose = mp_pose.Pose()

    results = pose.process(frame)
    pose.close()
    del pose

    # Draw pose landmarks on the frame for upper body parts only
    if results.pose_landmarks:
        temp = []
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if idx in UPPER_BODY_PARTS:
                h, w, c = frame.shape
                temp.append([landmark.x, landmark.y, landmark.z])
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        data = np.array(temp)
        center_x = data[:, 0].mean()
        center_y = data[:, 1].mean()
        center_z = data[:, 2].mean()

        data[:, 0] = (data[:, 0] - center_x) * 500  # X coordinates
        data[:, 1] = (data[:, 1] - center_y) * 500  # Y coordinates
        data[:, 2] = (data[:, 2] - center_z) * 500  # Z coordinates

        print(data)
        print(data.shape)
        nose_coordianters = [temp[0][0] * w, temp[0][1] * h]
        return data, nose_coordianters
    return None


def predict_gesture(data, nose):
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

    prediction_json = json.dumps({"gesture": label, "x": nose[0], "y": nose[1]})

    return prediction_json


# Create a Unix domain socket server
socket_path = "/tmp/gesturease.sock"
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(socket_path)
server.listen(1)

print("Waiting for connection...")

while True:
    conn, _ = server.accept()
    print("Connected.")

    # Receive image data
    image_bytes = conn.recv(1024)
    if not image_bytes:
        break

    # Preprocess image
    image, nose = preprocess_image(image_bytes)

    # Predict gesture
    gesture_prediction = predict_gesture(image, nose)

    # Send gesture prediction back to Rust
    conn.sendall(gesture_prediction.encode())

    conn.close()
