import time
import socket
import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

import tensorflow as tf


# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
DETECTION_RESULT = None
PERSON = 0
# Visualization parameters
fps_avg_frame_count = 10
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="Posenet\lite_gesture_model.tflite")
label_map = np.load("Posenet/Dataset/lable_map.npy", allow_pickle=True).item()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.allocate_tensors()


def save_result(
    result: vision.PoseLandmarkerResult,
    unused_output_image: mp.Image,
    timestamp_ms: int,
):
    global FPS, COUNTER, START_TIME, DETECTION_RESULT

    # Calculate the FPS
    if COUNTER % fps_avg_frame_count == 0:
        FPS = fps_avg_frame_count / (time.time() - START_TIME)
        START_TIME = time.time()

    DETECTION_RESULT = result
    COUNTER += 1


# Initialize the pose landmarker model
base_options = python.BaseOptions(model_asset_path="Posenet\pose_landmarker_lite.task")
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


def pre_process(data):
    data = np.array(data)
    center_x = data[:, 0].mean()
    center_y = data[:, 1].mean()
    center_z = data[:, 2].mean()

    data[:, 0] = (data[:, 0] - center_x) * 500  # X coordinates
    data[:, 1] = (data[:, 1] - center_y) * 500  # Y coordinates
    data[:, 2] = (data[:, 2] - center_z) * 500  # Z coordinates

    return data


def predict(data, frame):
    global PERSON
    interpreter.set_tensor(
        input_details[0]["index"], data.reshape(1, 9, 3).astype(np.float32)
    )

    # run the inference
    interpreter.invoke()

    # output_details[0]['index'] = the index which provides the input
    prediction = interpreter.get_tensor(output_details[0]["index"])
    output = np.argmax(prediction)
    label = label_map[output]
    display_text = label + " : " + str(round(prediction[0][output] * 100, 2)) + " %"

    if prediction[0][output] < 0.9:
        display_text = "None"
        label = "None"

    label_coords = (50, 50 + PERSON * 50)
    print("Predicted Label: ", label_map[output])
    cv2.putText(
        frame,
        display_text,
        label_coords,
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("Upper Body Pose Detection", frame)

    # Display the frame

    del prediction
    del output
    del label
    del display_text
    del data
    del frame

    return True


def capture(cap):
    global DETECTION_RESULT, PERSON

    # Read frame from webcam
    ret, frame = cap.read()

    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    detector.detect_async(mp_image, time.time_ns() // 1_000_000)
    # detector.detect_

    if not ret:
        return False

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(frame_rgb.shape)

    # Process the frame with MediaPipe PoseNet
    # results = DETECTION_RESULT
    UPPER_BODY_PARTS = [0, 7, 8, 11, 12, 13, 14, 15, 16]
    if DETECTION_RESULT is not None:
        # print(np.array(DETECTION_RESULT.pose_landmarks).shape)

        keypoints = []
        for pose_landmarks in DETECTION_RESULT.pose_landmarks:
            temp = []
            for idx, landmark in enumerate(pose_landmarks):
                if idx in UPPER_BODY_PARTS:
                    h, w, c = frame.shape
                    temp.append([landmark.x, landmark.y, landmark.z])
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            data = pre_process(temp)
            keypoints.append(data)
        keypoints = np.array(keypoints)

        print(keypoints.shape)
        for d in keypoints:
            predict(d, frame)
            PERSON += 1
        DETECTION_RESULT = None
        PERSON = 0
        time.sleep(0.1)
        # predictions = pose_model.predict(keypoints)
        # print(predictions)
        # gesture_indexes = [np.argmax(prediction) for prediction in predictions]
        # gesture_probabilities = [np.max(prediction) for prediction in predictions]

        # gesture_names = [gestures[index] for index in gesture_indexes]
        # for i, gesture_name in enumerate(gesture_names):
        #     cv2.putText(
        #         image,
        #         f"Person {i+1}: {gesture_name} | Probability: {gesture_probabilities[i]*100:.2f}",
        #         (10, 30 * (i + 1)),
        #         font,
        #         1,
        #         (255, 0, 0),
        #         2,
        #         cv2.LINE_AA,
        #     )

    # if DETECTION_RESULT is not None:

    #     # Draw pose landmarks on the frame for upper body parts only
    #     for pose_landmarks in DETECTION_RESULT.pose_landmarks:
    #         # Draw the pose landmarks.
    #         pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    #         pose_landmarks_proto.landmark.extend(
    #             [
    #                 landmark_pb2.NormalizedLandmark(
    #                     x=landmark.x, y=landmark.y, z=landmark.z
    #                 )
    #                 for landmark in pose_landmarks
    #             ]
    #         )
    #         # print("Proto length : ", pose_landmarks_proto.landmark)

    #         temp = []
    #         for idx, landmark in enumerate(pose_landmarks_proto.landmark):
    #             if idx in UPPER_BODY_PARTS:
    #                 # h, w, c = frame.shape
    #                 h, w, c = frame.shape
    #                 cx, cy = int(landmark.x * w), int(landmark.y * h)
    #                 cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
    #                 temp.append([landmark.x, landmark.y, landmark.z])

    #         data = pre_process(temp)
    # print(data)
    # print(type(data))
    # predict(data, frame)

    # Clean up resources
    # del temp
    # del data
    # del h
    # del w
    # del c
    # del cx
    # del cy

    # cv2.imshow("Upper Body Pose Detection", frame)

    # del results
    del frame_rgb
    del frame
    del ret

    return True


# Initialize VideoCapture object to read from webcam
cap = cv2.VideoCapture(0)

# Body part indices for upper body
UPPER_BODY_PARTS = [0, 7, 8, 11, 12, 13, 14, 15, 16]

while True:

    capture(cap)

    # Release resources
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
# pose.close()
cap.release()
cv2.destroyAllWindows()


# socket_path = "/tmp/gesturease.sock"
# server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
# server.bind(socket_path)
# server.listen(1)

# print("Waiting for connection...")

# while True:
#     conn, _ = server.accept()
#     print("Connected.")

#     # Receive image data
#     image_bytes = conn.recv(1024)
#     if not image_bytes:
#         break

#     # Preprocess image
#     image, nose = preprocess_image(image_bytes)

#     detector.detect_async(image, time.time_ns() // 1_000_000)
#     # Predict gesture
#     gesture_prediction = predict_gesture(image, nose)

#     # Send gesture prediction back to Rust
#     conn.sendall(gesture_prediction.encode())

#     conn.close()
