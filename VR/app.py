import cv2
import dlib
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

# Load dlib's pre-trained face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the shirt image
shirt_img = cv2.imread('shirt.png', cv2.IMREAD_UNCHANGED)
if shirt_img is None:
    raise FileNotFoundError("shirt.png file not found or cannot be read.")

# Function to overlay the shirt image on the frame

# Function to overlay the shirt image on the frame


def overlay_shirt(frame, shirt_img, landmarks):
    # Get key facial landmarks
    left_ear = (landmarks.part(0).x, landmarks.part(0).y)
    right_ear = (landmarks.part(16).x, landmarks.part(16).y)
    chin = (landmarks.part(8).x, landmarks.part(8).y)

    # Calculate face dimensions
    face_width = right_ear[0] - left_ear[0]
    face_height = chin[1] - (left_ear[1] + right_ear[1]) // 2

    # Estimate body proportions
    body_width = int(face_width * 3.5)  # Adjust this multiplier if needed
    # Adjust this multiplier for longer/shorter shirts
    body_height = int(face_height * 5)

    # Calculate shirt position
    x1_shirt = left_ear[0] - (body_width - face_width) // 2
    y1_shirt = chin[1]
    x2_shirt = x1_shirt + body_width
    y2_shirt = y1_shirt + body_height

    # Ensure the coordinates are within frame boundaries
    y1_shirt = max(0, y1_shirt)
    y2_shirt = min(frame.shape[0], y2_shirt)
    x1_shirt = max(0, x1_shirt)
    x2_shirt = min(frame.shape[1], x2_shirt)

    # Resize the shirt image to fit the calculated size
    shirt_resized = cv2.resize(
        shirt_img, (x2_shirt - x1_shirt, y2_shirt - y1_shirt))

    # Overlay the shirt image on the frame
    alpha_shirt = shirt_resized[:, :, 3] / 255.0
    alpha_frame = 1.0 - alpha_shirt

    for c in range(0, 3):
        frame[y1_shirt:y2_shirt, x1_shirt:x2_shirt, c] = (
            alpha_shirt * shirt_resized[:, :, c] +
            alpha_frame * frame[y1_shirt:y2_shirt, x1_shirt:x2_shirt, c]
        )

    return frame


# Function to generate video frames


def generate_frames():
    cap = cv2.VideoCapture(0)  # Try index 0 first
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)  # Try index 1 if index 0 doesn't work
        if not cap.isOpened():
            print("Error: Could not open video capture.")
            return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            frame = overlay_shirt(frame, shirt_img, landmarks)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Failed to encode frame.")
            continue

        # Convert frame to bytes and yield for response
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Route for index page


@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering template: {e}")
        return "Error rendering template."

# Route for video feed


@app.route('/video_feed')
def video_feed():
    try:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error in video feed: {e}")
        return "Error in video feed."


# Main function to run Flask app
if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        print(f"Error starting Flask app: {e}")
