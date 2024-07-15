import cv2
import dlib
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

# Load dlib's pre-trained face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the hat image
hat_img = cv2.imread('hat.png', cv2.IMREAD_UNCHANGED)
if hat_img is None:
    raise FileNotFoundError("hat.png file not found or cannot be read.")

# Function to overlay the hat image on the frame


def overlay_hat(frame, hat_img, landmarks):
    forehead_center = (landmarks.part(27).x, landmarks.part(27).y - 30)
    face_width = landmarks.part(16).x - landmarks.part(0).x

    # Calculate hat size and position
    hat_width = int(1.5 * face_width)  # Keep hat width scaling
    hat_height = int(hat_width * hat_img.shape[0] / hat_img.shape[1])
    x1_hat = forehead_center[0] - hat_width // 2
    y1_hat = forehead_center[1] - hat_height
    x2_hat = x1_hat + hat_width
    y2_hat = forehead_center[1]

    # Ensure hat stays within frame boundaries
    y1_hat = max(0, y1_hat)
    y2_hat = min(frame.shape[0], y2_hat)
    x1_hat = max(0, x1_hat)
    x2_hat = min(frame.shape[1], x2_hat)

    # Resize hat image to fit
    hat_resized = cv2.resize(hat_img, (hat_width, hat_height))

    # Alpha blending to overlay hat
    alpha_hat = hat_resized[:, :, 3] / \
        255.0 if hat_resized.shape[2] == 4 else np.ones(hat_resized.shape[:2])
    alpha_frame = 1.0 - alpha_hat

    for c in range(0, 3):
        frame[y1_hat:y2_hat, x1_hat:x2_hat, c] = (
            alpha_hat * hat_resized[:, :, c] +
            alpha_frame * frame[y1_hat:y2_hat, x1_hat:x2_hat, c]
        )

    return frame

# Function to generate video frames


def generate_frames():
    cap = cv2.VideoCapture(0)  # Try using camera index 0
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    # Optionally set frame width and height to improve performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            frame = overlay_hat(frame, hat_img, landmarks)

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
        # Added host for external access if needed
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Error starting Flask app: {e}")
