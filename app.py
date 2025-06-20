from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import base64
import io
from PIL import Image
import threading
import time

app = Flask(__name__)

# Global variables
drawing_mode = False
prediction_mode = False
predicted_digit = None
confidence = 0.0
canvas = None
camera = None
hands_processor = None
model = None
clear_canvas_flag = False

# Initialize MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

class CameraHandler:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)  # Width
        self.cap.set(4, 720)   # Height
        self.canvas = np.zeros((720, 1280), dtype=np.uint8)
        self.prev_x, self.prev_y = 0, 0
        
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()

def load_mnist_model():
    """Load the MNIST model"""
    global model
    try:
        model = load_model('lenet5_mnist_model.h5')
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure 'lenet5_mnist_model.h5' exists in the same directory.")
        return False

def preprocess_for_mnist(canvas_region):
    """Preprocess the canvas region for MNIST prediction"""
    # Resize to 28x28
    resized = cv2.resize(canvas_region, (28, 28))
    
    # Normalize to 0-1 range
    normalized = resized.astype('float32') / 255.0
    
    # Reshape for model input (1, 28, 28, 1)
    reshaped = normalized.reshape(1, 28, 28, 1)
    
    return reshaped

def predict_digit(canvas_region):
    """Predict digit from canvas region"""
    global model
    if model is None:
        return None, 0.0
        
    try:
        processed_img = preprocess_for_mnist(canvas_region)
        prediction = model.predict(processed_img, verbose=0)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)
        return int(digit), float(confidence)
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, 0.0

def generate_frames():
    """Generate video frames for streaming"""
    global drawing_mode, prediction_mode, predicted_digit, confidence, clear_canvas_flag
    
    camera = CameraHandler()
    x1, y1 = 150, 100
    x2, y2 = 450, 400
    
    while True:
        # Check if canvas should be cleared
        if clear_canvas_flag:
            camera.canvas = np.zeros((720, 1280), dtype=np.uint8)
            predicted_digit = None
            confidence = 0.0
            clear_canvas_flag = False
        ret, frame = camera.cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # Draw drawing area (purple square)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        
        # Display mode status
        drawing_status = "Drawing: ON" if drawing_mode else "Drawing: OFF"
        prediction_status = "Prediction: ON" if prediction_mode else "Prediction: OFF"
        
        cv2.putText(frame, drawing_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if drawing_mode else (0, 0, 255), 2)
        cv2.putText(frame, prediction_status, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if prediction_mode else (0, 0, 255), 2)
        
        # Display prediction result
        if predicted_digit is not None and prediction_mode:
            pred_text = f"Digit: {predicted_digit} ({confidence:.2f})"
            cv2.putText(frame, pred_text, (50, 510), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 0), 3)
        
        # Display controls
        cv2.putText(frame, "Web Controls Available", (900, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Use buttons below", (900, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
                
                # Get index finger tip
                lm = handLms.landmark[8]
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                # Draw finger position
                cv2.circle(frame, (cx, cy), 10, (255, 255, 0), -1)
                
                # Only draw if inside the drawing area and drawing mode is active
                if x1 < cx < x2 and y1 < cy < y2 and drawing_mode:
                    if camera.prev_x == 0 and camera.prev_y == 0:
                        camera.prev_x, camera.prev_y = cx, cy
                    # Draw on grayscale canvas with white color (255)
                    cv2.line(camera.canvas, (camera.prev_x, camera.prev_y), (cx, cy), 255, 15)
                    camera.prev_x, camera.prev_y = cx, cy
                else:
                    camera.prev_x, camera.prev_y = 0, 0  # Reset when outside or drawing is off
        
        # Convert grayscale canvas to BGR for blending
        canvas_bgr = cv2.cvtColor(camera.canvas, cv2.COLOR_GRAY2BGR)
        
        # Blend canvas with live feed
        frame = cv2.addWeighted(frame, 0.7, canvas_bgr, 0.3, 0)
        
        # Perform prediction if prediction mode is on
        if prediction_mode and model is not None:
            # Extract the drawing area from canvas
            canvas_region = camera.canvas[y1:y2, x1:x2]
            
            # Only predict if there's something drawn (not all zeros)
            if np.sum(canvas_region) > 0:
                predicted_digit, confidence = predict_digit(canvas_region)
            else:
                predicted_digit, confidence = None, 0.0
            
            # Show the processed region for debugging (optional)
            if canvas_region.size > 0:
                # Resize for display
                display_region = cv2.resize(canvas_region, (150, 150))
                display_region_bgr = cv2.cvtColor(display_region, cv2.COLOR_GRAY2BGR)
                frame[500:650, 900:1050] = display_region_bgr
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_drawing', methods=['POST'])
def toggle_drawing():
    """Toggle drawing mode"""
    global drawing_mode
    drawing_mode = not drawing_mode
    return jsonify({'drawing_mode': drawing_mode})

@app.route('/toggle_prediction', methods=['POST'])
def toggle_prediction():
    """Toggle prediction mode"""
    global prediction_mode, predicted_digit, confidence
    prediction_mode = not prediction_mode
    if not prediction_mode:
        predicted_digit = None
        confidence = 0.0
    return jsonify({'prediction_mode': prediction_mode})

@app.route('/clear_canvas', methods=['POST'])
def clear_canvas():
    """Clear the canvas"""
    global predicted_digit, confidence, clear_canvas_flag
    clear_canvas_flag = True
    predicted_digit = None
    confidence = 0.0
    return jsonify({'status': 'canvas_cleared'})

@app.route('/get_status')
def get_status():
    """Get current status"""
    return jsonify({
        'drawing_mode': drawing_mode,
        'prediction_mode': prediction_mode,
        'predicted_digit': predicted_digit,
        'confidence': confidence,
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    # Load the model at startup
    model_loaded = load_mnist_model()
    if not model_loaded:
        print("Warning: Running without MNIST model. Please ensure 'lenet5_mnist_model.h5' is available.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)