import cv2
import serial
import time
import numpy as np
print("=== PROJECT SENTINEL: CUSTOM AI COMMAND NODE ===")
#ESP32 CONNECTION
ESP32_PORT = 'Your COM'
BAUD_RATE = 9600
try:
    esp32 = serial.Serial(ESP32_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to ESP32 on {ESP32_PORT}")
    time.sleep(2) 
except Exception as e:
    print(f"Could not connect to ESP32 on {ESP32_PORT}.")
    esp32 = None
#Load Bouzine's AI Model
print("Loading Custom Sentinel Neural Network")
#Bzaaaaaaf d code
print("AI Model Loaded Successfully.")
#Connecting phone camera
PHONE_IP_URL = "http://192.168.1.45:8080/video"
print("Connecting to Phone Camera stream")
video_capture = cv2.VideoCapture(PHONE_IP_URL)
if not video_capture.isOpened():
    print("Could not connect to phone camera!")
print("System Ready")
# Detection Loop
while True:
    ret, frame = video_capture.read()
    if not ret:
        continue
    # ---------------------------------------------------------
    # 🛠️ TEAMMATE PRE-PROCESSING ZONE
    # Neural networks usually need specific image sizes (like 224x224).
    # Teammate: Resize and normalize the frame to match your training data!
    # ---------------------------------------------------------
    # Example:
    # resized_frame = cv2.resize(frame, (224, 224))
    # normalized_frame = resized_frame / 255.0
    # input_tensor = np.expand_dims(normalized_frame, axis=0) 
    
    # ---------------------------------------------------------
    # 🔮 TEAMMATE PREDICTION ZONE
    # Run the input through your custom model to get the result.
    # ---------------------------------------------------------
    # Example:
    # prediction = my_custom_model.predict(input_tensor)
    # confidence = prediction[0][0] # Assuming output is between 0 (Fake/Unknown) and 1 (Real/VIP)
    
    # ⚠️ For the sake of this template running without crashing, we will mock the prediction logic:
    # Replace the line below with your actual model's IF statement!
    is_authorized = False # Change this based on your model's prediction output
    
    # Telling ESP32 that all is good
    if is_authorized:
        print("Match Found, Sending OPEN signal to ESP32...")
        # Draw a green box/text for the UI
        cv2.putText(frame, "ACCESS GRANTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if esp32 and esp32.is_open:
            esp32.write(b'1') # Open the gates!  
        # Cooldown so the ESP32 can check the weight sensors and actuate the servos safely
        time.sleep(4)   
    else:
        # If it's a stranger, or background noise
        cv2.putText(frame, "SCANNING / UNKNOWN", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if esp32 and esp32.is_open:
            esp32.write(b'0') # Ensure gates remain locked
    # Show the live feed on the laptop screen
    cv2.imshow('Project Sentinel - Custom AI Node', frame)
    # Press 'q' to shut down safely
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Shutting down Sentinel System...")
        break
# Clean up
video_capture.release()
cv2.destroyAllWindows()
if esp32 and esp32.is_open:
    esp32.close()