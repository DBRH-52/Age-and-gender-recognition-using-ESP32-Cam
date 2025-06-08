# Age and gender recognition - ESP32-Cam

- camera system that combines an ESP32 camera module with a Python-based server for real-time age and gender detection
- uses TensorFlow models to analyze images captured by the camera

## Project Components

### Hardware
- ESP32 AI Thinker Camera Module
- PSRAM (required for high-resolution images)
- WiFi 

### Software
- Arduino-based camera server
- Python Flask server for AI processing
- TensorFlow models for age and gender detection

## Prerequisites

### Software Requirements
- Arduino IDE with ESP32 board support
- Python 3.x
- Required Python packages:
  - TensorFlow
  - Flask
  - NumPy
  - OpenCV (for image processing)

## Project Structure

- Flask server for AI processing
- image preprocessing utils
- Model testing utils
- Camera stream capture utils
- ESP32 camera pin definitions
- Web interface HTML/CSS/JS
- ESP32 HTTP server implementation

## API Endpoints

The server exposes the following endpoint:

- `POST /predict`
  - Accepts JSON with image data
  - Returns JSON with gender and age predictions

