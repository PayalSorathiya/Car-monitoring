BMW AI Interior Camera Person Detection
System
Complete Setup and Usage Guide
System Overview
This BMW AI system performs real-time person detection on interior camera footage using:
 YOLOv5 Nano for on-device person detection
 Google Gemini 1.5 Flash for intelligent analysis
 Python Flask Backend for data logging and management
 ExoPlayer for smooth video playback
Prerequisites
Hardware Requirements
 Android Device: Android 7.0+ (API 24+)
 Computer: For running Python backend server
 Network: Both devices on same WiFi network
Software Requirements
 Android Studio (for app development/installation)
 Python 3.8+ with pip
 Video Files: MP4
🔧 Step 1: Backend Server Setup
1.1 Install Python Dependencies
# see the requirements.txt for configuration
1.3 Start Backend Server
# Navigate to backend directory
backend_final.py
📱 Step 2: Android App Configuration
2.1 Get Your Computer's IP Address
Find your computer's local IP address:
Windows:
ipconfig
Look for "IPv4 Address" under your WiFi adapter.
Mac/Linux:
ifconfig | grep "inet " | grep -v 127.0.0.1
Example IP: 192.168.1.100
2.2 Configure Android App
In MainActivity.kt, update these constants:
// Line 67-68: Update with YOUR computer's IP address
private val BACKEND_BASE_URL = "http://YOUR_COMPUTER_IP:5000" // e.g.,
"http://192.168.1.100:5000"
private val BACKEND_DATA_ENDPOINT = "$BACKEND_BASE_URL/api/data"
// Line 65: Add your Gemini API key
private val GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
In network_security_config.kt, update yout Ip address
// Line 67-68: Update with YOUR computer's IP address
<domain includeSubdomains="true">172.20.10.8</domain>
2.3 Get Gemini API Key (Optional but Recommended)
1. Visit Google AI Studio
2. Create new API key
3. Copy and paste into GEMINI_API_KEY constant
Note: App works without Gemini (uses local analysis), but AI insights are enhanced with
Gemini.
2.4 Add Required Model
Place yolov5n.tflite model file in app/src/main/assets/ directory.
🚀 Step 3: Installation and Usage
3.1 Network Setup
1. Ensure both devices are on same WiFi network
2. Start backend server first: backend_final.py
3. Note the server IP address from terminal output
4. Update Android app with correct IP address
3.2 Install Android App
# Build and install via Android Studio
# OR use command line:
./gradlew installDebug
3.3 Using the Application
Step-by-Step Usage:
1. Start Backend: Run python backend_final.py on your computer
2. Open Android App: Launch BMW AI app on your device
3. Select Video: Tap "Select Video" → choose MP4/WebM file
4. Start Analysis: Tap "Start" to begin AI processing
5. View Results: Real-time detection + AI summary generation
6. Backend Logging: Analysis automatically sent to backend server
Recommended Video Types:
 Interior camera footage (dashboard/rearview mirror perspective)
 Carpool Karaoke style videos (multiple people in car)
 Any MP4 with people visible
🔍 Step 4: Monitoring and Troubleshooting
4.1 Backend Monitoring
Monitor your backend terminal for real-time data:
📊 BMW AI Analysis Received:
 Session: bmw_session_1672847293847
 Video Duration: 45s
 Detection Events: 127
 Saved to: bmw_data/analysis_20231204_143456.json
4.2 Common Issues and Solutions
Backend Connection Failed
Symptoms: "Cannot reach backend server" in Android app Solutions:
1. Verify backend server is running (python backend_final.py)
2. Check IP address in Android app matches computer IP
3. Ensure both devices on same WiFi network
4. Try disabling computer firewall temporarily
5. Test with browser: http://YOUR_IP:5000/api/health
Gemini API Issues
Symptoms: "AI Unavailable - Using Local Analysis" Solutions:
1. Verify API key is correct in GEMINI_API_KEY
2. Check internet connection on Android device
3. App continues working with local analysis if Gemini fails
Video Loading Issues
Symptoms: "Error loading video" Solutions:
1. Ensure video is MP4 format
2. Try shorter video files (< 2 minutes for testing)
3. Grant storage permissions when prompted
4.3 Testing Backend Connection
Test backend manually in browser or curl:
# Health check
curl http://YOUR_IP:5000/api/health
# View stored data
curl http://YOUR_IP:5000/api/data
Step 5: Data Analysis
5.1 Viewing Analysis Results
Analysis data is saved in data/ directory as JSON files:
{
 "timestamp": 1672847293847,
 "bmw_ai_summary": "Executive Summary: Comprehensive person detection
analysis...",
 "video_analysis": {
 "video_duration_seconds": 45,
 "total_detection_events": 127,
 "final_people_detected": 2
 },
 "device": {
 "model": "Samsung Galaxy S21",
 "android_version": "13"
 }
}
5.2 API Endpoints
 GET /api/health - Check server status
 POST /api/data - Upload analysis data (used by Android app)
 GET /api/data - Retrieve all analysis summaries
Expected Results
Successful Setup Indicators:
 ✅ Backend shows "BMW AI Backend Server Starting..." message
 ✅ Android app displays "AI Model loaded: YOLOv5 + Gemini AI Ready"
 ✅ Video loads successfully with duration/resolution display
 ✅ Real-time person detection with bounding boxes
 ✅ AI summary generated after video completion
 ✅ Backend receives and logs analysis data
Performance Expectations:
 Detection: 1-3 people detected in typical interior footage
 Processing: Real-time at ~10 FPS on modern Android devices
 AI Summary: Generated in 5-15 seconds with Gemini
 Backend Logging: Immediate data transmission after analysis
BMW AI Interior Camera Person Detection System - Ready for automotive safety research
and development.
