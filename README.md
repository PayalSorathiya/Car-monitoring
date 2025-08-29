# BMW AI Interior Camera Person Detection System

A real-time AI-powered person detection system designed for automotive interior monitoring using YOLOv5 and Google Gemini AI integration. This system combines computer vision with intelligent analysis for enhanced automotive safety research and development.

## Overview

This system provides comprehensive person detection and analysis for interior camera footage using:
- **YOLOv5 Nano** for efficient on-device person detection
- **Google Gemini 1.5 Flash** for intelligent behavioral analysis
- **Python Flask Backend** for data logging and management
- **Android ExoPlayer** for smooth video playback and processing

## Key Features

- Real-time person detection with bounding boxes
- AI-powered behavioral analysis and insights
- Backend data logging and storage
- Cross-platform compatibility (Android + Python backend)
- Network-based communication between devices
- JSON-based analysis export

## System Architecture

```
┌─────────────────┐    Network    ┌─────────────────┐
│   Android App   │◄─────────────►│ Python Backend │
│                 │    (WiFi)     │                 │
│ • YOLOv5 Model  │               │ • Flask Server  │
│ • Video Player  │               │ • Data Storage  │
│ • UI Interface  │               │ • API Endpoints │
└─────────────────┘               └─────────────────┘
         │                                 │
         ▼                                 ▼
┌─────────────────┐               ┌─────────────────┐
│  Gemini AI API  │               │  JSON Analysis  │
│                 │               │     Files       │
│ • Text Analysis │               │                 │
│ • Insights      │               │ • Timestamped   │
│ • Summaries     │               │ • Structured    │
└─────────────────┘               └─────────────────┘
```

## Prerequisites

### Hardware Requirements
- **Android Device**: Android 7.0+ (API 24+)
- **Computer**: For running Python backend server
- **Network**: Both devices must be on the same WiFi network

### Software Requirements
- Android Studio (for app development/installation)
- Python 3.8+ with pip
- Video files in MP4 or WebM format

## Installation

### Step 1: Backend Server Setup

1. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

2. **Start the backend server**:
```bash
python backend_final.py
```

The server will start on `http://0.0.0.0:5000`

### Step 2: Android App Configuration

1. **Get your computer's IP address**:

   **Windows**:
   ```cmd
   ipconfig
   ```
   Look for "IPv4 Address" under your WiFi adapter.

   **Mac/Linux**:
   ```bash
   ifconfig | grep "inet " | grep -v 127.0.0.1
   ```

2. **Update Android app configuration**:

   In `MainActivity.kt`, update these constants:
   ```kotlin
   // Line 67-68: Update with YOUR computer's IP address
   private val BACKEND_BASE_URL = "http://YOUR_COMPUTER_IP:5000" 
   // Example: "http://192.168.1.100:5000"
   private val BACKEND_DATA_ENDPOINT = "$BACKEND_BASE_URL/api/data"
   
   // Line 65: Add your Gemini API key
   private val GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
   ```

   In `network_security_config.xml`, update your IP address:
   ```xml
   <domain includeSubdomains="true">YOUR_COMPUTER_IP</domain>
   ```

3. **Get Gemini API Key** (Optional but recommended):
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy and paste into `GEMINI_API_KEY` constant
   
   *Note: The app works without Gemini using local analysis, but AI insights are enhanced with Gemini.*

4. **Add the YOLOv5 model**:
   Place `yolov5n.tflite` model file in `app/src/main/assets/` directory.

### Step 3: Build and Install

```bash
# Build and install via Android Studio
# OR use command line:
./gradlew installDebug
```

## Usage

### Step-by-Step Operation

1. **Start Backend**: Run `python backend_final.py` on your computer
2. **Open Android App**: Launch the BMW AI app on your device
3. **Select Video**: Tap "Select Video" and choose an MP4/WebM file
4. **Start Analysis**: Tap "Start" to begin AI processing
5. **View Results**: Monitor real-time detection with AI summary generation
6. **Backend Logging**: Analysis is automatically sent to the backend server

### Recommended Video Types
- Interior camera footage (dashboard/rearview mirror perspective)
- Carpool Karaoke style videos (multiple people in car)
- Any MP4 with people visible in automotive interior settings

## API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Check server status |
| POST | `/api/data` | Upload analysis data (used by Android app) |
| GET | `/api/data` | Retrieve all analysis summaries |

### Sample API Response

```json
{
  "timestamp": 1672847293847,
  "bmw_ai_summary": "Executive Summary: Comprehensive person detection analysis...",
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
```

## Project Structure

```
Car-monitoring/
├── android/
│   ├── app/
│   │   ├── src/main/
│   │   │   ├── assets/
│   │   │   │   └── yolov5n.tflite
│   │   │   ├── java/.../
│   │   │   │   └── MainActivity.kt
│   │   │   └── res/
│   │   └── build.gradle
│   └── gradle files
├── backend/
│   ├── backend_final.py
│   ├── requirements.txt
│   └── data/
│       └── analysis_*.json
└── README.md
```

## Performance Expectations

- **Detection**: 1-3 people detected in typical interior footage
- **Processing**: Real-time at ~10 FPS on modern Android devices
- **AI Summary**: Generated in 5-15 seconds with Gemini
- **Backend Logging**: Immediate data transmission after analysis

## Troubleshooting

### Backend Connection Issues

**Symptoms**: "Cannot reach backend server" in Android app

**Solutions**:
1. Verify backend server is running (`python backend_final.py`)
2. Check IP address in Android app matches computer IP
3. Ensure both devices are on the same WiFi network
4. Try disabling computer firewall temporarily
5. Test with browser: `http://YOUR_IP:5000/api/health`

### Gemini API Issues

**Symptoms**: "AI Unavailable - Using Local Analysis"

**Solutions**:
1. Verify API key is correct in `GEMINI_API_KEY`
2. Check internet connection on Android device
3. App continues working with local analysis if Gemini fails

### Video Loading Issues

**Symptoms**: "Error loading video"

**Solutions**:
1. Ensure video is MP4 format
2. Try shorter video files (< 2 minutes for testing)
3. Grant storage permissions when prompted

## Testing

Test backend connection manually:

```bash
# Health check
curl http://YOUR_IP:5000/api/health

# View stored data
curl http://YOUR_IP:5000/api/data
```

Expected backend output:
```
📊 BMW AI Analysis Received:
Session: bmw_session_1672847293847
Video Duration: 45s
Detection Events: 127
Saved to: bmw_data/analysis_20231204_143456.json
```

## Data Analysis

Analysis data is automatically saved in the `data/` directory as timestamped JSON files. Each file contains:
- Detection events and counts
- Video metadata (duration, resolution)
- Device information
- AI-generated behavioral insights
- Session timestamps

## Applications

- **Automotive Safety Research**: Driver and passenger behavior analysis
- **Fleet Management**: Interior monitoring for commercial vehicles
- **Insurance Analytics**: Risk assessment based on occupancy patterns
- **Autonomous Vehicle Development**: Interior state awareness systems
- **Security Systems**: Real-time occupancy detection

## Technical Stack

- **Mobile**: Android (Kotlin), ExoPlayer, TensorFlow Lite
- **Backend**: Python, Flask, JSON storage
- **AI/ML**: YOLOv5 Nano, Google Gemini 1.5 Flash
- **Communication**: REST API, WiFi networking

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make your changes
4. Test thoroughly with different video inputs
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YOLOv5 by Ultralytics for efficient object detection
- Google Gemini AI for intelligent analysis capabilities
- TensorFlow Lite for mobile inference optimization

## Contact

For questions or support regarding this BMW AI system:
- GitHub: [@PayalSorathiya](https://github.com/PayalSorathiya)
- Project: [Car-monitoring](https://github.com/PayalSorathiya/Car-monitoring)

---

**Disclaimer**: This system is designed for research and development purposes in automotive safety applications. Ensure compliance with privacy regulations when processing interior camera footage.
