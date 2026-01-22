# AI CCTV Engine 🎥🤖

An intelligent video surveillance system powered by AI for real-time object detection, tracking, and security monitoring in warehouse environments.

## 📋 Overview

AI CCTV Engine is a smart surveillance solution that leverages computer vision and deep learning to provide automated monitoring, object tracking, and security analytics for CCTV footage. The system uses YOLOv8 for object detection and provides real-time analysis with evidence capture capabilities.

## ✨ Features

- **Real-time Object Detection**: Powered by YOLOv8n for fast and accurate object detection
- **Object Tracking**: Multi-object tracking across video frames
- **Evidence Collection**: Automatic snapshot capture of detected events
- **Analytics Dashboard**: Comprehensive logging and analysis of warehouse activities
- **REST API**: Server-based architecture for easy integration
- **CSV Logging**: Detailed warehouse activity logs for audit trails
- **Customizable Detection**: Configurable detection thresholds and alert systems

## 🏗️ Project Structure

```
ai-cctv-engine/
├── analytics/              # Analytics and visualization modules
├── evidence_snapshots/     # Captured evidence images
├── input_videos/           # Source video files for processing
├── server.py              # Flask/FastAPI server for API endpoints
├── tracking2.py           # Core object tracking implementation
├── warehouse_logs.csv     # Activity logs and analytics data
├── yolov8n.pt            # YOLOv8 nano model weights
├── requirements.txt       # Python dependencies
└── .gitignore            # Git ignore rules
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip package manager
- Webcam or video files for processing
- (Optional) GPU with CUDA support for better performance

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Rishab0572/ai-cctv-engine.git
   cd ai-cctv-engine
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLOv8 model** (if not included)
   ```bash
   # The yolov8n.pt model should already be in the repository
   # If not, it will be automatically downloaded on first run
   ```

### Quick Start

1. **Run the tracking system**
   ```bash
   python tracking2.py
   ```

2. **Start the API server**
   ```bash
   python server.py
   ```

3. **Place your video files**
   - Add CCTV footage to the `input_videos/` directory
   - The system will process videos and generate analytics

## 🔧 Configuration

### Detection Settings

Edit the configuration in `tracking2.py` to customize:
- Detection confidence threshold
- Object classes to track
- Frame processing rate
- Evidence capture frequency

### Server Settings

Configure API endpoints and ports in `server.py`:
```python
# Example configuration
HOST = "0.0.0.0"
PORT = 5000
DEBUG = True
```

## 📊 Analytics

The system generates comprehensive analytics including:
- Object detection counts
- Tracking duration
- Entry/exit logs
- Activity heatmaps
- Timestamp-based event logs

All analytics are stored in `warehouse_logs.csv` and can be visualized using the analytics module.

## 📁 Output Files

### Evidence Snapshots
- Location: `evidence_snapshots/`
- Format: Timestamped image files
- Automatic capture on detection events

### Warehouse Logs
- Location: `warehouse_logs.csv`
- Contains: Timestamp, object type, position, duration
- Format: CSV for easy analysis in Excel/Python

## 🔌 API Endpoints

### Example API Usage

```bash
# Get detection status
GET /api/status

# Start video processing
POST /api/process
{
  "video_path": "input_videos/warehouse_cam1.mp4",
  "confidence": 0.5
}

# Get analytics
GET /api/analytics?date=2026-01-22
```

## 🛠️ Technology Stack

- **YOLOv8**: Object detection model
- **OpenCV**: Video processing and computer vision
- **Python**: Core programming language
- **Flask/FastAPI**: Web server framework
- **NumPy**: Numerical computations
- **Pandas**: Data analysis and logging
- **Ultralytics**: YOLO model implementation

## 📈 Use Cases

- **Warehouse Monitoring**: Track inventory movement and personnel
- **Security Surveillance**: Detect unauthorized access or suspicious activity
- **Retail Analytics**: Customer tracking and behavior analysis
- **Traffic Monitoring**: Vehicle detection and counting
- **Safety Compliance**: PPE detection and safety protocol monitoring

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Rishab Rajanikanth Rebala**
- GitHub: [@Rishab0572](https://github.com/Rishab0572)

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- OpenCV community for computer vision tools
- Contributors and testers

## 📧 Contact

For questions, suggestions, or support, please open an issue on GitHub or contact the repository owner.

---

⭐ If you find this project helpful, please consider giving it a star!
