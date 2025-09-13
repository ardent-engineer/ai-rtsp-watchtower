
---

# Intelligent Watch Tower

A high-performance, AI-powered surveillance application that processes multiple camera feeds concurrently without lag. Built with Python, PyQt5, and YOLOv5.

![alt text](<illustrations/app.png>)

## Key Features

-   **Lag-Free Concurrent Processing**: Utilizes a multi-threaded architecture where each camera feed is processed independently. This prevents slow or disconnected streams from impacting the performance of others, ensuring a responsive UI with no video lag buildup.
![alt text](illustrations/graph.png)
-   **Multi-Source Video**: Supports RTSP streams, local video files, and direct screen captures.
-   **Real-Time AI Detection**: Detects Persons, Vehicles, and Animals using a YOLOv5 model.
-   **Customizable Detection Zones**: Define specific Regions of Interest (ROI) and Exclusion Regions (ER) for precise monitoring.
![alt text](<illustrations/roi-selector.png>)

![alt text](<illustrations/roi-selected.png>)
-   **Priority-Based Audible Alerts**: Assign different alarm sounds based on camera priority.
-   **Dynamic Grid View**: Switch between 2x2, 3x3, and 4x4 layouts on the fly.
-   **Reduced False Alarms**: Implements logic to reduce false alarms.
## Quick Setup

Getting started is designed to be as simple as possible.

### 1. Clone This Repository

```bash
git clone https://github.com/ardent-engineer/ai-rtsp-watchtower.git
cd intelligent-watch-tower
```

### 2. Clone YOLOv5

The object detection model is a direct dependency. Clone the official repository into the project's root directory.

```bash
# This MUST be cloned inside the 'intelligent-watch-tower' folder
git clone https://github.com/ultralytics/yolov5.git
```

### 3. Install Dependencies & Configure

Install the requirements and set up your cameras.

```bash
# (Optional but Recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install all required packages
pip install -r requirements.txt
```

**Next, edit `data/sources.csv` to add your video streams.**

The format is `DisplayName,SourceURL,ConfidenceThreshold`. For example:
```csv
Front Door,rtsp://user:pass@192.168.1.50/stream1,0.45
Test Video,C:/Users/Admin/Videos/test.mp4,0.5
Main Monitor,s1,0.3
```

## Run the Application

Run the application as a module from the **root directory**.

```bash
python -m watch_tower
```
> **Admin Password**: The default is `123`.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.