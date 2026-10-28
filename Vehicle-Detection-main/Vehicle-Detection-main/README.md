# Vehicle Detection

## Overview

This project implements a vehicle detection and counting system using OpenCV and Python. It processes a video file, detects moving vehicles, and counts them when they pass a predefined line.

## Features

- Uses OpenCV's `createBackgroundSubtractorMOG2` for motion detection.
- Detects and tracks vehicles based on contours.
- Counts vehicles when they cross a specified line.
- Displays real-time video with detection and counting overlays.

## Requirements

Make sure you have the following dependencies installed:

```bash
pip install opencv-python numpy
```

## How It Works

1. The script reads the input video (`video.mp4`).
2. It converts frames to grayscale and applies Gaussian blur.
3. The background subtractor detects moving objects.
4. Contours are drawn around detected vehicles.
5. If a vehicle crosses the counting line, the counter increments.
6. The processed frames with detections and the vehicle count are displayed in real-time.

## Usage

Run the script using:

```bash
python vehicle_detection.py
```

### Controls

- **Press Enter (key 13)** to exit the program.

## Code Explanation

- **Background Subtraction:** Uses `cv2.createBackgroundSubtractorMOG2()` to identify moving vehicles.
- **Contour Detection:** `cv2.findContours()` detects vehicle shapes.
- **Counting Mechanism:** When the center of a detected vehicle crosses the counting line, it increments the counter.
- **Visualization:** The script draws bounding boxes around detected vehicles and overlays the count on the video.

## Project Structure

```
Vehicle Detection/
│── video.mp4                # Input video file
│── vehicle_detection.py      # Main script
│── README.md                # Project documentation
```

## Future Improvements

- Implement object tracking to improve counting accuracy.
- Enhance detection using deep learning models like YOLO.
- Support real-time vehicle detection via a webcam or live feed.

## License

This project is open-source and available under the MIT License.

## Author

Avinash Kumar -[https://github.com/Avistriker](https://github.com/Avistriker)



## **Contributions**

Feel free to submit issues or pull requests to improve the project!
