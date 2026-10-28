import cv2
import numpy as np
import os

# -------------------------------
# 1. Load Video
# -------------------------------

# Build the path relative to this script's folder (works on any PC)
script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, "video.mp4")

print("Looking for video at:", video_path)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Video file not found or cannot be opened!")
    print("Please check that video.mp4 is in the same folder as this script:")
    print(script_dir)
    exit()

print("Video opened successfully!")
# -------------------------------
# 2. Vehicle Detection Settings
# -------------------------------

min_width_react = 80
min_height_react = 80

count_line_position = 550
offset = 6

counter = 0

# Store previously detected vehicle centers
detect = []

# Background Subtractor
algo = cv2.createBackgroundSubtractorMOG2(
    history=100,
    varThreshold=40,
    detectShadows=False
)

# -------------------------------
# 3. Center Point Function
# -------------------------------

def center_handle(x, y, w, h):
    cx = int(x + w / 2)
    cy = int(y + h / 2)

    return cx, cy


# -------------------------------
# 4. Main Vehicle Detection Loop
# -------------------------------

while True:

    ret, frame1 = cap.read()

    if not ret:
        print("Video ended.")
        break

    # Convert frame to grayscale
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(grey, (3, 3), 5)

    # Background subtraction
    img_sub = algo.apply(blur)

    # Dilation
    dilat = cv2.dilate(
        img_sub,
        np.ones((5, 5), np.uint8),
        iterations=2
    )

    # Morphological operations
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    dilatada = cv2.morphologyEx(
        dilat,
        cv2.MORPH_CLOSE,
        kernel
    )

    dilatada = cv2.morphologyEx(
        dilatada,
        cv2.MORPH_CLOSE,
        kernel
    )

    # -------------------------------
    # 5. Find Contours
    # -------------------------------

    contours, _ = cv2.findContours(
        dilatada,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw counting line
    cv2.line(
        frame1,
        (25, count_line_position),
        (1200, count_line_position),
        (255, 127, 0),
        3
    )

    # -------------------------------
    # 6. Detect Vehicles
    # -------------------------------

    for c in contours:

        x, y, w, h = cv2.boundingRect(c)

        validate_counter = (
            w >= min_width_react
            and h >= min_height_react
        )

        if not validate_counter:
            continue

        center = center_handle(x, y, w, h)

        # Draw rectangle around vehicle
        cv2.rectangle(
            frame1,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Draw center point
        cv2.circle(
            frame1,
            center,
            4,
            (0, 0, 255),
            -1
        )

        # Vehicle label
        cv2.putText(
            frame1,
            "Vehicle",
            (x, y - 20),
            cv2.FONT_HERSHEY_TRIPLEX,
            1,
            (255, 244, 0),
            2
        )

        # -------------------------------
        # 7. Vehicle Counting Logic
        # -------------------------------

        if (
            count_line_position - offset
            < center[1]
            < count_line_position + offset
        ):

            # Check whether this vehicle
            # has already been counted

            already_counted = False

            for previous_center in detect:

                distance = abs(
                    center[0] - previous_center[0]
                )

                if distance < 50:
                    already_counted = True
                    break

            if not already_counted:

                counter += 1

                detect.append(center)

                print("Vehicle Counter:", counter)

                # Change counting line color
                cv2.line(
                    frame1,
                    (25, count_line_position),
                    (1200, count_line_position),
                    (0, 127, 255),
                    3
                )

    # -------------------------------
    # 8. Display Vehicle Count
    # -------------------------------

    cv2.putText(
        frame1,
        "VEHICLE COUNTER: " + str(counter),
        (450, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 255),
        4
    )

    # -------------------------------
    # 9. Display Video
    # -------------------------------

    cv2.imshow(
        "Vehicle Detection and Counting",
        frame1
    )

    # Press Enter or Q to exit
    key = cv2.waitKey(1) & 0xFF

    if key == 13 or key == ord("q"):
        break


# -------------------------------
# 10. Release Resources
# -------------------------------

cap.release()

cv2.destroyAllWindows()

print("Total Vehicles Counted:", counter)
print("Program completed successfully!")
