import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')

min_width_react = 80
min_height_react = 80

count_line_position = 550

# Use createBackgroundSubtractorMOG2
algo = cv2.createBackgroundSubtractorMOG2()

def center_handle(x, y, w, h):
    cx = int(x + w / 2)
    cy = int(y + h / 2)
    return cx, cy

detect = []   
offset = 6
counter = 0

while True:
    ret, frame1 = cap.read()
    if not ret:
        break  # Exit if the video has ended

    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)

    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    counterShape, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the counting line
    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)

    for (i, c) in enumerate(counterShape):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_react) and (h >= min_height_react)
        if not validate_counter:
            continue
        
        center = center_handle(x, y, w, h)

        # Check if the center of the vehicle is near the counting line
        if center[1] < (count_line_position + offset) and center[1] > (count_line_position - offset):
            if center not in detect:  # Only count if this vehicle hasn't been counted yet
                counter += 1
                detect.append(center)  # Add to detected list to avoid double counting
                print("Vehicle Counter: " + str(counter))
                # Change the color of the line after counting
                cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)
        
        # Draw rectangle around detected vehicle
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame1, f"Vehicle {counter}", (x, y - 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 244, 0), 2)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

    # Display the vehicle count
    cv2.putText(frame1, "VEHICLE COUNTER: " + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    # Show the processed frames
    cv2.imshow('Video Original', frame1)

    if cv2.waitKey(1) == 13:  # Press Enter to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()