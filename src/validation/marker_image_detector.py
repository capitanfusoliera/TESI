import cv2
import numpy as np


LOWER_MAGENTA = np.array([135, 70, 90])
UPPER_MAGENTA = np.array([175, 255, 255])

MIN_AREA = 20


def detect_markers_frame(image):

    # Convert image from BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create mask for magenta markers
    marker_mask = cv2.inRange(
        hsv_image,
        LOWER_MAGENTA,
        UPPER_MAGENTA
    )

    # Find contours
    contours, _ = cv2.findContours(
        marker_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    markers = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_AREA:
            continue

        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            continue

        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]

        markers.append({
            "x": center_x,
            "y": center_y,
            "area": area
        })

    return markers, marker_mask

def detect_markers_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    markers, marker_mask = detect_markers_frame(image)

    print(f"\nDetected markers: {len(markers)}")

    for marker_id, marker in enumerate(markers):

        print(
            f"Marker {marker_id}: "
            f"X = {marker['x']:.1f} px | "
            f"Y = {marker['y']:.1f} px | "
            f"area = {marker['area']:.1f}"
        )

    return markers