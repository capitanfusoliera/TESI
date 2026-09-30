import cv2
import numpy as np


# HSV range for magenta markers
LOWER_MAGENTA = np.array([135, 70, 90])
UPPER_MAGENTA = np.array([175, 255, 255])

# Marker area limits
MIN_AREA = 20
MAX_AREA = 5000

# Used only for visualization
DISPLAY_SCALE = 0.4


def detect_markers_frame(frame):

    # Convert BGR frame to HSV
    hsv_image = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

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

        # Ignore regions that are too small or too large
        if area < MIN_AREA or area > MAX_AREA:
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

    markers, marker_mask = detect_markers_frame(
        image
    )

    debug_image = image.copy()

    for marker_id, marker in enumerate(markers):

        x = int(marker["x"])
        y = int(marker["y"])

        cv2.circle(
            debug_image,
            (x, y),
            8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            debug_image,
            str(marker_id),
            (x + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    display_image = cv2.resize(
        image,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE,
        interpolation=cv2.INTER_AREA
    )

    display_mask = cv2.resize(
        marker_mask,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE,
        interpolation=cv2.INTER_NEAREST
    )

    display_debug_image = cv2.resize(
        debug_image,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE,
        interpolation=cv2.INTER_AREA
    )

    cv2.imshow(
        "Original image",
        display_image
    )

    cv2.imshow(
        "Marker mask",
        display_mask
    )

    cv2.imshow(
        "Detected markers",
        display_debug_image
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return markers