#si occupa di rilevare i marker aruco e restituire le coordinate dei marker rilevati
import cv2
import numpy as np


# range primo rosso in HSV
LOWER_RED_1 = np.array([0, 100, 100])
UPPER_RED_1 = np.array([10, 255, 255])

# range secondo rosso in HSV
LOWER_RED_2 = np.array([170, 100, 100])
UPPER_RED_2 = np.array([179, 255, 255])


# Minimum marker area in pixels.
# Small red regions below this value are ignored.
MIN_AREA = 100


def detect_markers(image_path):

    # apertura e lettura dell'immagine
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # conversione da BGR to HSV per facilitare la rilevazione del colore rosso
    hsv_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # creo due maschere per il rosso, una per il primo range e una per il secondo
    # rendo bianchi solo i pixel rossi e neri tutti gli altri pixel

    mask_1 = cv2.inRange(
        hsv_image,
        LOWER_RED_1,
        UPPER_RED_1
    )

    mask_2 = cv2.inRange(
        hsv_image,
        LOWER_RED_2,
        UPPER_RED_2
    )

    # combino tutti i pixel bianchi
    red_mask = cv2.bitwise_or(
        mask_1,
        mask_2
    )

    # Scale used only to display the images on screen
    DISPLAY_SCALE = 0.4

    # Create smaller copies only for visualization
    display_image = cv2.resize(
        image,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE,
        interpolation=cv2.INTER_AREA
    )

    display_mask = cv2.resize(
        red_mask,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE,
        interpolation=cv2.INTER_NEAREST
    )

    # Display the resized images
    cv2.imshow("Original image", display_image)
    cv2.imshow("Red mask", display_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # creo dei quadrati di pixel bianchi nei punti trovate dalle maschere
    contours, _ = cv2.findContours(
        red_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    markers = []

    for contour in contours:    #cosa fare nei contrours rilevati

        area = cv2.contourArea(contour)

        # valuto un area minima per considerare un marker valido, altrimenti lo ignoro
        if area < MIN_AREA:
            continue

        # si crea un oggetto moments per calcolare il centro del marker
        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            continue

        # calcolo il centro del marker
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]

        markers.append({
            "x": center_x,
            "y": center_y,
            "area": area
        })

        # Debug: draw every detected marker
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

    # Creo una copia ridimensionata solo per la visualizzazione
    display_debug_image = cv2.resize(
        debug_image,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE,
        interpolation=cv2.INTER_AREA
    )

    cv2.imshow("Detected markers", display_debug_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return markers