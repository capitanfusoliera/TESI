# Si occupa di rilevare i marker colorati
# e restituire le coordinate dei marker rilevati

import cv2
import numpy as np


# Range HSV stretto per il magenta scelto
# Colore teorico del marker:
# HSV OpenCV circa = [147, 196, 246]

LOWER_MAGENTA = np.array([135, 70, 90])
UPPER_MAGENTA = np.array([175, 255, 255])

# Area minima del marker in pixel
# Le regioni più piccole vengono ignorate
MIN_AREA = 100


def detect_markers_image(image_path):

    # Apertura e lettura dell'immagine
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Conversione da BGR a HSV
    # HSV facilita il riconoscimento del colore
    hsv_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # Creo una maschera:
    # i pixel compresi nel range magenta diventano bianchi
    # tutti gli altri diventano neri
    marker_mask = cv2.inRange(
        hsv_image,
        LOWER_MAGENTA,
        UPPER_MAGENTA
    )

    # Scala utilizzata solamente per visualizzare
    # le immagini sullo schermo
    DISPLAY_SCALE = 0.4

    # Creo una copia ridimensionata dell'immagine originale
    display_image = cv2.resize(
        image,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE,
        interpolation=cv2.INTER_AREA
    )

    # Creo una copia ridimensionata della maschera
    display_mask = cv2.resize(
        marker_mask,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE,
        interpolation=cv2.INTER_NEAREST
    )

    # Cerco i contorni delle regioni bianche nella maschera
    contours, _ = cv2.findContours(
        marker_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Lista che conterrà tutti i marker trovati
    markers = []

    # Analizzo ogni contorno trovato
    for contour in contours:

        # Calcolo l'area del contorno
        area = cv2.contourArea(contour)

        # Se l'area è troppo piccola,
        # ignoro questa regione
        if area < MIN_AREA:
            continue

        # Calcolo i momenti geometrici del contorno
        moments = cv2.moments(contour)

        # Evito una possibile divisione per zero
        if moments["m00"] == 0:
            continue

        # Calcolo il centro geometrico del marker
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]

        # Salvo il marker trovato
        markers.append({
            "x": center_x,
            "y": center_y,
            "area": area
        })

    # Creo una copia dell'immagine originale
    # sulla quale disegnare i marker trovati
    debug_image = image.copy()

    # Disegno ogni marker trovato
    for marker_id, marker in enumerate(markers):

        x = int(marker["x"])
        y = int(marker["y"])

        # Disegno un cerchio verde sul centro del marker
        cv2.circle(
            debug_image,
            (x, y),
            8,
            (0, 255, 0),
            2
        )

        # Scrivo l'ID del marker vicino al centro
        cv2.putText(
            debug_image,
            str(marker_id),
            (x + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Creo una copia ridimensionata
    # dell'immagine con i marker disegnati
    display_debug_image = cv2.resize(
        debug_image,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE,
        interpolation=cv2.INTER_AREA
    )

    # Visualizzo:
    # 1. immagine originale
    # 2. maschera del magenta
    # 3. marker individuati
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