import cv2
import numpy as np


def create_skeleton_frame(body, width, height):

    # creo un frame completamente nero
    frame = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )


    # prendo tutti i segmenti già creati nel BodyModel
    for segment_name, segment in vars(body.segments).items():

        start_x = int(segment.start.x)
        start_y = int(segment.start.y)

        end_x = int(segment.end.x)
        end_y = int(segment.end.y)


        # disegno l'osso
        cv2.line(
            frame,
            (start_x, start_y),
            (end_x, end_y),
            (0, 255, 0),
            2
        )


        # disegno i punti alle estremità
        cv2.circle(
            frame,
            (start_x, start_y),
            4,
            (0, 255, 0),
            -1
        )

        cv2.circle(
            frame,
            (end_x, end_y),
            4,
            (0, 255, 0),
            -1
        )


    return frame