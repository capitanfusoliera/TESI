# controlla se durante lo squat
# uno dei talloni si solleva dalla posizione iniziale

import math

HEEL_LIFT_RATIO = 0.3


def check_heel_lift(
    body,
    reference_left_heel_y,
    reference_right_heel_y
):

    if (
        reference_left_heel_y is None
        or
        reference_right_heel_y is None
    ):
        print("reference heel y is None, cannot check heel lift")
        return False


    left_heel = body.landmarks.tallone_sinistro
    right_heel = body.landmarks.tallone_destro

    left_forefoot = body.landmarks.avampiede_sinistro
    right_forefoot = body.landmarks.avampiede_destro


    # calcolo la lunghezza dei piedi nel frame corrente

    left_foot_length = math.hypot(
        left_forefoot.x - left_heel.x,
        left_forefoot.y - left_heel.y
    )

    right_foot_length = math.hypot(
        right_forefoot.x - right_heel.x,
        right_forefoot.y - right_heel.y
    )


    # calcolo la soglia in base alla lunghezza del piede

    left_threshold = HEEL_LIFT_RATIO * left_foot_length
    right_threshold = HEEL_LIFT_RATIO * right_foot_length


    # calcolo quanto si è spostato verticalmente il tallone
    # rispetto alla posizione iniziale

    left_heel_difference = abs(
        left_heel.y - reference_left_heel_y
    )

    right_heel_difference = abs(
        right_heel.y - reference_right_heel_y
    )


    if (
        left_heel_difference >= left_threshold
        or
        right_heel_difference >= right_threshold
    ):
        return True


    return False