import math


# inclinazione massima consentita
# rispetto alla posizione iniziale

TORSO_INCLINATION_THRESHOLD = 45


def check_torso_inclination(
    body,
    reference_torso_length
):

    if reference_torso_length is None:
        return False

    if reference_torso_length == 0:
        return False


    shoulder_center = body.virtual_points.centro_spalle
    hip_center = body.virtual_points.centro_anche


    # calcolo la lunghezza attuale del torso
    # vista frontalmente

    current_torso_length = math.hypot(
        shoulder_center.x - hip_center.x,
        shoulder_center.y - hip_center.y
    )


    # rapporto tra lunghezza attuale
    # e lunghezza iniziale

    torso_ratio = (
        current_torso_length
        /
        reference_torso_length
    )


    # limito il rapporto tra 0 e 1
    # per evitare problemi con acos
    # dovuti a piccoli errori di mediapipe

    torso_ratio = max(
        0.0,
        min(1.0, torso_ratio)
    )


    # stimo l'inclinazione del torso

    torso_inclination = math.degrees(
        math.acos(
            torso_ratio
        )
    )


    if (
        torso_inclination
        > TORSO_INCLINATION_THRESHOLD
    ):
        return True


    return False