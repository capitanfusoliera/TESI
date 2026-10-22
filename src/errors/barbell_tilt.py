# si controlla che l'inclinazione del bilanciere sia corretta
# si sceglie una soglia e si controlla frame per frame
# che la soglia non venga superata

BARBELL_TILT_THRESHOLD_RATIO = 0.025

#riceve il i dati del frame e l'altezza dell'atleta in pixel
def check_barbell_tilt(body, reference_body_height):

    left_wrist = body.landmarks.polso_sinistro
    right_wrist = body.landmarks.polso_destro

    left_ankle = body.landmarks.caviglia_sinistra
    right_ankle = body.landmarks.caviglia_destra


    # differenza in pixel fra i polsi
    wrist_delta_y = abs(left_wrist.y - right_wrist.y)

    # differenza in pixel fra le caviglie
    ankle_delta_y = abs(left_ankle.y - right_ankle.y)


    if reference_body_height is None:
        return False

    if reference_body_height == 0:
        return False


    # confronto la differenza dei polsi
    # con l'altezza iniziale della persona in piedi
    wrist_difference_ratio = (wrist_delta_y/reference_body_height)

    detected = (wrist_difference_ratio>BARBELL_TILT_THRESHOLD_RATIO)

    return detected