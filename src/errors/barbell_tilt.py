# si controlla che l'inclinazione del bilanciere sia corretta
# si sceglie una soglia e si controlla frame per frame
# che la soglia non venga superata

BARBELL_TILT_THRESHOLD_RATIO = 0.25

def check_barbell_tilt(body, reference_body_height):
    left_wrist = body.landmarks.polso_sinistro
    right_wrist = body.landmarks.polso_destro

    # differenza in pixel fra i polsi sull'asse y
    wrist_delta_y = abs(left_wrist.y - right_wrist.y)

    # distanza in pixel fra i polsi sull'asse x
    wrist_delta_x = abs(left_wrist.x - right_wrist.x)

    # calcolo la differenza massima in altezza
    # consentita in base alla larghezza della presa
    threshold = wrist_delta_x * BARBELL_TILT_THRESHOLD_RATIO

    if wrist_delta_y > threshold:
        return True

    return False