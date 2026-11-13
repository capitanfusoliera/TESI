# si controlla che l'inclinazione del bilanciere sia corretta
# si sceglie una soglia e si controlla frame per frame
# che la soglia non venga superata

BARBELL_TILT_THRESHOLD_RATIO = 0.05

#riceve i dati del frame
def check_barbell_tilt(body):

    left_wrist = body.landmarks.polso_sinistro
    right_wrist = body.landmarks.polso_destro

    # differenza in pixel fra i polsi sull'asse y
    wrist_delta_y = abs(left_wrist.y - right_wrist.y)

    # differenza in pixel fra i polsi sull'asse x
    wrist_delta_x = abs(left_wrist.x - right_wrist.x)

    if wrist_delta_x == 0:
        return False

    # calcolo la soglia in base
    # alla distanza orizzontale fra i polsi
    threshold = wrist_delta_x * BARBELL_TILT_THRESHOLD_RATIO

    if wrist_delta_y > threshold:
        return True


    return False