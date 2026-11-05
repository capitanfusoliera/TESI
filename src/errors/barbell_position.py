# controlla che il bilanciere non sia posizionato
# troppo in alto rispetto alle spalle

BARBELL_POSITION_THRESHOLD_RATIO = 0.02


def check_barbell_position(body,reference_body_height):

    if reference_body_height is None or reference_body_height == 0:
        return False

    #ho bisogno della posizione di polsi e spalle
    left_wrist = body.landmarks.polso_sinistro
    right_wrist = body.landmarks.polso_destro

    left_shoulder = body.landmarks.spalla_sinistra
    right_shoulder = body.landmarks.spalla_destra


    # calcolo il centro verticale dei polsi
    wrist_center_y = (left_wrist.y + right_wrist.y) / 2

    # calcolo il centro verticale delle spalle
    shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2

    # dato che l'asse y cresce verso il basso
    # il valore è positivo quando i polsi
    # si trovano sopra le spalle
    wrist_height_difference = (shoulder_center_y - wrist_center_y)

    if (wrist_height_difference < 0):
        return False

    threshold = reference_body_height * BARBELL_POSITION_THRESHOLD_RATIO

    if ( wrist_height_difference > threshold):
        return True


    return False