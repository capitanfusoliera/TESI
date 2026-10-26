# controlla che il bilanciere non sia posizionato
# troppo in alto rispetto alle spalle

BARBELL_POSITION_THRESHOLD_RATIO = 0.03


def check_barbell_position(
    body,
    reference_body_height
):

    if reference_body_height is None:
        return False

    if reference_body_height == 0:
        return False


    left_wrist = body.landmarks.polso_sinistro
    right_wrist = body.landmarks.polso_destro

    left_shoulder = body.landmarks.spalla_sinistra
    right_shoulder = body.landmarks.spalla_destra


    # calcolo il centro verticale dei polsi

    wrist_center_y = (
        left_wrist.y + right_wrist.y
    ) / 2


    # calcolo il centro verticale delle spalle

    shoulder_center_y = (
        left_shoulder.y + right_shoulder.y
    ) / 2


    # dato che l'asse y cresce verso il basso
    # il valore è positivo quando i polsi
    # si trovano sopra le spalle

    wrist_height_difference = (
        shoulder_center_y - wrist_center_y
    )


    # rapporto la distanza all'altezza
    # della persona

    wrist_height_ratio = (
        wrist_height_difference
        /
        reference_body_height
    )


    if (
        wrist_height_ratio
        > BARBELL_POSITION_THRESHOLD_RATIO
    ):
        return True


    return False