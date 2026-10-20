# si controlla che durante lo squat
# le ginocchia non convergano verso l'interno

def check_knee_convergence(body):

    left_hip = body.landmarks.anca_sinistra
    right_hip = body.landmarks.anca_destra

    left_knee = body.landmarks.ginocchio_sinistro
    right_knee = body.landmarks.ginocchio_destro

    # creo il range orizzontale compreso fra le due anche

    hip_min_x = min(left_hip.x, right_hip.x)
    hip_max_x = max(left_hip.x, right_hip.x)


    # controllo se il ginocchio sinistro o destro entra nel range delle anche

    if (hip_min_x <= left_knee.x <= hip_max_x) or (hip_min_x <= right_knee.x <= hip_max_x):
        return True

    return False