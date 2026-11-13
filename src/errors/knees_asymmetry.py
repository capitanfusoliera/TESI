# controlla che le ginocchia siano posizionate
# in maniera abbastanza simmetrica rispetto al centro del bacino

#moltiplicatore dell'errore di asimmetria delle ginocchia rispetto all'altezza tallone anca
KNEE_ASYMMETRY_THRESHOLD_RATIO = 0.03


def check_knee_asymmetry(body,reference_heel_hip_height):

    if reference_heel_hip_height is None:
        return True

    hip_center = body.virtual_points.centro_anche

    left_knee = body.landmarks.ginocchio_sinistro
    right_knee = body.landmarks.ginocchio_destro

    # distanza orizzontale delle ginocchia
    # rispetto al centro del bacino
    left_distance = abs(left_knee.x - hip_center.x)

    right_distance = abs(right_knee.x - hip_center.x)


    # differenza tra le due distanze in valore assoluto
    distance_difference = abs(left_distance - right_distance)

    # calcolo la soglia in base
    # all'altezza tallone anca
    threshold = reference_heel_hip_height * KNEE_ASYMMETRY_THRESHOLD_RATIO

    if distance_difference > threshold:
        return True


    return False