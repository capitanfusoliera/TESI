# controlla che i talloni siano posizionati
# in maniera abbastanza simmetrica rispetto al centro del bacino

HEEL_ASYMMETRY_THRESHOLD_RATIO = 0.045


def check_heel_asymmetry(body, reference_heel_hip_height):

    hip_center = body.virtual_points.centro_anche

    left_heel = body.landmarks.tallone_sinistro
    right_heel = body.landmarks.tallone_destro

    # distanza orizzontale dei talloni
    # rispetto al centro del bacino
    left_distance = abs(left_heel.x - hip_center.x)
    right_distance = abs(right_heel.x - hip_center.x)

    # differenza tra le due distanze
    distance_difference = abs(left_distance - right_distance)

    threshold = reference_heel_hip_height * HEEL_ASYMMETRY_THRESHOLD_RATIO

    if distance_difference > threshold:
        return True


    return False