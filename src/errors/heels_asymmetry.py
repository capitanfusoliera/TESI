# controlla che i talloni siano posizionati
# in maniera abbastanza simmetrica rispetto al centro del bacino

HEEL_ASYMMETRY_THRESHOLD_RATIO = 0.2


def check_heel_asymmetry(body):

    hip_center = body.virtual_points.centro_anche

    left_heel = body.landmarks.tallone_sinistro
    right_heel = body.landmarks.tallone_destro


    # distanza orizzontale dei talloni
    # rispetto al centro del bacino

    left_distance = abs(
        left_heel.x - hip_center.x
    )

    right_distance = abs(
        right_heel.x - hip_center.x
    )


    # distanza media dei talloni dal centro

    average_distance = (
        left_distance + right_distance
    ) / 2


    if average_distance == 0:
        return False


    # differenza tra le due distanze

    distance_difference = abs(
        left_distance - right_distance
    )


    # rapporto la differenza alla distanza media

    difference_ratio = (
        distance_difference
        /
        average_distance
    )


    if difference_ratio > HEEL_ASYMMETRY_THRESHOLD_RATIO:
        return True


    return False