# controlla che le ginocchia siano posizionate
# in maniera abbastanza simmetrica rispetto al centro del bacino

KNEE_ASYMMETRY_THRESHOLD_RATIO = 0.15


def check_knee_asymmetry(body):

    hip_center = body.virtual_points.centro_anche

    left_knee = body.landmarks.ginocchio_sinistro
    right_knee = body.landmarks.ginocchio_destro


    # distanza orizzontale delle ginocchia
    # rispetto al centro del bacino

    left_distance = abs(
        left_knee.x - hip_center.x
    )

    right_distance = abs(
        right_knee.x - hip_center.x
    )


    # distanza media delle ginocchia dal centro

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


    if difference_ratio > KNEE_ASYMMETRY_THRESHOLD_RATIO:
        return True


    return False