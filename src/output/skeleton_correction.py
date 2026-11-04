from copy import deepcopy
import math

from src.errors.barbell_position import BARBELL_POSITION_THRESHOLD_RATIO
from src.errors.torso_inclination import TORSO_INCLINATION_THRESHOLD


# aggiorna i punti virtuali dopo aver modificato
# alcuni landmark del corpo

def _update_virtual_points(body):

    left_shoulder = body.landmarks.spalla_sinistra
    right_shoulder = body.landmarks.spalla_destra

    left_hip = body.landmarks.anca_sinistra
    right_hip = body.landmarks.anca_destra


    body.virtual_points.centro_spalle.x = (
        left_shoulder.x + right_shoulder.x
    ) / 2

    body.virtual_points.centro_spalle.y = (
        left_shoulder.y + right_shoulder.y
    ) / 2


    body.virtual_points.centro_anche.x = (
        left_hip.x + right_hip.x
    ) / 2

    body.virtual_points.centro_anche.y = (
        left_hip.y + right_hip.y
    ) / 2



# sposta un punto mantenendo invariato
# tutto il resto

def _move_point(
    point,
    delta_x=0,
    delta_y=0
):

    point.x += delta_x
    point.y += delta_y



# ----------------------------------------------------
# BARBELL TILT
# ----------------------------------------------------

# se i due polsi sono ad altezze diverse
# li porto alla stessa altezza media

def _correct_barbell_tilt(body):

    left_wrist = body.landmarks.polso_sinistro
    right_wrist = body.landmarks.polso_destro


    wrist_center_y = (
        left_wrist.y + right_wrist.y
    ) / 2


    left_wrist.y = wrist_center_y

    right_wrist.y = wrist_center_y



# ----------------------------------------------------
# BARBELL POSITION
# ----------------------------------------------------

# se i polsi sono troppo alti rispetto alle spalle
# li riporto all'altezza massima consentita

def _correct_barbell_position(
    body,
    reference_body_height
):

    if reference_body_height is None:
        return

    if reference_body_height == 0:
        return


    left_wrist = body.landmarks.polso_sinistro
    right_wrist = body.landmarks.polso_destro

    left_shoulder = body.landmarks.spalla_sinistra
    right_shoulder = body.landmarks.spalla_destra


    wrist_center_y = (
        left_wrist.y + right_wrist.y
    ) / 2

    shoulder_center_y = (
        left_shoulder.y + right_shoulder.y
    ) / 2


    # posizione più alta consentita
    # ai polsi rispetto alle spalle

    allowed_wrist_y = (
        shoulder_center_y
        -
        (
            BARBELL_POSITION_THRESHOLD_RATIO
            *
            reference_body_height
        )
    )


    # y aumenta andando verso il basso
    # quindi un valore minore significa
    # che i polsi sono troppo in alto

    if wrist_center_y < allowed_wrist_y:

        correction_y = (
            allowed_wrist_y
            -
            wrist_center_y
        )


        left_wrist.y += correction_y

        right_wrist.y += correction_y



# ----------------------------------------------------
# KNEES
# ----------------------------------------------------

# corregge sia la convergenza delle ginocchia
# sia l'asimmetria fra ginocchio destro e sinistro

def _correct_knees(body):

    hip_center = body.virtual_points.centro_anche

    left_hip = body.landmarks.anca_sinistra
    right_hip = body.landmarks.anca_destra

    left_knee = body.landmarks.ginocchio_sinistro
    right_knee = body.landmarks.ginocchio_destro


    left_distance = abs(
        left_knee.x - hip_center.x
    )

    right_distance = abs(
        right_knee.x - hip_center.x
    )


    # distanza media attuale delle ginocchia
    # rispetto al centro del bacino

    average_distance = (
        left_distance + right_distance
    ) / 2


    # evito che la correzione lasci
    # le ginocchia dentro il range delle anche

    hip_width = abs(
        left_hip.x - right_hip.x
    )

    minimum_distance = (
        hip_width / 2
    ) * 1.05


    corrected_distance = max(
        average_distance,
        minimum_distance
    )


    # utilizzo le anche per capire
    # da quale lato si trova ogni gamba

    if left_hip.x < hip_center.x:

        left_sign = -1

    else:

        left_sign = 1


    if right_hip.x < hip_center.x:

        right_sign = -1

    else:

        right_sign = 1


    left_knee.x = (
        hip_center.x
        +
        left_sign * corrected_distance
    )

    right_knee.x = (
        hip_center.x
        +
        right_sign * corrected_distance
    )



# ----------------------------------------------------
# HEELS ASYMMETRY
# ----------------------------------------------------

# posiziona i due piedi in maniera simmetrica
# rispetto al centro del bacino

def _correct_heel_asymmetry(body):

    hip_center = body.virtual_points.centro_anche

    left_hip = body.landmarks.anca_sinistra
    right_hip = body.landmarks.anca_destra

    left_heel = body.landmarks.tallone_sinistro
    right_heel = body.landmarks.tallone_destro


    left_distance = abs(
        left_heel.x - hip_center.x
    )

    right_distance = abs(
        right_heel.x - hip_center.x
    )


    corrected_distance = (
        left_distance + right_distance
    ) / 2


    if left_hip.x < hip_center.x:

        left_sign = -1

    else:

        left_sign = 1


    if right_hip.x < hip_center.x:

        right_sign = -1

    else:

        right_sign = 1


    corrected_left_x = (
        hip_center.x
        +
        left_sign * corrected_distance
    )

    corrected_right_x = (
        hip_center.x
        +
        right_sign * corrected_distance
    )


    left_correction_x = (
        corrected_left_x
        -
        left_heel.x
    )

    right_correction_x = (
        corrected_right_x
        -
        right_heel.x
    )


    # sposto tutto il piede sinistro
    # della stessa quantità

    _move_point(
        body.landmarks.tallone_sinistro,
        delta_x=left_correction_x
    )

    _move_point(
        body.landmarks.caviglia_sinistra,
        delta_x=left_correction_x
    )

    _move_point(
        body.landmarks.avampiede_sinistro,
        delta_x=left_correction_x
    )


    # sposto tutto il piede destro
    # della stessa quantità

    _move_point(
        body.landmarks.tallone_destro,
        delta_x=right_correction_x
    )

    _move_point(
        body.landmarks.caviglia_destra,
        delta_x=right_correction_x
    )

    _move_point(
        body.landmarks.avampiede_destro,
        delta_x=right_correction_x
    )



# ----------------------------------------------------
# HEEL LIFT
# ----------------------------------------------------

# riporta il tallone alla posizione verticale
# salvata inizialmente

def _correct_heel_lift(
    body,
    reference_left_heel_y,
    reference_right_heel_y
):

    if (
        reference_left_heel_y is None
        or
        reference_right_heel_y is None
    ):
        return


    left_heel = body.landmarks.tallone_sinistro
    right_heel = body.landmarks.tallone_destro


    # correggo solamente se il tallone
    # si trova più in alto del riferimento

    if left_heel.y < reference_left_heel_y:

        left_heel.y = (
            reference_left_heel_y
        )


    if right_heel.y < reference_right_heel_y:

        right_heel.y = (
            reference_right_heel_y
        )



# ----------------------------------------------------
# TORSO INCLINATION
# ----------------------------------------------------

# se la lunghezza apparente del torso
# indica un'inclinazione superiore alla soglia
# riporto il torso alla massima inclinazione consentita

def _correct_torso_inclination(
    body,
    reference_torso_length
):

    if reference_torso_length is None:
        return

    if reference_torso_length == 0:
        return


    shoulder_center = body.virtual_points.centro_spalle
    hip_center = body.virtual_points.centro_anche


    delta_x = (
        shoulder_center.x - hip_center.x
    )

    delta_y = (
        shoulder_center.y - hip_center.y
    )


    current_torso_length = math.hypot(
        delta_x,
        delta_y
    )


    if current_torso_length == 0:
        return


    # lunghezza minima che il torso deve avere
    # nell'immagine per non superare
    # l'inclinazione massima consentita

    minimum_torso_length = (
        reference_torso_length
        *
        math.cos(
            math.radians(
                TORSO_INCLINATION_THRESHOLD
            )
        )
    )


    if current_torso_length >= minimum_torso_length:
        return


    correction_ratio = (
        minimum_torso_length
        /
        current_torso_length
    )


    corrected_shoulder_center_x = (
        hip_center.x
        +
        delta_x * correction_ratio
    )

    corrected_shoulder_center_y = (
        hip_center.y
        +
        delta_y * correction_ratio
    )


    correction_x = (
        corrected_shoulder_center_x
        -
        shoulder_center.x
    )

    correction_y = (
        corrected_shoulder_center_y
        -
        shoulder_center.y
    )


    # sposto tutta la parte superiore del corpo
    # per non deformare braccia e spalle

    upper_body_points = [
        body.landmarks.spalla_sinistra,
        body.landmarks.spalla_destra,
        body.landmarks.gomito_sinistro,
        body.landmarks.gomito_destro,
        body.landmarks.polso_sinistro,
        body.landmarks.polso_destro,
        body.landmarks.naso
    ]


    for point in upper_body_points:

        _move_point(
            point,
            delta_x=correction_x,
            delta_y=correction_y
        )


    _update_virtual_points(
        body
    )



# ----------------------------------------------------
# CREATE CORRECTED BODY
# ----------------------------------------------------

def create_corrected_body(
    body,
    detected_errors,
    error_manager
):

    # creo una copia del corpo
    # per non modificare quello originale

    corrected_body = deepcopy(
        body
    )


    # correggo prima il torso
    # perché può modificare anche spalle e polsi

    if detected_errors.get(
        "torso_inclination",
        False
    ):

        _correct_torso_inclination(
            corrected_body,
            error_manager.reference_torso_length
        )


    # convergenza e asimmetria delle ginocchia
    # possono utilizzare la stessa correzione

    if (
        detected_errors.get(
            "knee_convergence",
            False
        )
        or
        detected_errors.get(
            "knee_asymmetry",
            False
        )
    ):

        _correct_knees(
            corrected_body
        )


    if detected_errors.get(
        "heel_asymmetry",
        False
    ):

        _correct_heel_asymmetry(
            corrected_body
        )


    if detected_errors.get(
        "heel_lift",
        False
    ):

        _correct_heel_lift(
            corrected_body,
            error_manager.reference_left_heel_y,
            error_manager.reference_right_heel_y
        )


    # correggo la posizione del bilanciere
    # dopo aver sistemato il torso

    if detected_errors.get(
        "barbell_position",
        False
    ):

        _correct_barbell_position(
            corrected_body,
            error_manager.reference_body_height
        )


    # per ultimo rendo i polsi alla stessa altezza

    if detected_errors.get(
        "barbell_tilt",
        False
    ):

        _correct_barbell_tilt(
            corrected_body
        )


    _update_virtual_points(
        corrected_body
    )


    return corrected_body