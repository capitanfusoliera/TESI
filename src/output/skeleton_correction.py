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
# li porto alla stessa altezza
def _correct_barbell_tilt(body):

    left_wrist = body.landmarks.polso_sinistro
    right_wrist = body.landmarks.polso_destro


    # altezza media dei due polsi
    wrist_center_y = (
        left_wrist.y
        +
        right_wrist.y
    ) / 2


    # mettendo i due polsi alla stessa altezza
    # la differenza verticale diventa zero
    left_wrist.y = wrist_center_y

    right_wrist.y = wrist_center_y



# ----------------------------------------------------
# BARBELL POSITION
# ----------------------------------------------------

# se il bilanciere è troppo alto
# rispetto alle spalle
# abbasso entrambi i polsi
# fino a portarli dentro la soglia corretta
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
        left_wrist.y
        +
        right_wrist.y
    ) / 2


    shoulder_center_y = (
        left_shoulder.y
        +
        right_shoulder.y
    ) / 2


    # soglia utilizzata anche
    # nel controllo dell'errore
    threshold = (
        reference_body_height
        *
        BARBELL_POSITION_THRESHOLD_RATIO
    )


    # non porto il bilanciere esattamente
    # sul limite ma leggermente
    # dentro la zona corretta
    corrected_difference = (
        threshold
        *
        0.90
    )


    # y aumenta andando verso il basso
    corrected_wrist_y = (
        shoulder_center_y
        -
        corrected_difference
    )


    correction_y = (
        corrected_wrist_y
        -
        wrist_center_y
    )


    # sposto entrambi i polsi
    # della stessa quantità
    left_wrist.y += correction_y

    right_wrist.y += correction_y



# ----------------------------------------------------
# KNEES
# ----------------------------------------------------

# corregge la convergenza delle ginocchia
# e l'asimmetria fra ginocchio destro
# e ginocchio sinistro
def _correct_knees(body):

    hip_center = body.virtual_points.centro_anche

    left_hip = body.landmarks.anca_sinistra
    right_hip = body.landmarks.anca_destra

    left_knee = body.landmarks.ginocchio_sinistro
    right_knee = body.landmarks.ginocchio_destro


    # distanza attuale delle ginocchia
    # rispetto al centro del bacino
    left_distance = abs(
        left_knee.x
        -
        hip_center.x
    )

    right_distance = abs(
        right_knee.x
        -
        hip_center.x
    )


    # uso la distanza media
    # in modo da rendere le due ginocchia
    # perfettamente simmetriche
    average_distance = (
        left_distance
        +
        right_distance
    ) / 2


    # calcolo la larghezza delle anche
    hip_width = abs(
        left_hip.x
        -
        right_hip.x
    )


    # il limite della convergenza
    # corrisponde alla metà
    # della larghezza delle anche
    minimum_distance = (
        hip_width
        /
        2
    )


    # porto le ginocchia leggermente
    # fuori dal limite delle anche
    safe_distance = (
        minimum_distance
        *
        1.10
    )


    # se la distanza media attuale
    # è già maggiore utilizzo quella,
    # altrimenti utilizzo quella sicura
    corrected_distance = max(
        average_distance,
        safe_distance
    )


    # capisco da quale lato
    # si trova ogni gamba
    if left_hip.x < hip_center.x:

        left_sign = -1

    else:

        left_sign = 1


    if right_hip.x < hip_center.x:

        right_sign = -1

    else:

        right_sign = 1


    # porto entrambe le ginocchia
    # alla stessa distanza dal centro
    # del bacino
    left_knee.x = (
        hip_center.x
        +
        left_sign
        *
        corrected_distance
    )


    right_knee.x = (
        hip_center.x
        +
        right_sign
        *
        corrected_distance
    )



# ----------------------------------------------------
# HEELS ASYMMETRY
# ----------------------------------------------------

# corregge la posizione asimmetrica
# dei piedi rispetto al centro del bacino
def _correct_heel_asymmetry(body):

    hip_center = body.virtual_points.centro_anche

    left_hip = body.landmarks.anca_sinistra
    right_hip = body.landmarks.anca_destra

    left_heel = body.landmarks.tallone_sinistro
    right_heel = body.landmarks.tallone_destro


    # distanza dei due talloni
    # rispetto al centro del bacino
    left_distance = abs(
        left_heel.x
        -
        hip_center.x
    )

    right_distance = abs(
        right_heel.x
        -
        hip_center.x
    )


    # porto i due piedi
    # alla stessa distanza
    corrected_distance = (
        left_distance
        +
        right_distance
    ) / 2


    # determino da quale lato
    # deve stare ogni piede
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
        left_sign
        *
        corrected_distance
    )


    corrected_right_x = (
        hip_center.x
        +
        right_sign
        *
        corrected_distance
    )


    # calcolo quanto deve spostarsi
    # ciascun piede
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
    # per non deformarlo
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

# se viene rilevato un sollevamento
# del tallone lo riporto
# all'altezza iniziale di riferimento
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


    # y cresce andando verso il basso
    # quindi se il valore è minore
    # il tallone è più alto

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

# corregge l'inclinazione eccessiva
# del torso riportando la sua
# lunghezza apparente dentro la soglia
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
        shoulder_center.x
        -
        hip_center.x
    )

    delta_y = (
        shoulder_center.y
        -
        hip_center.y
    )


    current_torso_length = math.hypot(
        delta_x,
        delta_y
    )


    if current_torso_length == 0:
        return


    # la soglia dell'errore è 50 gradi
    # porto il torso a 45 gradi
    # in modo da stare dentro
    # il range corretto
    corrected_inclination = (
        TORSO_INCLINATION_THRESHOLD
        -
        5
    )


    # calcolo la lunghezza apparente
    # che deve avere il torso
    # per trovarsi dentro la soglia
    corrected_torso_length = (
        reference_torso_length
        *
        math.cos(
            math.radians(
                corrected_inclination
            )
        )
    )


    # se per qualche motivo
    # il torso è già abbastanza lungo
    # non devo modificarlo
    if (
        current_torso_length
        >=
        corrected_torso_length
    ):

        return


    # mantengo la stessa direzione
    # del torso ma aumento
    # la sua lunghezza apparente
    correction_ratio = (
        corrected_torso_length
        /
        current_torso_length
    )


    corrected_shoulder_center_x = (
        hip_center.x
        +
        delta_x
        *
        correction_ratio
    )


    corrected_shoulder_center_y = (
        hip_center.y
        +
        delta_y
        *
        correction_ratio
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


    # sposto tutta la parte superiore
    # del corpo insieme
    # per non deformare lo scheletro
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

# crea una copia del bodymodel originale
# e corregge solamente gli errori
# validati per quel frame
def create_corrected_body(
    body,
    detected_errors,
    error_manager
):

    # creo una copia del corpo
    # per non modificare
    # il bodymodel originale
    corrected_body = deepcopy(
        body
    )


    # correggo prima il torso
    # perché sposta anche
    # spalle, braccia e polsi
    if detected_errors.get(
        "torso_inclination",
        False
    ):

        _correct_torso_inclination(
            corrected_body,
            error_manager.reference_torso_length
        )


    # correggo convergenza
    # e asimmetria delle ginocchia
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


    # correggo l'asimmetria
    # della posizione dei piedi
    if detected_errors.get(
        "heel_asymmetry",
        False
    ):

        _correct_heel_asymmetry(
            corrected_body
        )


    # correggo il sollevamento
    # dei talloni
    if detected_errors.get(
        "heel_lift",
        False
    ):

        _correct_heel_lift(
            corrected_body,
            error_manager.reference_left_heel_y,
            error_manager.reference_right_heel_y
        )


    # correggo la posizione
    # verticale del bilanciere
    if detected_errors.get(
        "barbell_position",
        False
    ):

        _correct_barbell_position(
            corrected_body,
            error_manager.reference_body_height
        )


    # correggo per ultimo
    # l'inclinazione del bilanciere
    # portando i polsi
    # alla stessa altezza
    if detected_errors.get(
        "barbell_tilt",
        False
    ):

        _correct_barbell_tilt(
            corrected_body
        )


    # aggiorno i punti virtuali
    # dopo tutte le modifiche
    _update_virtual_points(
        corrected_body
    )


    return corrected_body