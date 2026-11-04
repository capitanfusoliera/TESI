import cv2
import numpy as np


# colori in BGR
BACKGROUND_COLOR = (0, 0, 0)
BONE_COLOR = (255, 255, 255)
JOINT_COLOR = (0, 255, 0)
ERROR_COLOR = (0, 0, 255)


def _get_point(landmarks, point_name):

    if not hasattr(landmarks, point_name):
        return None

    point = getattr(landmarks, point_name)

    return (
        int(point.x),
        int(point.y)
    )


def _draw_point(
    frame,
    landmarks,
    point_name,
    color,
    radius=6
):

    point = _get_point(
        landmarks,
        point_name
    )

    if point is None:
        return

    cv2.circle(
        frame,
        point,
        radius,
        color,
        -1
    )


def _draw_line(
    frame,
    landmarks,
    point_a,
    point_b,
    color,
    thickness=3
):

    a = _get_point(
        landmarks,
        point_a
    )

    b = _get_point(
        landmarks,
        point_b
    )

    if a is None or b is None:
        return

    cv2.line(
        frame,
        a,
        b,
        color,
        thickness
    )


def _get_midpoint(
    landmarks,
    point_a,
    point_b
):

    a = _get_point(
        landmarks,
        point_a
    )

    b = _get_point(
        landmarks,
        point_b
    )

    if a is None or b is None:
        return None

    return (
        int((a[0] + b[0]) / 2),
        int((a[1] + b[1]) / 2)
    )


def _draw_trunk_center_line(
    frame,
    landmarks,
    color,
    thickness=3
):

    shoulder_center = _get_midpoint(
        landmarks,
        "spalla_sinistra",
        "spalla_destra"
    )

    hip_center = _get_midpoint(
        landmarks,
        "anca_sinistra",
        "anca_destra"
    )

    if shoulder_center is None or hip_center is None:
        return

    cv2.line(
        frame,
        shoulder_center,
        hip_center,
        color,
        thickness
    )


def _draw_base_skeleton(
    frame,
    landmarks
):

    base_lines = [

        # testa / spalle
        ("spalla_sinistra", "spalla_destra"),

        # braccio sinistro
        ("spalla_sinistra", "gomito_sinistro"),
        ("gomito_sinistro", "polso_sinistro"),

        # braccio destro
        ("spalla_destra", "gomito_destro"),
        ("gomito_destro", "polso_destro"),

        # tronco
        ("spalla_sinistra", "anca_sinistra"),
        ("spalla_destra", "anca_destra"),
        ("anca_sinistra", "anca_destra"),

        # gamba sinistra
        ("anca_sinistra", "ginocchio_sinistro"),
        ("ginocchio_sinistro", "caviglia_sinistra"),

        # gamba destra
        ("anca_destra", "ginocchio_destro"),
        ("ginocchio_destro", "caviglia_destra"),

        # piede sinistro
        ("caviglia_sinistra", "tallone_sinistro"),
        ("tallone_sinistro", "avampiede_sinistro"),

        # piede destro
        ("caviglia_destra", "tallone_destro"),
        ("tallone_destro", "avampiede_destro")
    ]

    for point_a, point_b in base_lines:

        _draw_line(
            frame,
            landmarks,
            point_a,
            point_b,
            BONE_COLOR,
            3
        )

    _draw_trunk_center_line(
        frame,
        landmarks,
        BONE_COLOR,
        3
    )

    base_points = [
        "naso",

        "spalla_sinistra",
        "spalla_destra",

        "gomito_sinistro",
        "gomito_destro",

        "polso_sinistro",
        "polso_destro",

        "anca_sinistra",
        "anca_destra",

        "ginocchio_sinistro",
        "ginocchio_destro",

        "caviglia_sinistra",
        "caviglia_destra",

        "tallone_sinistro",
        "tallone_destro",

        "avampiede_sinistro",
        "avampiede_destro"
    ]

    for point_name in base_points:

        _draw_point(
            frame,
            landmarks,
            point_name,
            JOINT_COLOR,
            5
        )


def _highlight_barbell_position(
    frame,
    landmarks
):

    # polsi rossi
    _draw_point(
        frame,
        landmarks,
        "polso_sinistro",
        ERROR_COLOR,
        8
    )

    _draw_point(
        frame,
        landmarks,
        "polso_destro",
        ERROR_COLOR,
        8
    )

    # linea tra i polsi
    _draw_line(
        frame,
        landmarks,
        "polso_sinistro",
        "polso_destro",
        ERROR_COLOR,
        4
    )

    # riga alta del torso = linea spalle
    _draw_line(
        frame,
        landmarks,
        "spalla_sinistra",
        "spalla_destra",
        ERROR_COLOR,
        4
    )


def _highlight_barbell_tilt(
    frame,
    landmarks
):

    _draw_point(
        frame,
        landmarks,
        "polso_sinistro",
        ERROR_COLOR,
        8
    )

    _draw_point(
        frame,
        landmarks,
        "polso_destro",
        ERROR_COLOR,
        8
    )

    _draw_line(
        frame,
        landmarks,
        "polso_sinistro",
        "polso_destro",
        ERROR_COLOR,
        4
    )


def _highlight_feet(
    frame,
    landmarks
):

    # sinistro
    _draw_point(
        frame,
        landmarks,
        "caviglia_sinistra",
        ERROR_COLOR,
        8
    )

    _draw_point(
        frame,
        landmarks,
        "tallone_sinistro",
        ERROR_COLOR,
        8
    )

    _draw_point(
        frame,
        landmarks,
        "avampiede_sinistro",
        ERROR_COLOR,
        8
    )

    _draw_line(
        frame,
        landmarks,
        "caviglia_sinistra",
        "tallone_sinistro",
        ERROR_COLOR,
        4
    )

    _draw_line(
        frame,
        landmarks,
        "tallone_sinistro",
        "avampiede_sinistro",
        ERROR_COLOR,
        4
    )

    # destro
    _draw_point(
        frame,
        landmarks,
        "caviglia_destra",
        ERROR_COLOR,
        8
    )

    _draw_point(
        frame,
        landmarks,
        "tallone_destro",
        ERROR_COLOR,
        8
    )

    _draw_point(
        frame,
        landmarks,
        "avampiede_destro",
        ERROR_COLOR,
        8
    )

    _draw_line(
        frame,
        landmarks,
        "caviglia_destra",
        "tallone_destro",
        ERROR_COLOR,
        4
    )

    _draw_line(
        frame,
        landmarks,
        "tallone_destro",
        "avampiede_destro",
        ERROR_COLOR,
        4
    )


def _highlight_knees_and_legs(
    frame,
    landmarks
):

    # punti
    for point_name in [
        "anca_sinistra",
        "anca_destra",
        "ginocchio_sinistro",
        "ginocchio_destro",
        "caviglia_sinistra",
        "caviglia_destra"
    ]:

        _draw_point(
            frame,
            landmarks,
            point_name,
            ERROR_COLOR,
            8
        )

    # cosce
    _draw_line(
        frame,
        landmarks,
        "anca_sinistra",
        "ginocchio_sinistro",
        ERROR_COLOR,
        4
    )

    _draw_line(
        frame,
        landmarks,
        "anca_destra",
        "ginocchio_destro",
        ERROR_COLOR,
        4
    )

    # tibie
    _draw_line(
        frame,
        landmarks,
        "ginocchio_sinistro",
        "caviglia_sinistra",
        ERROR_COLOR,
        4
    )

    _draw_line(
        frame,
        landmarks,
        "ginocchio_destro",
        "caviglia_destra",
        ERROR_COLOR,
        4
    )


def _highlight_torso(
    frame,
    landmarks
):

    torso_points = [
        "spalla_sinistra",
        "spalla_destra",
        "anca_sinistra",
        "anca_destra"
    ]

    for point_name in torso_points:

        _draw_point(
            frame,
            landmarks,
            point_name,
            ERROR_COLOR,
            8
        )

    _draw_line(
        frame,
        landmarks,
        "spalla_sinistra",
        "spalla_destra",
        ERROR_COLOR,
        4
    )

    _draw_line(
        frame,
        landmarks,
        "anca_sinistra",
        "anca_destra",
        ERROR_COLOR,
        4
    )

    _draw_line(
        frame,
        landmarks,
        "spalla_sinistra",
        "anca_sinistra",
        ERROR_COLOR,
        4
    )

    _draw_line(
        frame,
        landmarks,
        "spalla_destra",
        "anca_destra",
        ERROR_COLOR,
        4
    )

    # riga della spinta / linea centrale del tronco
    _draw_trunk_center_line(
        frame,
        landmarks,
        ERROR_COLOR,
        4
    )


def create_skeleton_frame(
    body,
    width,
    height,
    detected_errors=None
):

    frame = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    landmarks = body.landmarks

    if detected_errors is None:
        detected_errors = {}

    # ----------------------------------------------------
    # SKELETON BASE
    # ----------------------------------------------------

    _draw_base_skeleton(
        frame,
        landmarks
    )

    # ----------------------------------------------------
    # ERROR HIGHLIGHTS
    # ----------------------------------------------------

    if detected_errors.get("barbell_position", False):

        _highlight_barbell_position(
            frame,
            landmarks
        )

    if detected_errors.get("barbell_tilt", False):

        _highlight_barbell_tilt(
            frame,
            landmarks
        )

    if (
        detected_errors.get("heel_asymmetry", False)
        or
        detected_errors.get("heels_asymmetry", False)
    ):

        _highlight_feet(
            frame,
            landmarks
        )

    if detected_errors.get("heel_lift", False):

        _highlight_feet(
            frame,
            landmarks
        )

    if (
        detected_errors.get("knee_convergence", False)
        or
        detected_errors.get("knee_asymmetry", False)
        or
        detected_errors.get("knees_asymmetry", False)
    ):

        _highlight_knees_and_legs(
            frame,
            landmarks
        )

    if (
        detected_errors.get("torso_inclinato", False)
        or
        detected_errors.get("torso_inclination", False)
        or
        detected_errors.get("torso_lean", False)
        or
        detected_errors.get("inclinazione_tronco", False)
    ):

        _highlight_torso(
            frame,
            landmarks
        )

    return frame