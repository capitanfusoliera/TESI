
#per rendere ogni squat della solita l'unghezza 
#prende un quantitativo di frame fissato da ogni esecuzione
#normalizza le varie proporizioni della persona 
#in questo modo vengono evitati problemi relativi alla distanza

import numpy as np


# numero di frame con cui rappresento
# ogni ripetizione
TARGET_FRAMES = 30


# landmark utilizzati dal machine learning
LANDMARK_NAMES = [
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


# calcolo una misura corporea iniziale
# utilizzata per normalizzare le coordinate
def get_reference_scale(frames):

    first_frame = frames[0]

    landmarks = first_frame["landmarks"]


    left_hip = landmarks["anca_sinistra"]

    right_hip = landmarks["anca_destra"]

    left_heel = landmarks["tallone_sinistro"]

    right_heel = landmarks["tallone_destro"]


    left_distance = np.hypot(
        left_heel["x"] - left_hip["x"],
        left_heel["y"] - left_hip["y"]
    )


    right_distance = np.hypot(
        right_heel["x"] - right_hip["x"],
        right_heel["y"] - right_hip["y"]
    )


    reference_scale = (
        left_distance + right_distance
    ) / 2


    if reference_scale == 0:

        return 1


    return reference_scale



# trasforma un singolo frame
# in un vettore numerico normalizzato
def frame_to_features(
    frame,
    reference_scale
):

    landmarks = frame["landmarks"]


    left_hip = landmarks["anca_sinistra"]

    right_hip = landmarks["anca_destra"]


    # centro del bacino
    hip_center_x = (
        left_hip["x"] + right_hip["x"]
    ) / 2

    hip_center_y = (
        left_hip["y"] + right_hip["y"]
    ) / 2


    frame_features = []


    for landmark_name in LANDMARK_NAMES:

        landmark = landmarks[
            landmark_name
        ]


        # porto le coordinate rispetto
        # al centro del bacino

        normalized_x = (
            landmark["x"] - hip_center_x
        ) / reference_scale

        normalized_y = (
            landmark["y"] - hip_center_y
        ) / reference_scale


        frame_features.append(
            normalized_x
        )

        frame_features.append(
            normalized_y
        )


    return frame_features



# trasforma una ripetizione completa
# nel vettore utilizzato dal machine learning
def extract_repetition_features(
    repetition
):

    frames = repetition["frames"]


    if not frames:

        return None


    reference_scale = get_reference_scale(
        frames
    )


    # tutte le ripetizioni devono avere
    # lo stesso numero di frame

    frame_indexes = np.linspace(
        0,
        len(frames) - 1,
        TARGET_FRAMES
    ).astype(int)


    repetition_features = []


    for frame_index in frame_indexes:

        frame = frames[
            frame_index
        ]


        frame_features = frame_to_features(
            frame,
            reference_scale
        )


        repetition_features.extend(
            frame_features
        )


    return np.array(
        repetition_features,
        dtype=np.float32
    )