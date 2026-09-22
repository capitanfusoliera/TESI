#si occupa di rilevare i landmarker corporei
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


LANDMARK_NAMES = {
     0: "naso",

    11: "spalla_sinistra",
    12: "spalla_destra",

    13: "gomito_sinistro",
    14: "gomito_destro",

    15: "polso_sinistro",
    16: "polso_destro",

    23: "anca_sinistra",
    24: "anca_destra",

    25: "ginocchio_sinistro",
    26: "ginocchio_destro",

    27: "caviglia_sinistra",
    28: "caviglia_destra",

    29: "tallone_sinistro",
    30: "tallone_destro",

    31: "avampiede_sinistro",
    32: "avampiede_destro"
}


def detect_pose(image_path, model_path):

    # Configurazione base del modello
    base_options = python.BaseOptions(
        model_asset_path=model_path
    )

    # Configurazione del Pose Landmarker per una foto con una singola persona
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1
    )

    # Caricamento immagine
    image = mp.Image.create_from_file(image_path)

    width = image.width
    height = image.height

    # Esecuzione del modello
    with vision.PoseLandmarker.create_from_options(options) as landmarker:

        result = landmarker.detect(image)

    # Se non viene rilevata nessuna posa
    if not result.pose_landmarks:
        return []

    # rileva la prima persona
    pose_landmarks = result.pose_landmarks[0]

    landmarks = {}

    # Leggiamo tutti i 33 landmark
    for landmark_id, name in LANDMARK_NAMES.items():

        landmark = pose_landmarks[landmark_id]

        x_pixel = landmark.x * width
        y_pixel = landmark.y * height

        landmarks[name] = {
            "x": x_pixel,
            "y": y_pixel,
            "visibility": landmark.visibility
        }

    return landmarks
