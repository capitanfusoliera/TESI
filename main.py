from src.mp_detector import detect_pose


MODEL_PATH = "models/pose_landmarker_full.task"
IMAGE_PATH = "images/test_1.jpeg"


# Analizziamo l'immagine con MediaPipe
landmarks = detect_pose(
    IMAGE_PATH,
    MODEL_PATH
)


# Controlliamo se MediaPipe ha trovato una posa
if not landmarks:

    print("Nessuna posa rilevata")

else:

    print("Posa rilevata!\n")

    # Stampiamo i landmark rilevati
    for nome, dati in landmarks.items():

        print(
            f"{nome}: "
            f"X = {dati['x']:.1f} px, "
            f"Y = {dati['y']:.1f} px, "
            f"visibility = {dati['visibility']:.3f}"
        )