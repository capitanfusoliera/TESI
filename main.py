import mediapipe as mp 
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision #funzionalità per computier vision

MODEL_PATH = "models/pose_landmarker_full.task"  #percorso modello
IMAGE_PATH = "images/test_1.jpeg"    #percorso immagine

#configurazione del modello BaseOption 
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
) 

#configurazione del modello PoseLandmarkerOptions con base e immagine
options = vision.PoseLandmarkerOptions(
    base_options = base_options,
    running_mode = vision.RunningMode.IMAGE
)

# apertura e trasformazione dell'immagine in un oggetto mediapipe
image = mp.Image.create_from_file(IMAGE_PATH)

# creazione di un landmarker in questo blocco 
with vision.PoseLandmarker.create_from_options(options) as landmarker:

    result = landmarker.detect(image)

# se sono stati creati dei landmarker
if result.pose_landmarks:
    print("persona rilevata")
    landmarks = result.pose_landmarks[0]
    print("Numero landmark:", len(landmarks)) #numero landmarker trovati

else:

    print("Nessuna persona rilevata")