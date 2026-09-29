from src.mp_image_detector import detect_pose_image
from src.marker_image_detector import detect_markers_image
from src.marker_video_detector import detect_markers_frame

MODEL_PATH = "models/pose_landmarker_full.task"
IMAGE_PATH = "images/test_9.jpeg"
VIDEO_PATH = "videos/marker_squat_1.mp4"

# Analizziamo l'immagine con MediaPipe
landmarks = detect_pose_image(
    IMAGE_PATH,
    MODEL_PATH
)
markers = 0
#markers = detect_markers_image(IMAGE_PATH)

if not markers:

    print("No red markers detected")

else:

    print(f"Detected markers: {len(markers)}\n")

    for marker_id, marker in enumerate(markers):

        print(
            f"Marker {marker_id}: "
            f"X = {marker['x']:.1f} px | "
            f"Y = {marker['y']:.1f} px | "
            f"area = {marker['area']:.1f} px²"
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

video_results = detect_markers_frame(VIDEO_PATH)

print(
    f"\nAnalyzed frames: "
    f"{len(video_results)}"
)