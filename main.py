import argparse
import cv2
import os
import numpy as np
from src.output.skeleton_correction import create_corrected_body
from src.output.scheleton_ouput import create_skeleton_frame
from src.mediapipe.mp_video_detector import detect_pose_video
from src.body.body_models import build_body_model
from src.squat.squat_phase import SquatPhaseDetector
from src.squat.squat_analyzer import SquatAnalyzer
from src.errors.error_manager import ErrorManager
from src.validation.marker_video_detector import detect_markers_video
from src.validation.video_comparison import create_validation_outputs
from src.ml.dataset_builder import save_video_dataset

# Indirizzo modello mediaPipe
MODEL_PATH = "models/pose_landmarker_full.task"


def analyze_video(video_path):

    #tramite mediaPipe analizzo il video passato per parametro
    mediapipe_results = detect_pose_video(video_path=video_path, model_path=MODEL_PATH, target_fps=60)

    # creo la cartella degli output se non esiste
    os.makedirs("outputs", exist_ok=True)

    # apro il video originale prendo il nome e gli do il path
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = (f"outputs/{video_name}_analysis_control.mp4")

    # percorso video scheltri
    skeleton_output_path = (f"outputs/{video_name}_skeleton.mp4")
    corrected_skeleton_output_path = (f"outputs/{video_name}_corrected_skeleton.mp4")

    #controllo se sto esaminando lo stesso video
    try:

        # se esiste rimuovo il vecchio output
        if os.path.exists(output_path):
            os.remove(output_path)

        if os.path.exists(skeleton_output_path):
            os.remove(skeleton_output_path)

        if os.path.exists(corrected_skeleton_output_path):
            os.remove(corrected_skeleton_output_path)

    except PermissionError:
        print("ERRORE: uno dei video di output è aperto")
        return


    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # creo il video video per l'uptut
    video_writer = cv2.VideoWriter(output_path, fourcc,
        mediapipe_results["analysis_fps"],
        (mediapipe_results["width"], mediapipe_results["height"])
    )


    # creo il video dello scheletro per l'uotput
    skeleton_video_writer = cv2.VideoWriter(skeleton_output_path, fourcc,mediapipe_results["analysis_fps"],
        (mediapipe_results["width"], mediapipe_results["height"]))

    # creo il video dello scheletro corretto per l'output
    corrected_skeleton_video_writer = cv2.VideoWriter(corrected_skeleton_output_path,fourcc,mediapipe_results["analysis_fps"],
        (mediapipe_results["width"],mediapipe_results["height"]))


    if (not video_writer.isOpened() or not skeleton_video_writer.isOpened() or not corrected_skeleton_video_writer.isOpened()):
        print("ERRORE: impossibile creare gli output")
        cap.release()
        video_writer.release()
        skeleton_video_writer.release()
        corrected_skeleton_video_writer.release()
        return

    phase_detector = SquatPhaseDetector()

    error_manager = ErrorManager()

    squat_analyzer = SquatAnalyzer(
        phase_detector=phase_detector,
        error_manager=error_manager
    )

    # salvo i bodymodel per creare successivamente
    # lo scheletro corretto
    bodies_by_frame = {}

    #per ogni frame analizzato da mediapipe
    for frame_data in mediapipe_results["frames"]:

        # prendo frame e punti corporei
        frame_number = frame_data["frame"]
        landmarks = frame_data["landmarks"]

        # controllo che mediapipe sia riuscito a leggere questo frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()

        if not ret:
            continue


        # se MediaPipe non ha trovato il corpo
        # salvo comunque il frame normale
        # e un frame nero nello skeleton
        if not landmarks:

            video_writer.write(frame)

            skeleton_video_writer.write(frame * 0)

            continue

        #creo il bodymodel
        body = build_body_model(mediapipe_landmarks=landmarks, frame_number=frame_number, time_ms=frame_data["time_ms"])

        #inizio l'analisi
        detected_errors = squat_analyzer.update(body)

        # salvo il bodymodel per creare successivamente
        # lo scheletro corretto
        bodies_by_frame[frame_number] = body

        #creo il video dello scheletro
        skeleton_frame = create_skeleton_frame(body,mediapipe_results["width"],mediapipe_results["height"],detected_errors)

        skeleton_video_writer.write(skeleton_frame)

        # disegno i landmark
        for landmark_name, landmark in landmarks.items():

            x = int(landmark["x"])
            y = int(landmark["y"])

            #creo un cerchio per ogni landmarker
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)


        #inserisco i dati nel video
        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Fase: {squat_analyzer.current_phase.value}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Ripetizione: {squat_analyzer.rep_count}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        video_writer.write(frame)


    # metto questa chiusura fuori per situazioni in cui
    # il video è termintato senza che l'utente sia in stending
    squat_analyzer.finish_analysis()


    # creo lo scheletro corretto dopo aver
    # validato tutti gli errori

    for frame_data in mediapipe_results["frames"]:

        frame_number = frame_data["frame"]


        # se mediapipe non ha trovato il corpo
        # creo un frame nero

        if frame_number not in bodies_by_frame:

            black_frame = np.zeros(
                (
                    mediapipe_results["height"],
                    mediapipe_results["width"],
                    3
                ),
                dtype=np.uint8
            )

            corrected_skeleton_video_writer.write(
                black_frame
            )

            continue


        body = bodies_by_frame[
            frame_number
        ]


        # recupero solamente gli errori
        # validati per questo frame

        validated_errors = (
            squat_analyzer
            .validated_errors_by_frame
            .get(
                frame_number,
                {}
            )
        )


        # creo il body corretto

        corrected_body = create_corrected_body(
            body,
            validated_errors,
            error_manager
        )


        # creo il frame dello scheletro corretto

        corrected_skeleton_frame = create_skeleton_frame(
            corrected_body,
            mediapipe_results["width"],
            mediapipe_results["height"],
            {}
        )


        corrected_skeleton_video_writer.write(
            corrected_skeleton_frame
        )


    cap.release()
    video_writer.release()
    skeleton_video_writer.release()
    corrected_skeleton_video_writer.release()

    print()
    print(f"Video di controllo salvato in: " f"{output_path}")
    print(f"Video skeleton salvato in: "f"{skeleton_output_path}")
    print(f"Video skeleton corretto salvato in: "f"{corrected_skeleton_output_path}")

    return squat_analyzer



#questo metodo crea il dataset per il machine learning
def create_dataset(video_path):

    #tramite mediaPipe analizzo il video passato per parametro
    mediapipe_results = detect_pose_video(video_path=video_path, model_path=MODEL_PATH, target_fps=60)


    phase_detector = SquatPhaseDetector()

    error_manager = ErrorManager()


    # creo lo squat analyzer solamente
    # per riconoscere le ripetizioni
    # senza effettuare l'analisi degli errori
    squat_analyzer = SquatAnalyzer(
        phase_detector=phase_detector,
        error_manager=error_manager,
        analyze_errors=False
    )


    #per ogni frame analizzato da mediapipe
    for frame_data in mediapipe_results["frames"]:

        frame_number = frame_data["frame"]

        landmarks = frame_data["landmarks"]


        # se mediapipe non trova il corpo
        # passo al frame successivo
        if not landmarks:
            continue


        #creo il bodymodel
        body = build_body_model(
            mediapipe_landmarks=landmarks,
            frame_number=frame_number,
            time_ms=frame_data["time_ms"]
        )


        # passo il bodymodel allo squat analyzer
        # solamente per riconoscere
        # le varie ripetizioni
        squat_analyzer.update(body)


    # chiudo eventuali analisi rimaste aperte
    squat_analyzer.finish_analysis()


    # salvo tutte le ripetizioni trovate
    # dentro il json del video
    save_video_dataset(
        video_path,
        squat_analyzer.completed_reps
    )



#questo è il metodo che si occupa di confrontare i marker con i risultati di mediapipe
def validation_video(video_path):

    marker_results = detect_markers_video(video_path,target_fps=30)

    mediapipe_results = detect_pose_video(video_path=video_path,model_path=MODEL_PATH,target_fps=30)

    create_validation_outputs(marker_results,mediapipe_results)



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("mode",choices=["analyze","validation","dataset"])

    parser.add_argument("--video",required=True)

    args = parser.parse_args()


    # faccio l'analisi dello squt
    if args.mode == "analyze":

        analyze_video(args.video)


    # faccio la validazione dello squat
    elif args.mode == "validation":

        validation_video(args.video)


    # creo il dataset per il machine learning
    elif args.mode == "dataset":

        create_dataset(args.video)



# faccio partire il main
if __name__ == "__main__":
    main()