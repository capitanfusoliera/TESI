import argparse
import cv2
import os

from src.output.scheleton_ouput import create_skeleton_frame

from src.mediapipe.mp_video_detector import detect_pose_video
from src.body.body_models import build_body_model

from src.squat.squat_phase import SquatPhaseDetector
from src.squat.squat_analyzer import SquatAnalyzer

from src.errors.error_manager import ErrorManager
from src.validation.marker_video_detector import detect_markers_video
from src.validation.video_comparison import create_validation_outputs


# Modello MediaPipe utilizzato per la pose detection
MODEL_PATH = "models/pose_landmarker_full.task"


def analyze_video(video_path):

    # ----------------------------------------------------
    # MEDIAPIPE
    # ----------------------------------------------------

    mediapipe_results = detect_pose_video(
        video_path=video_path,
        model_path=MODEL_PATH,
        target_fps=60
    )


    # ----------------------------------------------------
    # VIDEO DI CONTROLLO
    # ----------------------------------------------------

    # creo la cartella degli output se non esiste
    os.makedirs("outputs", exist_ok=True)

    # apro il video originale
    cap = cv2.VideoCapture(video_path)

    # prendo il nome del video senza estensione
    video_name = os.path.splitext(
        os.path.basename(video_path)
    )[0]


    # percorso video normale
    output_path = (
        f"outputs/{video_name}_analysis_control.mp4"
    )


    # percorso video skeleton
    skeleton_output_path = (
        f"outputs/{video_name}_skeleton.mp4"
    )


    # se sto rianalizzando lo stesso video
    # elimino il vecchio output
    if os.path.exists(output_path):

        try:
            os.remove(output_path)

        except PermissionError:

            print(
                "ERRORE: il video di output è aperto. "
                "Chiudilo e riprova."
            )

            return


    if os.path.exists(skeleton_output_path):

        try:
            os.remove(skeleton_output_path)

        except PermissionError:

            print(
                "ERRORE: il video skeleton è aperto. "
                "Chiudilo e riprova."
            )

            return


    fourcc = cv2.VideoWriter_fourcc(*"mp4v")


    # creo il video normale
    video_writer = cv2.VideoWriter(
        output_path,
        fourcc,
        mediapipe_results["analysis_fps"],
        (
            mediapipe_results["width"],
            mediapipe_results["height"]
        )
    )


    # creo il video skeleton
    skeleton_video_writer = cv2.VideoWriter(
        skeleton_output_path,
        fourcc,
        mediapipe_results["analysis_fps"],
        (
            mediapipe_results["width"],
            mediapipe_results["height"]
        )
    )


    if not video_writer.isOpened():

        print(
            "ERRORE: impossibile creare "
            "il video di output."
        )

        cap.release()

        return


    if not skeleton_video_writer.isOpened():

        print(
            "ERRORE: impossibile creare "
            "il video skeleton."
        )

        cap.release()
        video_writer.release()

        return


    # ----------------------------------------------------
    # CREATE ANALYSIS MODULES
    # ----------------------------------------------------

    phase_detector = SquatPhaseDetector()

    error_manager = ErrorManager()

    squat_analyzer = SquatAnalyzer(
        phase_detector=phase_detector,
        error_manager=error_manager
    )


    # ----------------------------------------------------
    # ANALYZE EVERY MEDIAPIPE FRAME
    # ----------------------------------------------------

    for frame_data in mediapipe_results["frames"]:

        frame_number = frame_data["frame"]

        landmarks = frame_data["landmarks"]


        # vado esattamente al frame analizzato da mediapipe
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

        ret, frame = cap.read()

        if not ret:
            continue


        # se MediaPipe non ha trovato il corpo
        # salvo comunque il frame normale
        # e un frame nero nello skeleton
        if not landmarks:

            video_writer.write(
                frame
            )

            skeleton_video_writer.write(
                frame * 0
            )

            continue


        # ----------------------------------------------------
        # BODY MODEL
        # ----------------------------------------------------

        body = build_body_model(
            mediapipe_landmarks=landmarks,
            frame_number=frame_number,
            time_ms=frame_data["time_ms"]
        )


        # passo il corpo allo SquatAnalyzer
        squat_analyzer.update(
            body
        )


        # ----------------------------------------------------
        # VIDEO SKELETON
        # ----------------------------------------------------

        skeleton_frame = create_skeleton_frame(
            body,
            mediapipe_results["width"],
            mediapipe_results["height"]
        )


        skeleton_video_writer.write(
            skeleton_frame
        )


        # ----------------------------------------------------
        # VIDEO DI CONTROLLO NORMALE
        # ----------------------------------------------------

        # disegno i landmark
        for landmark_name, landmark in landmarks.items():

            x = int(landmark["x"])
            y = int(landmark["y"])

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )


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


        video_writer.write(
            frame
        )


    # ----------------------------------------------------
    # FINE ANALISI
    # ----------------------------------------------------

    # deve stare fuori dal for
    squat_analyzer.finish_analysis()


    # ----------------------------------------------------
    # CHIUSURA VIDEO
    # ----------------------------------------------------

    cap.release()

    video_writer.release()

    skeleton_video_writer.release()


    print()
    print(
        f"Video di controllo salvato in: "
        f"{output_path}"
    )

    print(
        f"Video skeleton salvato in: "
        f"{skeleton_output_path}"
    )


    # ----------------------------------------------------
    # PRINT RESULTS
    # ----------------------------------------------------

    print()
    print("===== RISULTATI ANALISI =====")

    for rep in squat_analyzer.completed_reps:

        print()

        print(
            f"Ripetizione {rep.number}"
        )

        print(
            f"Frame: "
            f"{rep.start_frame} -> "
            f"{rep.end_frame}"
        )

        print(
            f"Errori: {rep.errors}"
        )

def validation_video(video_path):

    marker_results = detect_markers_video(
        video_path,
        target_fps=30
    )

    mediapipe_results = detect_pose_video(
        video_path=video_path,
        model_path=MODEL_PATH,
        target_fps=30
    )

    create_validation_outputs(
        marker_results,
        mediapipe_results
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mode",
        choices=[
            "analyze",
            "validation"
        ]
    )

    parser.add_argument(
        "--video",
        required=True
    )

    args = parser.parse_args()


    # ANALISI NORMALE DELLO SQUAT
    if args.mode == "analyze":

        analyze_video(
            args.video
        )


    # VALIDAZIONE MEDIAPIPE CON MARKER
    elif args.mode == "validation":

        validation_video(
            args.video
        )


# EQUIVALENTE DEL MAIN DEL C
if __name__ == "__main__":

    main()