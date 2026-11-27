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
from src.ml.squat_classifier import SquatClassifier

# Indirizzo modello mediaPipe
MODEL_PATH = "models/pose_landmarker_full.task"

# Indirizzo modello machine learning
ML_MODEL_PATH = "models/squat_classifier.pkl"

def analyze_video(video_path):

    #tramite mediaPipe analizzo il video passato per parametro
    mediapipe_results = detect_pose_video(
        video_path=video_path,
        model_path=MODEL_PATH,
        target_fps=60
    )


    # creo la cartella degli output se non esiste
    os.makedirs(
        "outputs",
        exist_ok=True
    )


    # prendo il nome del video
    video_name = os.path.splitext(
        os.path.basename(
            video_path
        )
    )[0]


    # percorso video scheletri
    skeleton_output_path = (
        f"outputs/{video_name}_skeleton.mp4"
    )

    corrected_skeleton_output_path = (
        f"outputs/{video_name}_corrected_skeleton.mp4"
    )


    # controllo se sto esaminando
    # lo stesso video
    try:

        # se esiste rimuovo
        # il vecchio output
        if os.path.exists(
            skeleton_output_path
        ):

            os.remove(
                skeleton_output_path
            )


        if os.path.exists(
            corrected_skeleton_output_path
        ):

            os.remove(
                corrected_skeleton_output_path
            )


    except PermissionError:

        print(
            "ERRORE: uno dei video di output è aperto"
        )

        return


    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )


    # creo il video dello scheletro
    # per l'output
    skeleton_video_writer = cv2.VideoWriter(
        skeleton_output_path,
        fourcc,
        mediapipe_results[
            "analysis_fps"
        ],
        (
            mediapipe_results[
                "width"
            ],
            mediapipe_results[
                "height"
            ]
        )
    )


    # creo il video dello scheletro
    # corretto per l'output
    corrected_skeleton_video_writer = (
        cv2.VideoWriter(
            corrected_skeleton_output_path,
            fourcc,
            mediapipe_results[
                "analysis_fps"
            ],
            (
                mediapipe_results[
                    "width"
                ],
                mediapipe_results[
                    "height"
                ]
            )
        )
    )


    if (
        not skeleton_video_writer.isOpened()
        or
        not corrected_skeleton_video_writer.isOpened()
    ):

        print(
            "ERRORE: impossibile creare gli output"
        )

        skeleton_video_writer.release()

        corrected_skeleton_video_writer.release()

        return


    phase_detector = SquatPhaseDetector()


    error_manager = ErrorManager()


    squat_analyzer = SquatAnalyzer(
        phase_detector=phase_detector,
        error_manager=error_manager
    )


    # salvo i bodymodel
    # per creare successivamente
    # lo scheletro corretto
    bodies_by_frame = {}


    #per ogni frame analizzato da mediapipe
    for frame_data in mediapipe_results[
        "frames"
    ]:

        # prendo frame e punti corporei
        frame_number = frame_data[
            "frame"
        ]

        landmarks = frame_data[
            "landmarks"
        ]


        # se MediaPipe non ha trovato
        # il corpo creo un frame nero
        if not landmarks:

            black_frame = np.zeros(
                (
                    mediapipe_results[
                        "height"
                    ],
                    mediapipe_results[
                        "width"
                    ],
                    3
                ),
                dtype=np.uint8
            )


            skeleton_video_writer.write(
                black_frame
            )

            continue


        #creo il bodymodel
        body = build_body_model(
            mediapipe_landmarks=landmarks,
            frame_number=frame_number,
            time_ms=frame_data[
                "time_ms"
            ]
        )


        #inizio l'analisi
        detected_errors = squat_analyzer.update(
            body
        )


        # salvo il bodymodel
        # per creare successivamente
        # lo scheletro corretto
        bodies_by_frame[
            frame_number
        ] = body


        #creo il video dello scheletro
        skeleton_frame = create_skeleton_frame(
            body,
            mediapipe_results[
                "width"
            ],
            mediapipe_results[
                "height"
            ],
            detected_errors
        )


        skeleton_video_writer.write(
            skeleton_frame
        )


    # metto questa chiusura fuori
    # per situazioni in cui
    # il video è terminato senza
    # che l'utente sia in standing
    squat_analyzer.finish_analysis()


    # creo il classificatore
    # machine learning
    ml_classifier = SquatClassifier(
        ML_MODEL_PATH
    )


    print()

    print(
        "OUTPUT OTTENUTO DAL MACHINE LEARNING"
    )

    print()


    # classifico tutte le ripetizioni
    # completate durante il video
    for repetition in squat_analyzer.completed_reps:

        prediction = ml_classifier.predict(
            repetition
        )


        if prediction is None:

            print(
                f"Ripetizione "
                f"{repetition.number + 1}: "
                f"Impossibile classificare"
            )

            continue


        # trasformo la classificazione
        # in un messaggio leggibile
        if prediction == "correct":

            classification_text = (
                "Squat generalmente corretto"
            )


        elif prediction == "wrong":

            classification_text = (
                "Squat scorretto"
            )


        else:

            classification_text = (
                "Classificazione sconosciuta"
            )


        print(
            f"Ripetizione "
            f"{repetition.number + 1}: "
            f"{classification_text}"
        )


    print("")


    # creo lo scheletro corretto
    # dopo aver validato
    # tutti gli errori
    for frame_data in mediapipe_results[
        "frames"
    ]:

        frame_number = frame_data[
            "frame"
        ]


        # se mediapipe non ha trovato
        # il corpo creo un frame nero
        if frame_number not in bodies_by_frame:

            black_frame = np.zeros(
                (
                    mediapipe_results[
                        "height"
                    ],
                    mediapipe_results[
                        "width"
                    ],
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


        # creo il frame
        # dello scheletro corretto
        corrected_skeleton_frame = (
            create_skeleton_frame(
                corrected_body,
                mediapipe_results[
                    "width"
                ],
                mediapipe_results[
                    "height"
                ],
                {}
            )
        )


        corrected_skeleton_video_writer.write(
            corrected_skeleton_frame
        )


    skeleton_video_writer.release()

    corrected_skeleton_video_writer.release()


    print()

    print(
        f"Video skeleton salvato in: "
        f"{skeleton_output_path}"
    )

    print(
        f"Video skeleton corretto salvato in: "
        f"{corrected_skeleton_output_path}"
    )


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