# crea il file json prendendo ogni frame
# e per ogni frame i 17 landmark corporei


import json
import os


# landmark che salvo per ogni frame
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


# trasforma un bodymodel
# in dati salvabili dentro il json
def body_to_data(body):

    landmarks_data = {}


    for landmark_name in LANDMARK_NAMES:

        landmark = getattr(
            body.landmarks,
            landmark_name
        )


        landmarks_data[
            landmark_name
        ] = {

            "x": float(landmark.x),

            "y": float(landmark.y)

        }


    return {

        "frame": int(body.frame_number),

        "time_ms": float(body.time_ms),

        "landmarks": landmarks_data

    }



# salva tutte le ripetizioni
# di un video dentro un unico file json
def save_video_dataset(
    video_path,
    repetitions
):

    # creo la cartella dataset
    # se non esiste
    os.makedirs(
        "dataset",
        exist_ok=True
    )


    # recupero il nome del video
    # senza estensione
    video_name = os.path.splitext(
        os.path.basename(video_path)
    )[0]


    # lista che conterrà
    # tutte le ripetizioni del video
    repetitions_data = []


    for repetition in repetitions:

        frames_data = []


        # salvo tutti i frame
        # appartenenti alla ripetizione
        for body in repetition.frames:

            frames_data.append(
                body_to_data(body)
            )


        # creo i dati relativi
        # alla singola ripetizione
        repetition_data = {

            "repetition": repetition.number,

            "start_frame": repetition.start_frame,

            "end_frame": repetition.end_frame,

            # questo valore viene compilato
            # manualmente successivamente
            "label": None,

            "frames": frames_data

        }


        # aggiungo la ripetizione
        # alla lista del video
        repetitions_data.append(
            repetition_data
        )


    # creo il contenuto finale
    # del file json
    video_data = {

        "video": os.path.basename(
            video_path
        ),

        "number_of_repetitions": len(
            repetitions_data
        ),

        "repetitions": repetitions_data

    }


    # creo il percorso
    # del file json
    output_path = (
        f"dataset/{video_name}.json"
    )


    # salvo il json
    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            video_data,
            file,
            indent=4
        )


    print()
    print(
        f"Dataset salvato in: {output_path}"
    )

    print(
        f"Ripetizioni salvate: {len(repetitions_data)}"
    )