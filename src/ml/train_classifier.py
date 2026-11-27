#prende dal dataset tutti i file utilizzabili
#controlla la label e esegue l'algoritmo


import json
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import SMOTE

from src.ml.feature_extractor import extract_repetition_features
from src.ml.feature_extractor import TARGET_FRAMES
from src.ml.feature_extractor import LANDMARK_NAMES


DATASET_PATH = "dataset"

MODEL_PATH = "models/squat_classifier.pkl"


# associa i video appartenenti
# allo stesso partecipante
PARTICIPANT_GROUPS = {

    # persona 1
    "test_video_36.mp4": "participant_1",
    "test_video_sbagliato_3.mp4": "participant_1",
    "test_video_11.mp4": "participant_1",
    "test_corretto_2.mp4": "participant_1",
    "test_corretto_3.mp4": "participant_1",
    "test_corretto_4.mp4": "participant_1",

    # persona 2
    "test_video_sbagliato_4.mp4": "participant_2",
    "test_video_37.mp4": "participant_2",

    # persona 3
    "test_video_sbagliato_2.mp4": "participant_3",
    "test_video_corretto_1.mp4": "participant_3",
    "test_video_30.mp4": "participant_3",

    # persona 4
    "test_video_sbagliato_giusto_1.mp4": "participant_4",
    "test_video_24.mp4": "participant_4",

    # persona 5
    "test_video_sbagliato_1.mp4": "participant_5",
    "test_video_32.mp4": "participant_5",

    # persona 6
    "test_video_60.mp4": "participant_6",
    "test_video_51.mp4": "participant_6",

    # persona 7
    "test_video_64.mp4": "participant_7",
    "test_video_55.mp4": "participant_7",

    # persona 8
    "test_video_27.mp4": "participant_8",
    "test_video_scorretto_6.mp4": "participant_8",

    # persona 9
    "test_video_14.mp4": "participant_9",
    "test_video_scorretto_5.mp4": "participant_9"

}



# carica tutti i json presenti
# dentro la cartella dataset
def load_dataset():

    X = []

    y = []

    groups = []


    for file_name in os.listdir(
        DATASET_PATH
    ):

        if not file_name.endswith(".json"):
            continue


        file_path = os.path.join(
            DATASET_PATH,
            file_name
        )


        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            video_data = json.load(
                file
            )


        video_name = video_data[
            "video"
        ]


        # se il video appartiene
        # ad una persona presente
        # anche in altri video
        # utilizzo lo stesso gruppo
        participant_group = PARTICIPANT_GROUPS.get(
            video_name,
            video_name
        )


        for repetition in video_data[
            "repetitions"
        ]:

            label = repetition.get(
                "label"
            )


            # ignoro le ripetizioni
            # che non sono ancora state etichettate
            if label not in [
                "correct",
                "medium",
                "wrong"
            ]:

                continue


            # considero tutti gli squat medium come correct
            if label == "medium":

                label = "wrong"


            features = extract_repetition_features(
                repetition
            )


            if features is None:
                continue


            X.append(
                features
            )

            y.append(
                label
            )


            # salvo il partecipante
            # di provenienza
            # per separare training e test
            groups.append(
                participant_group
            )


    return (
        np.array(X),
        np.array(y),
        np.array(groups)
    )



def train_classifier():

    X, y, groups = load_dataset()


    print()

    print(
        f"Ripetizioni disponibili: {len(X)}"
    )


    if len(X) == 0:

        print(
            "ERRORE: nessuna ripetizione etichettata"
        )

        return


    correct_count = np.sum(
        y == "correct"
    )

    wrong_count = np.sum(
        y == "wrong"
    )


    print(
        f"Correct: {correct_count}"
    )

    print(
        f"Wrong: {wrong_count}"
    )


    # devo avere entrambe le classi
    if len(np.unique(y)) < 2:

        print(
            "ERRORE: servono squat correct "
            "e wrong"
        )

        return


    unique_participants = np.unique(
        groups
    )


    print()

    print(
        f"Partecipanti/gruppi disponibili: "
        f"{len(unique_participants)}"
    )


    # se ho abbastanza partecipanti
    # creo training set e test set
    if len(unique_participants) >= 3:

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.25,
            random_state=42
        )


        train_indexes, test_indexes = next(
            splitter.split(
                X,
                y,
                groups
            )
        )


        X_train = X[
            train_indexes
        ]

        X_test = X[
            test_indexes
        ]

        y_train = y[
            train_indexes
        ]

        y_test = y[
            test_indexes
        ]


        # recupero i partecipanti
        # presenti nei due insiemi
        train_participants = set(
            groups[
                train_indexes
            ]
        )

        test_participants = set(
            groups[
                test_indexes
            ]
        )


        participants_in_both = (
            train_participants.intersection(
                test_participants
            )
        )


        print()

        print(
            f"Partecipanti training: "
            f"{len(train_participants)}"
        )

        print(
            f"Partecipanti test: "
            f"{len(test_participants)}"
        )

        print(
            f"Partecipanti presenti in entrambi: "
            f"{len(participants_in_both)}"
        )


        print()

        print(
            f"Ripetizioni training: "
            f"{len(X_train)}"
        )

        print(
            f"Ripetizioni test: "
            f"{len(X_test)}"
        )


        # controllo che nel training
        # siano presenti entrambe le classi
        if len(np.unique(y_train)) >= 2:

            # applico SMOTE solamente
            # al training set
            smote = SMOTE(
                sampling_strategy="not majority",
                random_state=42,
                k_neighbors=5
            )


            X_train_resampled, y_train_resampled = (
                smote.fit_resample(
                    X_train,
                    y_train
                )
            )


            print()

            print(
                "Distribuzione training prima di SMOTE:"
            )

            print(
                f"Correct: "
                f"{np.sum(y_train == 'correct')}"
            )

            print(
                f"Wrong: "
                f"{np.sum(y_train == 'wrong')}"
            )


            print()

            print(
                "Distribuzione training dopo SMOTE:"
            )

            print(
                f"Correct: "
                f"{np.sum(y_train_resampled == 'correct')}"
            )

            print(
                f"Wrong: "
                f"{np.sum(y_train_resampled == 'wrong')}"
            )


            test_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight="balanced",
                random_state=42
            )


            # addestro la Random Forest
            # sui dati aumentati con SMOTE
            test_model.fit(
                X_train_resampled,
                y_train_resampled
            )


            # il test viene eseguito solamente
            # sui dati reali
            predictions = test_model.predict(
                X_test
            )


            print()

            print(
                "========== RISULTATI TEST =========="
            )

            print(
                f"Accuracy: "
                f"{accuracy_score(y_test,predictions):.3f}"
            )

            print()


            print(
                classification_report(
                    y_test,
                    predictions,
                    labels=[
                        "correct",
                        "wrong"
                    ],
                    zero_division=0
                )
            )


            print(
                "Confusion matrix:"
            )

            print(
                confusion_matrix(
                    y_test,
                    predictions,
                    labels=[
                        "correct",
                        "wrong"
                    ]
                )
            )


        else:

            print()

            print(
                "Dataset ancora troppo piccolo "
                "per effettuare un test affidabile"
            )


    else:

        print()

        print(
            "Dataset ancora troppo piccolo "
            "per separare training e test"
        )


    # creo il modello finale
    # utilizzando tutti i dati disponibili
    # aumentati tramite SMOTE
    final_smote = SMOTE(
        sampling_strategy="not majority",
        random_state=42,
        k_neighbors=5
    )


    X_resampled, y_resampled = (
        final_smote.fit_resample(
            X,
            y
        )
    )


    print()

    print(
        "Distribuzione dataset finale dopo SMOTE:"
    )

    print(
        f"Correct: "
        f"{np.sum(y_resampled == 'correct')}"
    )

    print(
        f"Wrong: "
        f"{np.sum(y_resampled == 'wrong')}"
    )


    # creo la Random Forest finale
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )


    # addestro il modello finale
    # sul dataset completo aumentato con SMOTE
    model.fit(
        X_resampled,
        y_resampled
    )


    os.makedirs(
        "models",
        exist_ok=True
    )


    model_data = {

        "model": model,

        "target_frames": TARGET_FRAMES,

        "landmark_names": LANDMARK_NAMES

    }


    joblib.dump(
        model_data,
        MODEL_PATH
    )


    print()

    print(
        "========== TRAINING COMPLETATO =========="
    )

    print(
        f"Modello salvato in: {MODEL_PATH}"
    )



if __name__ == "__main__":

    train_classifier()