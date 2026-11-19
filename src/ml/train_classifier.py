#prende dal dataset tutti i file utilizzabili
#controlla la label e esegue l'algoritmo


import json
import os
import joblib
import numpy as np

from sklearn.tree import DecisionTreeClassifier
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

            # salvo anche il video di provenienza
            # per separare training e test
            groups.append(
                video_name
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

    medium_count = np.sum(
        y == "medium"
    )

    wrong_count = np.sum(
        y == "wrong"
    )


    print(
        f"Correct: {correct_count}"
    )

    print(
        f"Medium: {medium_count}"
    )

    print(
        f"Wrong: {wrong_count}"
    )


    # devo avere tutte e tre le classi
    if len(np.unique(y)) < 3:

        print(
            "ERRORE: servono squat correct, "
            "medium e wrong"
        )

        return


    unique_videos = np.unique(
        groups
    )


    # se ho abbastanza video
    # creo training set e test set
    if len(unique_videos) >= 3:

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


        # controllo che nel training
        # siano presenti tutte e tre le classi

        if len(np.unique(y_train)) >= 3:

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
                f"Medium: "
                f"{np.sum(y_train == 'medium')}"
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
                f"Medium: "
                f"{np.sum(y_train_resampled == 'medium')}"
            )

            print(
                f"Wrong: "
                f"{np.sum(y_train_resampled == 'wrong')}"
            )


            test_model = DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            )


            test_model.fit(
                X_train_resampled,
                y_train_resampled
            )


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
                        "medium",
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
                        "medium",
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
        f"Medium: "
        f"{np.sum(y_resampled == 'medium')}"
    )

    print(
        f"Wrong: "
        f"{np.sum(y_resampled == 'wrong')}"
    )


    model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )


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