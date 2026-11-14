#utilizza il modello allenato
#per controllare la correttezza degli squat

import os
import joblib

from src.ml.dataset_builder import body_to_data
from src.ml.feature_extractor import extract_repetition_features


class SquatClassifier:

    def __init__(
        self,
        model_path="models/squat_classifier.pkl"
    ):

        # controllo che il modello
        # sia stato precedentemente addestrato
        if not os.path.exists(
            model_path
        ):

            raise FileNotFoundError(
                f"Modello non trovato: {model_path}"
            )


        # carico il modello allenato
        model_data = joblib.load(
            model_path
        )


        self.model = model_data[
            "model"
        ]



    # classifica una singola ripetizione
    def predict(
        self,
        repetition
    ):

        frames_data = []


        # trasformo i bodymodel
        # nello stesso formato utilizzato
        # durante la creazione del dataset
        for body in repetition.frames:

            frames_data.append(
                body_to_data(body)
            )


        repetition_data = {

            "frames": frames_data

        }


        # estraggo le feature utilizzando
        # lo stesso metodo usato nel training
        features = extract_repetition_features(
            repetition_data
        )


        if features is None:

            return None


        # il modello si aspetta una lista
        # di ripetizioni
        prediction = self.model.predict(
            [
                features
            ]
        )


        return str(
            prediction[0]
        )