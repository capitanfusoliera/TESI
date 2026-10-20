from src.squat.repetition import SquatPhase


class SquatPhaseDetector:

    def __init__(self):

        # soglia per passare da in piedi a discesa
        self.standing_threshold = 160

        # soglia per entrare nella fase di accosciata
        self.bottom_threshold = 110


    def detect(self, body, current_phase):

        left_knee_angle = body.angles.ginocchio_sinistro
        right_knee_angle = body.angles.ginocchio_destro


        # IN PIEDI
        # se entrambe le ginocchia scendono sotto i 160 gradi
        # inizia la fase di discesa

        if current_phase == SquatPhase.STANDING:

            if (
                left_knee_angle < self.standing_threshold
                and
                right_knee_angle < self.standing_threshold
            ):

                return SquatPhase.DESCENDING


        # DISCESA
        # quando entrambe le ginocchia scendono sotto i 110 gradi
        # si entra nella fase di accosciata

        elif current_phase == SquatPhase.DESCENDING:

            if (
                left_knee_angle < self.bottom_threshold
                and
                right_knee_angle < self.bottom_threshold
            ):

                return SquatPhase.BOTTOM


        # ACCOSCIATA
        # quando entrambe le ginocchia tornano sopra i 110 gradi
        # inizia la fase di risalita

        elif current_phase == SquatPhase.BOTTOM:

            if (
                left_knee_angle > self.bottom_threshold
                and
                right_knee_angle > self.bottom_threshold
            ):

                return SquatPhase.ASCENDING


        # RISALITA
        # quando entrambe le ginocchia tornano almeno a 160 gradi
        # la ripetizione è terminata e si torna in piedi

        elif current_phase == SquatPhase.ASCENDING:

            if (
                left_knee_angle >= self.standing_threshold
                and
                right_knee_angle >= self.standing_threshold
            ):

                return SquatPhase.STANDING


        return current_phase