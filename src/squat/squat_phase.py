from src.squat.repetition import SquatPhase


class SquatPhaseDetector:

    def __init__(self):

        # soglia per passare da in piedi a discesa
        self.standing_threshold = 170

        # soglia per entrare nella fase di accosciata
        self.bottom_threshold = 110

        # aumento minimo dell'angolo
        # per considerare iniziata la risalita
        self.ascending_threshold = 8

        # numero di frame consecutivi
        # per confermare la risalita
        self.ascending_confirmation_frames = 6

        self.ascending_frame_count = 0

        self.minimum_knee_angle = None


    def detect(self, body, current_phase):

        left_knee_angle = body.angles.ginocchio_sinistro
        right_knee_angle = body.angles.ginocchio_destro

        knee_angle = (left_knee_angle + right_knee_angle) / 2


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

        # DISCESA
        # controllo se viene raggiunto il bottom
        # oppure se la persona inizia a risalire prima

        elif current_phase == SquatPhase.DESCENDING:

            # aggiorno l'angolo minimo raggiunto
            # durante la discesa
            if (
                self.minimum_knee_angle is None
                or
                knee_angle < self.minimum_knee_angle
            ):

                self.minimum_knee_angle = knee_angle

                self.ascending_frame_count = 0


            # se raggiunge la profondità
            # entra nella fase di accosciata
            if (
                left_knee_angle < self.bottom_threshold
                and
                right_knee_angle < self.bottom_threshold
            ):

                self.minimum_knee_angle = None

                self.ascending_frame_count = 0

                return SquatPhase.BOTTOM


            # se l'angolo aumenta abbastanza
            # rispetto al minimo raggiunto
            # controllo se sta realmente risalendo
            if (
                knee_angle
                >=
                self.minimum_knee_angle + self.ascending_threshold
            ):

                self.ascending_frame_count += 1

            else:

                self.ascending_frame_count = 0


            # se la risalita è confermata
            # passo direttamente ad ascending
            if (
                self.ascending_frame_count
                >=
                self.ascending_confirmation_frames
            ):

                self.minimum_knee_angle = None

                self.ascending_frame_count = 0

                return SquatPhase.ASCENDING

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