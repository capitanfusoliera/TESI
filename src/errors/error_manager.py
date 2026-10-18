from src.errors.barbell_tilt import check_barbell_tilt
from src.squat.repetition import SquatPhase


class ErrorManager:

    def __init__(self):

        self.reference_body_height = None


    def set_reference_body_height(self, body, phase):

        # l'altezza viene calcolata una sola volta
        # quando la persona è ancora in piedi

        if (
            self.reference_body_height is None
            and
            phase == SquatPhase.STANDING
        ):

            left_heel = body.landmarks.tallone_sinistro
            right_heel = body.landmarks.tallone_destro

            nose = body.landmarks.naso


            heel_center_y = (
                left_heel.y
                +
                right_heel.y
            ) / 2


            self.reference_body_height = abs(
                heel_center_y - nose.y
            )


            print(
                f"ALTEZZA DI RIFERIMENTO | "
                f"Frame: {body.frame_number} | "
                f"Altezza: "
                f"{self.reference_body_height:.1f} px"
            )


    def check(self, body, phase):

        errors = {}


        # il tilt del bilanciere viene controllato
        # in tutte le fasi dello squat
        errors["barbell_tilt"] = check_barbell_tilt(
            body,
            self.reference_body_height
        )


        return errors