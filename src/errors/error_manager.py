from src.errors.barbell_tilt import check_barbell_tilt
from src.errors.knee_convergence import check_knee_convergence
from src.squat.repetition import SquatPhase
from src.errors.heel_lift import check_heel_lift
from src.errors.heels_asymmetry import check_heel_asymmetry


class ErrorManager:

    def __init__(self):

        self.reference_body_height = None
        self.reference_left_heel_y = None
        self.reference_right_heel_y = None

        self.phase_checks = {

            SquatPhase.STANDING: [
                ("barbell_tilt", self.check_barbell_tilt),
                ("heel_asymmetry", self.check_heel_asymmetry)
            ],

            SquatPhase.DESCENDING: [
                ("barbell_tilt", self.check_barbell_tilt),
                ("knee_convergence", self.check_knee_convergence),
                ("heel_lift", self.check_heel_lift),
                ("heel_asymmetry", self.check_heel_asymmetry)
            ],

            SquatPhase.BOTTOM: [
                ("barbell_tilt", self.check_barbell_tilt),
                ("knee_convergence", self.check_knee_convergence),
                ("heel_lift", self.check_heel_lift),
                ("heel_asymmetry", self.check_heel_asymmetry)
            ],

            SquatPhase.ASCENDING: [
                ("barbell_tilt", self.check_barbell_tilt),
                ("knee_convergence", self.check_knee_convergence),
                ("heel_lift", self.check_heel_lift),
                ("heel_asymmetry", self.check_heel_asymmetry)
            ]
        }


    def set_reference_body_height(self, body, phase):

        if self.reference_body_height is None and phase == SquatPhase.STANDING:

            left_heel = body.landmarks.tallone_sinistro
            right_heel = body.landmarks.tallone_destro
            nose = body.landmarks.naso

            heel_center_y = (left_heel.y + right_heel.y) / 2

            self.reference_body_height = abs(heel_center_y - nose.y)


    def check(self, body, phase):

        errors = {}

        for error_name, check_function in self.phase_checks.get(phase, []):

            errors[error_name] = check_function(body)

        return errors


    def check_barbell_tilt(self, body):

        return check_barbell_tilt(body,self.reference_body_height)


    def check_knee_convergence(self, body):

        return check_knee_convergence(body)

    def check_heel_lift(self, body):

        return check_heel_lift(
            body,
            self.reference_left_heel_y,
            self.reference_right_heel_y
        )

    def set_reference_heel_y(self, body, phase):

        if (
            self.reference_left_heel_y is None
            and
            self.reference_right_heel_y is None
            and
            phase == SquatPhase.STANDING
        ):

            self.reference_left_heel_y = (
                body.landmarks.tallone_sinistro.y
            )

            self.reference_right_heel_y = (
                body.landmarks.tallone_destro.y
            )

    def check_heel_asymmetry(self, body):

        return check_heel_asymmetry(body)