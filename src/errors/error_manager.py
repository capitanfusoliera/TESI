from src.errors.barbell_tilt import check_barbell_tilt
from src.errors.knee_convergence import check_knee_convergence
from src.squat.repetition import SquatPhase


class ErrorManager:

    def __init__(self):

        self.reference_body_height = None

        self.phase_checks = {

            SquatPhase.STANDING: [
                ("barbell_tilt", self.check_barbell_tilt)
            ],

            SquatPhase.DESCENDING: [
                ("barbell_tilt", self.check_barbell_tilt),
                ("knee_convergence", self.check_knee_convergence)
            ],

            SquatPhase.BOTTOM: [
                ("barbell_tilt", self.check_barbell_tilt),
                ("knee_convergence", self.check_knee_convergence)
            ],

            SquatPhase.ASCENDING: [
                ("barbell_tilt", self.check_barbell_tilt),
                ("knee_convergence", self.check_knee_convergence)
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