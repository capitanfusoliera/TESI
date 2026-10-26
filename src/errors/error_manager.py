from src.errors.barbell_tilt import check_barbell_tilt
from src.errors.knee_convergence import check_knee_convergence
from src.squat.repetition import SquatPhase
from src.errors.heel_lift import check_heel_lift
from src.errors.heels_asymmetry import check_heel_asymmetry
from src.errors.knees_asymmetry import check_knee_asymmetry
from src.errors.barbell_position import check_barbell_position
from src.errors.torso_inclination import check_torso_inclination
import math

class ErrorManager:

    def __init__(self):

        self.reference_torso_length = None
        self.reference_body_height = None
        self.reference_left_heel_y = None
        self.reference_right_heel_y = None

        self.phase_checks = {

            SquatPhase.STANDING: [
                ("barbell_tilt", self.check_barbell_tilt),
                ("heel_asymmetry", self.check_heel_asymmetry),
                ("heel_asymmetry", self.check_heel_asymmetry),
                ("barbell_position", self.check_barbell_position)
            ],

            SquatPhase.DESCENDING: [
                ("barbell_tilt", self.check_barbell_tilt),
                ("knee_convergence", self.check_knee_convergence),
                ("heel_lift", self.check_heel_lift),
                ("heel_asymmetry", self.check_heel_asymmetry),
                ("heel_asymmetry", self.check_heel_asymmetry),
                ("torso_inclination", self.check_torso_inclination)
            ],

            SquatPhase.BOTTOM: [
                ("barbell_tilt", self.check_barbell_tilt),
                ("knee_convergence", self.check_knee_convergence),
                ("heel_lift", self.check_heel_lift),
                ("heel_asymmetry", self.check_heel_asymmetry),
                ("heel_asymmetry", self.check_heel_asymmetry),
                ("torso_inclination", self.check_torso_inclination)
            ],

            SquatPhase.ASCENDING: [
                ("barbell_tilt", self.check_barbell_tilt),
                ("knee_convergence", self.check_knee_convergence),
                ("heel_lift", self.check_heel_lift),
                ("heel_asymmetry", self.check_heel_asymmetry),
                ("heel_asymmetry", self.check_heel_asymmetry),
                ("torso_inclination", self.check_torso_inclination)
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

    def check_knee_asymmetry(self, body):

        return check_knee_asymmetry(body)

    def check_barbell_position(self, body):

        return check_barbell_position(
            body,
            self.reference_body_height
        )

    def set_reference_torso_length(
        self,
        body,
        phase
    ):

        if (
            self.reference_torso_length is None
            and
            phase == SquatPhase.STANDING
        ):

            shoulder_center = body.virtual_points.centro_spalle
            hip_center = body.virtual_points.centro_anche


            self.reference_torso_length = math.hypot(
                shoulder_center.x - hip_center.x,
                shoulder_center.y - hip_center.y
            )

    def check_torso_inclination(self, body):

        return check_torso_inclination(
            body,
            self.reference_torso_length
        )