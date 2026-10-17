from src.squat.repetition import SquatPhase


class SquatPhaseDetector:

    def __init__(self):

        self.knee_threshold = 165

        self.left_knee_below = False
        self.right_knee_below = False


    def detect(
        self,
        body,
        current_phase
    ):

        left_knee_angle = (
            body.angles.left_knee_angle
        )

        right_knee_angle = (
            body.angles.right_knee_angle
        )


        # ----------------------------------------------------
        # STANDING
        # ----------------------------------------------------

        if current_phase == SquatPhase.STANDING:

            # Left knee crosses the threshold
            if (
                left_knee_angle < self.knee_threshold
                and
                not self.left_knee_below
            ):

                self.left_knee_below = True

                print(
                    f"LEFT knee below "
                    f"{self.knee_threshold}° "
                    f"at frame {body.frame_number} "
                    f"({left_knee_angle:.2f}°)"
                )


            # Right knee crosses the threshold
            if (
                right_knee_angle < self.knee_threshold
                and
                not self.right_knee_below
            ):

                self.right_knee_below = True

                print(
                    f"RIGHT knee below "
                    f"{self.knee_threshold}° "
                    f"at frame {body.frame_number} "
                    f"({right_knee_angle:.2f}°)"
                )


            # Both knees are now below the threshold
            if (
                left_knee_angle < self.knee_threshold
                and
                right_knee_angle < self.knee_threshold
            ):

                print(
                    f"DESCENDING starts "
                    f"at frame {body.frame_number}"
                )

                return SquatPhase.DESCENDING


        return current_phase