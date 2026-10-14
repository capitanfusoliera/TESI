from src.squat.repetition import (
    Repetition,
    SquatPhase
)


class SquatAnalyzer:

    def __init__(
        self,
        phase_detector,
        error_manager
    ):

        # Moduli utilizzati dall'analizzatore
        self.phase_detector = phase_detector
        self.error_manager = error_manager

        # Stato generale dello squat
        self.current_phase = SquatPhase.STANDING

        # Ripetizione attualmente in corso
        self.current_rep = None

        # Numero totale di ripetizioni iniziate
        self.rep_count = 0

        # Ripetizioni completate
        self.completed_reps = []


    def update(
        self,
        body
    ):

        # ----------------------------------------------------
        # 1. DETECT CURRENT PHASE
        # ----------------------------------------------------

        new_phase = self.phase_detector.detect(
            body,
            self.current_phase
        )


        # ----------------------------------------------------
        # 2. CHECK IF A NEW REP STARTS
        # ----------------------------------------------------

        if (
            self.current_phase == SquatPhase.STANDING
            and
            new_phase == SquatPhase.DESCENDING
        ):

            self._start_rep(
                body.frame_number
            )


        # ----------------------------------------------------
        # 3. UPDATE CURRENT PHASE
        # ----------------------------------------------------

        self.current_phase = new_phase


        # ----------------------------------------------------
        # 4. IF A REP IS ACTIVE, SAVE CURRENT FRAME
        # ----------------------------------------------------

        if self.current_rep is not None:

            self.current_rep.current_phase = (
                self.current_phase
            )

            self.current_rep.frames.append(
                body
            )


            # ------------------------------------------------
            # 5. CHECK ERRORS FOR CURRENT PHASE
            # ------------------------------------------------

            detected_errors = (
                self.error_manager.check(
                    body=body,
                    phase=self.current_phase
                )
            )

            self._update_errors(
                detected_errors
            )


        # ----------------------------------------------------
        # 6. CHECK IF CURRENT REP IS FINISHED
        # ----------------------------------------------------

        if (
            self.current_rep is not None
            and
            self.current_phase == SquatPhase.STANDING
        ):

            self._finish_rep(
                body.frame_number
            )


    def _start_rep(
        self,
        frame_number
    ):

        self.rep_count += 1

        self.current_rep = Repetition(
            number=self.rep_count,
            start_frame=frame_number,
            current_phase=(
                SquatPhase.DESCENDING
            )
        )

        print(
            f"Start repetition "
            f"{self.rep_count}"
        )


    def _finish_rep(
        self,
        frame_number
    ):

        self.current_rep.end_frame = (
            frame_number
        )

        self.completed_reps.append(
            self.current_rep
        )

        print(
            f"End repetition "
            f"{self.current_rep.number}"
        )

        self.current_rep = None


    def _update_errors(
        self,
        detected_errors
    ):

        for (
            error_name,
            detected
        ) in detected_errors.items():

            # Once an error becomes True
            # during the rep, keep it True
            if detected:

                self.current_rep.errors[
                    error_name
                ] = True

            else:

                self.current_rep.errors.setdefault(
                    error_name,
                    False
                )