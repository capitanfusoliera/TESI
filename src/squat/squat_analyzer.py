from src.squat.repetition import (
    Repetition,
    SquatPhase
)


class SquatAnalyzer:

    #inizializzo un ripetizone
    def __init__(self, phase_detector, error_manager):

        self.phase_detector = phase_detector
        self.error_manager = error_manager

        self.current_phase = SquatPhase.STANDING

        self.current_rep = None

        self.rep_count = 0

        self.completed_reps = []


    #ANALISI FRAME
    def update(self, body):

        #carico la fase rilevata 

        new_phase = self.phase_detector.detect(body, self.current_phase)


        if (
            self.current_phase == SquatPhase.STANDING
            and
            new_phase == SquatPhase.DESCENDING
        ):
            self._start_rep(body.frame_number)

        self.current_phase = new_phase

        #carico la ripetizione corrente e gli errori rilevati

        if self.current_rep is not None:

            self.current_rep.frames.append(body)

            detected_errors = (
                self.error_manager.check(body=body, phase=self.current_phase)
            )

            self._update_errors(detected_errors)

        #fine della ripetizione

        if (
            self.current_rep is not None
            and
            self.current_phase == SquatPhase.STANDING
        ):

            self._finish_rep(body.frame_number)


    #CREAZIONE DELLA RIPETIZIONE
    def _start_rep(self, frame_number):

        self.rep_count += 1

        self.current_rep = Repetition(
            number=self.rep_count,
            start_frame=frame_number,
            current_phase=(SquatPhase.DESCENDING)
        )

        print(
            f"Start repetition "
            f"{self.rep_count}"
        )

    #FINE RIPETIZIONE
    def _finish_rep(self, frame_number):

        self.current_rep.end_frame = (frame_number)

        self.completed_reps.append(self.current_rep)

        print(f"End repetition "f"{self.current_rep.number}")

        self.current_rep = None

    #AGGIORNAMENTO ERRORI
    def _update_errors(self, detected_errors):

        for (error_name, detected) in detected_errors.items():

            if detected:
                self.current_rep.errors[error_name] = True

            else:
                self.current_rep.errors.setdefault(error_name,False)