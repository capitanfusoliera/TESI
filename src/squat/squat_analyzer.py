# Questo modulo coordina l'intera analisi dello squat
# Gestisce l'inizio e la fine delle ripetizioni, aggiorna la fase corrente
# salva i BodyModel della ripetizione e richiama i controlli degli errori
# Conserva infine tutte le ripetizioni completate e i relativi risultati
from src.squat.repetition import Repetition, SquatPhase


class SquatAnalyzer:

    def __init__(self, phase_detector, error_manager):

        self.phase_detector = phase_detector
        self.error_manager = error_manager

        self.current_phase = SquatPhase.STANDING

        self.current_rep = None

        # numero di ripetizioni COMPLETATE
        self.rep_count = 0

        self.completed_reps = []


        # contiene i risultati frame per frame
        # della fase attuale
        self.phase_frames = []


    def update(self, body):

        # salvo una volta sola l'altezza iniziale
        # della persona quando è in piedi
        self.error_manager.set_reference_body_height(
            body,
            self.current_phase
        )


        # controllo se deve cambiare fase
        new_phase = self.phase_detector.detect(
            body,
            self.current_phase
        )


        # se la fase sta cambiando
        # stampo il riepilogo della fase appena terminata
        if new_phase != self.current_phase:

            self._print_phase_errors(
                self.current_phase
            )

            self.phase_frames = []


        # se passo da in piedi a discesa
        # inizia una nuova ripetizione
        if (
            self.current_phase == SquatPhase.STANDING
            and
            new_phase == SquatPhase.DESCENDING
        ):

            self._start_rep(
                body.frame_number
            )


        # aggiorno la fase corrente
        self.current_phase = new_phase


        # controllo gli errori SEMPRE
        # quindi anche nella fase STANDING
        detected_errors = self.error_manager.check(
            body=body,
            phase=self.current_phase
        )


        # salvo errori e frame della fase attuale
        self.phase_frames.append(
            (
                body.frame_number,
                detected_errors
            )
        )


        # se esiste una ripetizione aperta
        if self.current_rep is not None:

            self.current_rep.frames.append(
                body
            )

            self._update_errors(
                detected_errors
            )


        # se la ripetizione è aperta
        # e siamo tornati in piedi
        # la ripetizione è terminata
        if (
            self.current_rep is not None
            and
            self.current_phase == SquatPhase.STANDING
        ):

            self._finish_rep(
                body.frame_number
            )


    def _start_rep(self, frame_number):

        # il counter non viene ancora aumentato
        # perché la ripetizione non è ancora completata
        repetition_number = (
            self.rep_count + 1
        )

        self.current_rep = Repetition(
            number=repetition_number,
            start_frame=frame_number
        )

        print()
        print(
            f"===== INIZIO RIPETIZIONE "
            f"{repetition_number} | "
            f"Frame {frame_number} ====="
        )


    def _finish_rep(self, frame_number):

        self.current_rep.end_frame = (
            frame_number
        )

        self.completed_reps.append(
            self.current_rep
        )


        # il counter aumenta SOLO
        # quando la ripetizione è conclusa
        self.rep_count += 1


        print()
        print(
            f"===== FINE RIPETIZIONE "
            f"{self.current_rep.number} | "
            f"Frame {frame_number} ====="
        )

        print(
            f"Ripetizioni completate: "
            f"{self.rep_count}"
        )


        self.current_rep = None


    def _update_errors(
        self,
        detected_errors
    ):

        for error_name, detected in detected_errors.items():

            if detected:

                self.current_rep.errors[
                    error_name
                ] = True

            else:

                self.current_rep.errors.setdefault(
                    error_name,
                    False
                )


    def _print_phase_errors(
        self,
        phase
    ):

        if not self.phase_frames:
            return


        start_frame = (
            self.phase_frames[0][0]
        )

        end_frame = (
            self.phase_frames[-1][0]
        )


        print()

        print(
            f"----- {self._phase_name(phase)} | "
            f"Frame {start_frame} -> "
            f"{end_frame} -----"
        )


        # recupero tutti i tipi di errore
        # controllati durante questa fase
        error_names = set()

        for frame_number, errors in self.phase_frames:

            for error_name in errors:

                error_names.add(
                    error_name
                )


        phase_has_errors = False


        for error_name in error_names:

            error_ranges = (
                self._get_error_ranges(
                    error_name
                )
            )


            if error_ranges:

                phase_has_errors = True

                for start, end in error_ranges:

                    if start == end:

                        print(
                            f"{error_name} | "
                            f"errore frame {start}"
                        )

                    else:

                        print(
                            f"{error_name} | "
                            f"errore frame "
                            f"{start}-{end}"
                        )


        if not phase_has_errors:

            print(
                "Nessun errore"
            )


    def _get_error_ranges(
        self,
        error_name
    ):

        ranges = []

        start_error = None
        end_error = None


        for frame_number, errors in self.phase_frames:

            detected = errors.get(
                error_name,
                False
            )


            if detected:

                if start_error is None:

                    start_error = (
                        frame_number
                    )

                end_error = (
                    frame_number
                )


            else:

                if start_error is not None:

                    ranges.append(
                        (
                            start_error,
                            end_error
                        )
                    )

                    start_error = None
                    end_error = None


        # se la fase finisce mentre
        # l'errore è ancora presente
        if start_error is not None:

            ranges.append(
                (
                    start_error,
                    end_error
                )
            )


        return ranges


    def _phase_name(
        self,
        phase
    ):

        if phase == SquatPhase.STANDING:
            return "FASE 0 - IN PIEDI"

        elif phase == SquatPhase.DESCENDING:
            return "FASE DISCESA"

        elif phase == SquatPhase.BOTTOM:
            return "FASE ACCOSCIATA"

        elif phase == SquatPhase.ASCENDING:
            return "FASE RISALITA"


        return phase.value


    def finish_analysis(self):

        # stampa anche l'ultima fase del video
        # perché potrebbe non esserci un successivo
        # cambio di fase che la faccia stampare

        self._print_phase_errors(
            self.current_phase
        )

        self.phase_frames = []