# Questo modulo coordina l'intera analisi dello squat
# Gestisce l'inizio e la fine delle ripetizioni, aggiorna la fase corrente
# salva i BodyModel della ripetizione e richiama i controlli degli errori
# Conserva infine tutte le ripetizioni completate e i relativi risultati

from src.squat.repetition import Repetition, SquatPhase


class SquatAnalyzer:

    #come imput prende la fase degli squat e il gestore degli errori
    def __init__(self, phase_detector, error_manager, analyze_errors=True):

        self.phase_detector = phase_detector
        self.error_manager = error_manager

        # indica se devo effettuare
        # anche l'analisi deterministica degli errori
        self.analyze_errors = analyze_errors

        self.current_phase = SquatPhase.STANDING
        self.current_rep = None

        # numero di ripetizioni completate
        self.rep_count = 0

        self.completed_reps = []

        # contiene gli errori validati
        # per ogni frame
        self.validated_errors_by_frame = {}

        # contiene i risultati frame per frame della fase attuale
        self.phase_frames = []

        #conta quanti frame contigui servono per dichiare un errore errore
        self.min_error_frames = {
            "knee_convergence": 6,
            "heel_lift": 1,
            "heel_asymmetry": 6,
            "knee_asymmetry": 6,
            "barbell_position": 6,
            "torso_inclination": 6
        }

        # contiene i risultati delle varie fasi
        self.rep_phases = {}


    def update(self, body):

        # salvo una volta sola l'altezza iniziale
        # della persona quando è in piedi
        # solamente se sto facendo l'analisi degli errori
        if self.analyze_errors:

            self.error_manager.set_reference_body_height(
                body,
                self.current_phase
            )

            self.error_manager.set_reference_heel_y(
                body,
                self.current_phase
            )

            self.error_manager.set_reference_heel_hip_height(
                body,
                self.current_phase
            )

            self.error_manager.set_reference_torso_length(
                body,
                self.current_phase
            )


        # controllo se deve cambiare fase
        new_phase = self.phase_detector.detect(
            body,
            self.current_phase
        )


        # se passa direttamente da discesa a risalita
        # significa che il bottom non è stato raggiunto
        if (
            self.analyze_errors
            and
            self.current_phase == SquatPhase.DESCENDING
            and
            new_phase == SquatPhase.ASCENDING
            and
            self.current_rep is not None
        ):

            self.current_rep.errors[
                "insufficient_depth"
            ] = True


        # se la fase sta cambiando
        if new_phase != self.current_phase:

            # salvo gli errori della fase
            # solamente durante l'analisi deterministica
            if self.analyze_errors:

                self._save_phase_errors(
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


            # se ritorno in piedi
            # termino la ripetizione
            elif (
                self.current_phase != SquatPhase.STANDING
                and
                new_phase == SquatPhase.STANDING
                and
                self.current_rep is not None
            ):

                self._finish_rep(
                    body.frame_number
                )


        # aggiorno la fase corrente
        self.current_phase = new_phase


        # controllo gli errori solamente
        # se è attiva l'analisi deterministica
        if self.analyze_errors:

            detected_errors = self.error_manager.check(
                body=body,
                phase=self.current_phase
            )


            # salvo errori e frame
            # della fase attuale
            self.phase_frames.append(
                (
                    body.frame_number,
                    detected_errors
                )
            )


        else:

            detected_errors = {}


        # se esiste una ripetizione aperta
        if self.current_rep is not None:

            # salvo sempre il bodymodel
            # perché serve anche per il dataset
            self.current_rep.frames.append(
                body
            )


            # aggiorno gli errori solamente
            # durante l'analisi deterministica
            if self.analyze_errors:

                self._update_errors(
                    detected_errors
                )


        return detected_errors


    def _start_rep(
        self,
        frame_number
    ):

        self.current_rep = Repetition(
            number=self.rep_count,
            start_frame=frame_number
        )


        # aggiungo alla ripetizione anche eventuali errori
        # rilevati durante la fase STANDING iniziale
        # solamente durante l'analisi deterministica
        if self.analyze_errors:

            for phase_data in self.rep_phases.values():

                for error_name in phase_data[
                    "errors"
                ]:

                    self.current_rep.errors[
                        error_name
                    ] = True


    def _finish_rep(
        self,
        frame_number
    ):

        self.current_rep.end_frame = frame_number


        # salvo la ripetizione completata
        self.completed_reps.append(
            self.current_rep
        )


        # stampo tutti i risultati della ripetizione
        # solamente durante l'analisi deterministica
        if self.analyze_errors:

            self._print_rep_results(
                self.current_rep.number
            )


        # aumento il counter solamente
        # quando la ripetizione è completata
        self.rep_count += 1


        self.current_rep = None


        # preparo i risultati
        # per la prossima ripetizione
        self.rep_phases = {}


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


    def _save_phase_errors(
        self,
        phase
    ):

        if not self.phase_frames:
            return


        start_frame = self.phase_frames[
            0
        ][
            0
        ]

        end_frame = self.phase_frames[
            -1
        ][
            0
        ]


        # recupero tutti i tipi di errore
        # controllati durante questa fase
        error_names = []


        for frame_number, errors in self.phase_frames:

            for error_name in errors:

                if error_name not in error_names:

                    error_names.append(
                        error_name
                    )


        phase_errors = {}


        for error_name in error_names:

            error_ranges = self._get_error_ranges(
                error_name
            )


            if error_ranges:

                phase_errors[
                    error_name
                ] = error_ranges


                # salvo l'errore nei singoli frame
                # appartenenti agli intervalli validati
                for frame_number, errors in self.phase_frames:

                    for start_frame, end_frame in error_ranges:

                        if (
                            start_frame
                            <=
                            frame_number
                            <=
                            end_frame
                        ):

                            if (
                                frame_number
                                not in
                                self.validated_errors_by_frame
                            ):

                                self.validated_errors_by_frame[
                                    frame_number
                                ] = {}


                            self.validated_errors_by_frame[
                                frame_number
                            ][
                                error_name
                            ] = True

                            break


        self.rep_phases[
            phase
        ] = {

            "start_frame": start_frame,

            "end_frame": end_frame,

            "errors": phase_errors

        }


    def _get_error_ranges(
        self,
        error_name
    ):

        ranges = []

        start_error = None
        end_error = None

        error_frame_count = 0


        min_frames = self.min_error_frames.get(
            error_name,
            1
        )


        for frame_number, errors in self.phase_frames:

            detected = errors.get(
                error_name,
                False
            )


            if detected:

                if start_error is None:

                    start_error = frame_number

                    error_frame_count = 1


                else:

                    error_frame_count += 1


                end_error = frame_number


            else:

                if start_error is not None:

                    if error_frame_count >= min_frames:

                        ranges.append(
                            (
                                start_error,
                                end_error
                            )
                        )


                    start_error = None

                    end_error = None

                    error_frame_count = 0


        # se la fase finisce mentre
        # l'errore è ancora presente
        if start_error is not None:

            if error_frame_count >= min_frames:

                ranges.append(
                    (
                        start_error,
                        end_error
                    )
                )


        return ranges


    def _print_rep_results(
        self,
        repetition_number
    ):

        print()

        print(
            f"========== RIPETIZIONE "
            f"{repetition_number} =========="
        )


        phases = [
            SquatPhase.STANDING,
            SquatPhase.DESCENDING,
            SquatPhase.BOTTOM,
            SquatPhase.ASCENDING
        ]


        for phase in phases:

            if phase not in self.rep_phases:
                continue


            phase_data = self.rep_phases[
                phase
            ]


            print()

            print(
                self._phase_name(
                    phase
                )
            )


            # se nella fase non sono stati trovati errori
            if not phase_data[
                "errors"
            ]:

                print(
                    "Nessun errore"
                )

                continue


            # stampo tutti gli errori trovati
            for error_name, error_ranges in phase_data[
                "errors"
            ].items():

                for start, end in error_ranges:

                    if start == end:

                        print(
                            f"- {error_name} | "
                            f"errore frame {start}"
                        )


                    else:

                        print(
                            f"- {error_name} | "
                            f"errore frame {start}-{end}"
                        )


        # controllo se la ripetizione
        # non ha raggiunto la profondità sufficiente
        if self.current_rep.errors.get(
            "insufficient_depth",
            False
        ):

            print()

            print(
                "- insufficient_depth"
            )


        print()

        print(
            f"========== FINE RIPETIZIONE "
            f"{repetition_number} =========="
        )


    def _phase_name(
        self,
        phase
    ):

        if phase == SquatPhase.STANDING:

            return "STANDING"


        elif phase == SquatPhase.DESCENDING:

            return "DESCENDING"


        elif phase == SquatPhase.BOTTOM:

            return "BOTTOM"


        elif phase == SquatPhase.ASCENDING:

            return "ASCENDING"


        return phase.value


    def finish_analysis(self):

        # se il video finisce durante una ripetizione
        if self.current_rep is not None:


            # salvo e stampo gli errori
            # solamente durante l'analisi deterministica
            if self.analyze_errors:

                self._save_phase_errors(
                    self.current_phase
                )

                self.phase_frames = []


                print()

                print(
                    f"========== RIPETIZIONE "
                    f"{self.current_rep.number} "
                    f"INCOMPLETA =========="
                )


                phases = [
                    SquatPhase.STANDING,
                    SquatPhase.DESCENDING,
                    SquatPhase.BOTTOM,
                    SquatPhase.ASCENDING
                ]


                for phase in phases:

                    if phase not in self.rep_phases:
                        continue


                    phase_data = self.rep_phases[
                        phase
                    ]


                    print()

                    print(
                        self._phase_name(
                            phase
                        )
                    )


                    if not phase_data[
                        "errors"
                    ]:

                        print(
                            "Nessun errore"
                        )

                        continue


                    for error_name, error_ranges in phase_data[
                        "errors"
                    ].items():

                        for start, end in error_ranges:

                            if start == end:

                                print(
                                    f"- {error_name} | "
                                    f"errore frame {start}"
                                )


                            else:

                                print(
                                    f"- {error_name} | "
                                    f"errore frame {start}-{end}"
                                )