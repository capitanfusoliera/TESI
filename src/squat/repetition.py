# Questo modulo definisce la struttura dati di una singola ripetizione
# Memorizza il numero della ripetizione, i frame di inizio e fine
# i BodyModel appartenenti alla ripetizione e gli errori rilevati
# Definisce inoltre le possibili fasi dello squat

from dataclasses import dataclass, field
from enum import Enum

#Fasi durante l'esecuzione di uno squat
class SquatPhase(Enum):
    STANDING = "standing"
    DESCENDING = "descending"
    BOTTOM = "bottom"
    ASCENDING = "ascending"

# mi salvo numero della ripetizione frame inizo e fine fase e errori
@dataclass
class Repetition:

    number: int

    start_frame: int
    end_frame: int | None = None

    frames: list = field(
        default_factory=list
    )

    errors: dict = field(
        default_factory=dict
    )