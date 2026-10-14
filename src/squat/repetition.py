from dataclasses import dataclass, field
from enum import Enum


class SquatPhase(Enum):
    STANDING = "standing"
    DESCENDING = "descending"
    BOTTOM = "bottom"
    ASCENDING = "ascending"


@dataclass
class Repetition:

    number: int

    start_frame: int
    end_frame: int

    current_phase: SquatPhase | None = None

    frames: list = field(
        default_factory=list
    )

    errors: dict = field(
        default_factory=dict
    )