# costruiamo un modello del utente grazie a mediapipe
# ossa angoli e punti corporei

from dataclasses import dataclass
import math

# BASIC STRUCTURES

#punto di mediapipe
@dataclass
class Point:
    x: float
    y: float
    z: float
    visibility: float

#retta fra 2 punti 
@dataclass
class Segment:
    start: Point
    end: Point

    @property
    def length_2d(self) -> float:
        return math.hypot(
            self.end.x - self.start.x,
            self.end.y - self.start.y
        )


#  MEDIAPIPE LANDMARKS

@dataclass
class BodyLandmarks:

    # Testa
    naso: Point

    # Spalle
    spalla_sinistra: Point
    spalla_destra: Point

    # Gomiti
    gomito_sinistro: Point
    gomito_destro: Point

    # Polsi
    polso_sinistro: Point
    polso_destro: Point

    # Anche
    anca_sinistra: Point
    anca_destra: Point

    # Ginocchia
    ginocchio_sinistro: Point
    ginocchio_destro: Point

    # Caviglie
    caviglia_sinistra: Point
    caviglia_destra: Point

    # Talloni
    tallone_sinistro: Point
    tallone_destro: Point

    # Avampiede
    avampiede_sinistro: Point
    avampiede_destro: Point


#  VIRTUAL POINTS

#punti creati sulla base di mediapipe
@dataclass
class VirtualPoints:

    centro_spalle: Point
    centro_anche: Point


#  BODY SEGMENTS / BONES

@dataclass
class BodySegments:

    # Braccia
    omero_sinistro: Segment
    omero_destro: Segment

    avambraccio_sinistro: Segment
    avambraccio_destro: Segment

    # Tronco
    linea_spalle: Segment
    linea_bacino: Segment

    lato_tronco_sinistro: Segment
    lato_tronco_destro: Segment

    asse_tronco: Segment

    # Gambe
    femore_sinistro: Segment
    femore_destro: Segment

    gamba_sinistra: Segment
    gamba_destra: Segment

    # Piedi
    asse_piede_sinistro: Segment
    asse_piede_destro: Segment

    asse_caviglia_piede_sinistro: Segment
    asse_caviglia_piede_destro: Segment


#  BODY ANGLES

@dataclass
class BodyAngles:

    # Braccia
    gomito_sinistro: float
    gomito_destro: float

    spalla_sinistra: float
    spalla_destra: float

    # Anche
    anca_sinistra: float
    anca_destra: float

    # Ginocchia
    ginocchio_sinistro: float
    ginocchio_destro: float

    # Caviglie
    caviglia_sinistra: float
    caviglia_destra: float

    # Tronco
    inclinazione_tronco: float

    # Asimmetria spalle / bacino
    inclinazione_spalle: float
    inclinazione_bacino: float

    # Inclinazione gamba
    inclinazione_gamba_sinistra: float
    inclinazione_gamba_destra: float

    # Inclinazione piede
    inclinazione_piede_sinistro: float
    inclinazione_piede_destro: float

    


#  COMPLETE BODY MODEL
# rappresentazione totale della persona al frame n

@dataclass
class BodyModel:

    frame_number: int
    time_ms: int

    landmarks: BodyLandmarks
    virtual_points: VirtualPoints
    segments: BodySegments
    angles: BodyAngles


# 7. GEOMETRY FUNCTIONS

# per il punto centrale
def midpoint(
    point_a: Point,
    point_b: Point
) -> Point:

    return Point(
        x=(point_a.x + point_b.x) / 2,
        y=(point_a.y + point_b.y) / 2,
        z=(point_a.z + point_b.z) / 2,

        visibility=min(
            point_a.visibility,
            point_b.visibility
        )
    )


def calculate_angle(
    point_a: Point,
    vertex: Point,
    point_c: Point
) -> float:

    vector_a = (
        point_a.x - vertex.x,
        point_a.y - vertex.y
    )

    vector_c = (
        point_c.x - vertex.x,
        point_c.y - vertex.y
    )

    dot_product = (
        vector_a[0] * vector_c[0]
        +
        vector_a[1] * vector_c[1]
    )

    magnitude_a = math.hypot(
        vector_a[0],
        vector_a[1]
    )

    magnitude_c = math.hypot(
        vector_c[0],
        vector_c[1]
    )

    if magnitude_a == 0 or magnitude_c == 0:
        return 0.0

    cosine = (
        dot_product
        /
        (magnitude_a * magnitude_c)
    )

    # Avoid numerical problems such as 1.00000001
    cosine = max(
        -1.0,
        min(1.0, cosine)
    )

    angle_radians = math.acos(
        cosine
    )

    return math.degrees(
        angle_radians
    )


def vertical_inclination(
    segment: Segment
) -> float:

    dx = (
        segment.end.x
        - segment.start.x
    )

    dy = (
        segment.end.y
        - segment.start.y
    )

    return math.degrees(
        math.atan2(
            abs(dx),
            abs(dy)
        )
    )


def horizontal_inclination(
    segment: Segment
) -> float:

    dx = (
        segment.end.x
        - segment.start.x
    )

    dy = (
        segment.end.y
        - segment.start.y
    )

    return math.degrees(
        math.atan2(
            abs(dy),
            abs(dx)
        )
    )


def horizontal_tilt(
    segment: Segment
) -> float:

    dx = (
        segment.end.x
        - segment.start.x
    )

    dy = (
        segment.end.y
        - segment.start.y
    )

    if dx == 0:
        return 90.0

    angle = math.degrees(
        math.atan2(
            dy,
            dx
        )
    )

    # Keep result approximately between -90 and +90 degrees
    if angle > 90:
        angle -= 180

    elif angle < -90:
        angle += 180

    return angle


# ============================================================
# 8. BODY MODEL BUILDER
# ============================================================

def build_body_model(
    mediapipe_landmarks,
    frame_number=0,
    time_ms=0
) -> BodyModel:

    # --------------------------------------------------------
    # CREATE POINT FROM MEDIAPIPE LANDMARK
    # --------------------------------------------------------

    def create_point(name):

        landmark = mediapipe_landmarks[
            name
        ]

        return Point(
            x=landmark["x"],
            y=landmark["y"],
            z=landmark.get(
                "z",
                0.0
            ),
            visibility=landmark[
                "visibility"
            ]
        )

    # ========================================================
    # LANDMARKS
    # ========================================================

    landmarks = BodyLandmarks(

        nose=create_point(
            "naso"
        ),

        left_shoulder=create_point(
            "spalla_sinistra"
        ),

        right_shoulder=create_point(
            "spalla_destra"
        ),

        left_elbow=create_point(
            "gomito_sinistro"
        ),

        right_elbow=create_point(
            "gomito_destro"
        ),

        left_wrist=create_point(
            "polso_sinistro"
        ),

        right_wrist=create_point(
            "polso_destro"
        ),

        left_hip=create_point(
            "anca_sinistra"
        ),

        right_hip=create_point(
            "anca_destra"
        ),

        left_knee=create_point(
            "ginocchio_sinistro"
        ),

        right_knee=create_point(
            "ginocchio_destro"
        ),

        left_ankle=create_point(
            "caviglia_sinistra"
        ),

        right_ankle=create_point(
            "caviglia_destra"
        ),

        left_heel=create_point(
            "tallone_sinistro"
        ),

        right_heel=create_point(
            "tallone_destro"
        ),

        left_foot=create_point(
            "avampiede_sinistro"
        ),

        right_foot=create_point(
            "avampiede_destro"
        )
    )

    # ========================================================
    # VIRTUAL POINTS
    # ========================================================

    shoulder_center = midpoint(
        landmarks.left_shoulder,
        landmarks.right_shoulder
    )

    hip_center = midpoint(
        landmarks.left_hip,
        landmarks.right_hip
    )

    virtual_points = VirtualPoints(
        shoulder_center=shoulder_center,
        hip_center=hip_center
    )

    # ========================================================
    # BODY SEGMENTS / BONES
    # ========================================================

    segments = BodySegments(

        # ----------------------------------------------------
        # Arms
        # ----------------------------------------------------

        left_upper_arm=Segment(
            landmarks.left_shoulder,
            landmarks.left_elbow
        ),

        right_upper_arm=Segment(
            landmarks.right_shoulder,
            landmarks.right_elbow
        ),

        left_forearm=Segment(
            landmarks.left_elbow,
            landmarks.left_wrist
        ),

        right_forearm=Segment(
            landmarks.right_elbow,
            landmarks.right_wrist
        ),

        # ----------------------------------------------------
        # Torso
        # ----------------------------------------------------

        shoulder_line=Segment(
            landmarks.left_shoulder,
            landmarks.right_shoulder
        ),

        hip_line=Segment(
            landmarks.left_hip,
            landmarks.right_hip
        ),

        left_torso=Segment(
            landmarks.left_shoulder,
            landmarks.left_hip
        ),

        right_torso=Segment(
            landmarks.right_shoulder,
            landmarks.right_hip
        ),

        trunk_center=Segment(
            hip_center,
            shoulder_center
        ),

        # ----------------------------------------------------
        # Legs
        # ----------------------------------------------------

        left_thigh=Segment(
            landmarks.left_hip,
            landmarks.left_knee
        ),

        right_thigh=Segment(
            landmarks.right_hip,
            landmarks.right_knee
        ),

        left_shank=Segment(
            landmarks.left_knee,
            landmarks.left_ankle
        ),

        right_shank=Segment(
            landmarks.right_knee,
            landmarks.right_ankle
        ),

        # ----------------------------------------------------
        # Feet
        # ----------------------------------------------------

        left_foot=Segment(
            landmarks.left_heel,
            landmarks.left_foot
        ),

        right_foot=Segment(
            landmarks.right_heel,
            landmarks.right_foot
        ),

        left_ankle_to_foot=Segment(
            landmarks.left_ankle,
            landmarks.left_foot
        ),

        right_ankle_to_foot=Segment(
            landmarks.right_ankle,
            landmarks.right_foot
        )
    )

    # ========================================================
    # ANGLES
    # ========================================================

    angles = BodyAngles(

        # ----------------------------------------------------
        # Elbow angle
        #
        # shoulder -> elbow -> wrist
        # ----------------------------------------------------

        left_elbow=calculate_angle(
            landmarks.left_shoulder,
            landmarks.left_elbow,
            landmarks.left_wrist
        ),

        right_elbow=calculate_angle(
            landmarks.right_shoulder,
            landmarks.right_elbow,
            landmarks.right_wrist
        ),

        # ----------------------------------------------------
        # Shoulder angle
        #
        # elbow -> shoulder -> hip
        # ----------------------------------------------------

        left_shoulder=calculate_angle(
            landmarks.left_elbow,
            landmarks.left_shoulder,
            landmarks.left_hip
        ),

        right_shoulder=calculate_angle(
            landmarks.right_elbow,
            landmarks.right_shoulder,
            landmarks.right_hip
        ),

        # ----------------------------------------------------
        # Hip angle
        #
        # shoulder -> hip -> knee
        # ----------------------------------------------------

        left_hip=calculate_angle(
            landmarks.left_shoulder,
            landmarks.left_hip,
            landmarks.left_knee
        ),

        right_hip=calculate_angle(
            landmarks.right_shoulder,
            landmarks.right_hip,
            landmarks.right_knee
        ),

        # ----------------------------------------------------
        # Knee angle
        #
        # hip -> knee -> ankle
        # ----------------------------------------------------

        left_knee=calculate_angle(
            landmarks.left_hip,
            landmarks.left_knee,
            landmarks.left_ankle
        ),

        right_knee=calculate_angle(
            landmarks.right_hip,
            landmarks.right_knee,
            landmarks.right_ankle
        ),

        # ----------------------------------------------------
        # Ankle angle
        #
        # knee -> ankle -> forefoot
        # ----------------------------------------------------

        left_ankle=calculate_angle(
            landmarks.left_knee,
            landmarks.left_ankle,
            landmarks.left_foot
        ),

        right_ankle=calculate_angle(
            landmarks.right_knee,
            landmarks.right_ankle,
            landmarks.right_foot
        ),

        # ----------------------------------------------------
        # Trunk inclination from vertical
        #
        # 0 degrees = vertical
        # ----------------------------------------------------

        trunk_inclination=vertical_inclination(
            segments.trunk_center
        ),

        # ----------------------------------------------------
        # Shoulder tilt
        #
        # 0 degrees = horizontal
        # positive / negative tells which side is lower
        # ----------------------------------------------------

        shoulder_tilt=horizontal_tilt(
            segments.shoulder_line
        ),

        # ----------------------------------------------------
        # Hip tilt
        # ----------------------------------------------------

        hip_tilt=horizontal_tilt(
            segments.hip_line
        ),

        # ----------------------------------------------------
        # Shank inclination from vertical
        # ----------------------------------------------------

        left_shank_inclination=vertical_inclination(
            segments.left_shank
        ),

        right_shank_inclination=vertical_inclination(
            segments.right_shank
        ),

        # ----------------------------------------------------
        # Foot inclination from horizontal
        # ----------------------------------------------------

        left_foot_inclination=horizontal_inclination(
            segments.left_foot
        ),

        right_foot_inclination=horizontal_inclination(
            segments.right_foot
        )
    )

    # ========================================================
    # COMPLETE MODEL
    # ========================================================

    return BodyModel(
        frame_number=frame_number,
        time_ms=time_ms,

        landmarks=landmarks,
        virtual_points=virtual_points,
        segments=segments,
        angles=angles
    )