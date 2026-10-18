# Questo modulo costruisce il modello del corpo relativo a un singolo frame.
# A partire dai landmark MediaPipe crea punti, segmenti, punti virtuali
# e calcola gli angoli e le inclinazioni utilizzate nelle analisi successive.

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

    


# rappresentazione totale della persona al frame n

@dataclass
class BodyModel:

    frame_number: int
    time_ms: int

    landmarks: BodyLandmarks
    virtual_points: VirtualPoints
    segments: BodySegments
    angles: BodyAngles


# GEOMETRY FUNCTIONS

# media fra 2 punti 
def midpoint(point_a: Point, point_b: Point) -> Point:

    return Point(
        x=(point_a.x + point_b.x) / 2,
        y=(point_a.y + point_b.y) / 2,
        z=(point_a.z + point_b.z) / 2,

        #come standard minimo prendo la visibilita minore
        visibility=min(
            point_a.visibility,
            point_b.visibility
        )
    )

#calcola l'angolo su 3 punti 
def calculate_angle(point_a: Point, vertex: Point, point_c: Point) -> float:

    # creo 2 vettori partendo dal solito vertice 
    vector_a = (
        point_a.x - vertex.x,
        point_a.y - vertex.y
    )

    vector_c = (
        point_c.x - vertex.x,
        point_c.y - vertex.y
    )

    # calcolo il prodotto scalare tra due vettori
    dot_product = vector_a[0] * vector_c[0] + vector_a[1] * vector_c[1]
        
    #calcolo la lunghezza dei vettori
    magnitude_a = math.hypot(vector_a[0], vector_a[1])

    magnitude_c = math.hypot(vector_c[0], vector_c[1])

    if magnitude_a == 0 or magnitude_c == 0:
        return 0.0

    #calcolo del coseno dell'angolo
    #da li otteniamo l'arcoseno
    #si converte il risultato in gradi

    cosine = dot_product / (magnitude_a * magnitude_c)

    # forziamo il coseno a stare nei valori (evitiamo il rischio di arrotondamenti)
    cosine = max(-1.0, min(1.0, cosine))

    angle_radians = math.acos(cosine)

    return math.degrees(angle_radians)

def vertical_inclination(segment: Segment) -> float:

    #confrontando i segmenti dx e sx posso calcolare l'inclinazione verticale del segmento rispetto alla verticale
    dx = segment.end.x - segment.start.x
    dy = segment.end.y - segment.start.y

    return math.degrees(math.atan2(abs(dx), abs(dy)))


def horizontal_inclination(segment: Segment) -> float:

    #confrontando i segmenti dx e sx posso calcolare l'inclinazione orizzontale del segmento rispetto all'orizzontale
    dx = (segment.end.x - segment.start.x)
    dy = (segment.end.y - segment.start.y)

    return math.degrees(math.atan2(abs(dy), abs(dx)))


def horizontal_tilt(segment: Segment) -> float:

    #da che lato è inclinato il segmento rispetto all'orizzontale
    dx = (segment.end.x- segment.start.x)
    dy = (segment.end.y - segment.start.y)

    if dx == 0:
        return 90.0

    angle = math.degrees(math.atan2(dy,dx))

    if angle > 90:
        angle -= 180

    elif angle < -90:
        angle += 180

    return angle


#questo metodo si occupa di coordinare e creare 
#l'intera struttura corporea tramite i punti di mediapipe

def build_body_model(mediapipe_landmarks,frame_number=0, time_ms=0) -> BodyModel:

    # legge un landmark trovato e restituisce un point
    def create_point(name):

        landmark = mediapipe_landmarks[name]

        return Point(
            x=landmark["x"],
            y=landmark["y"],
            z=landmark.get("z",0.0),
            visibility=landmark["visibility"]
        )

    #creazione di tutti i landmark necessari
    landmarks = BodyLandmarks(

        naso=create_point("naso"),

        spalla_sinistra=create_point("spalla_sinistra"),
        spalla_destra=create_point("spalla_destra"),

        gomito_sinistro=create_point("gomito_sinistro"),
        gomito_destro=create_point("gomito_destro"),

        polso_sinistro=create_point("polso_sinistro"),
        polso_destro=create_point("polso_destro"),

        anca_sinistra=create_point("anca_sinistra"),
        anca_destra=create_point("anca_destra"),

        ginocchio_sinistro=create_point("ginocchio_sinistro"),
        ginocchio_destro=create_point("ginocchio_destro"),

        caviglia_sinistra=create_point("caviglia_sinistra"),
        caviglia_destra=create_point("caviglia_destra"),

        tallone_sinistro=create_point("tallone_sinistro"),
        tallone_destro=create_point("tallone_destro"),

        avampiede_sinistro=create_point("avampiede_sinistro"),
        avampiede_destro=create_point("avampiede_destro")
    )

    #creazione dei punti calcolati
    centro_spalle = midpoint(landmarks.spalla_sinistra, landmarks.spalla_destra)

    centro_anche = midpoint(landmarks.anca_sinistra, landmarks.anca_destra)

    virtual_points = VirtualPoints(
        centro_spalle=centro_spalle,
        centro_anche=centro_anche
    )

    segments = BodySegments(
        #BRACCIA

        omero_sinistro=Segment(
            landmarks.spalla_sinistra,
            landmarks.gomito_sinistro
        ),
        
        omero_destro=Segment(
            landmarks.spalla_destra,
            landmarks.gomito_destro
        ),

        avambraccio_sinistro=Segment(
            landmarks.gomito_sinistro,
            landmarks.polso_sinistro
        ),

        avambraccio_destro=Segment(
            landmarks.gomito_destro,
            landmarks.polso_destro
        ),

        #TORACE

        linea_spalle=Segment(
            landmarks.spalla_sinistra,
            landmarks.spalla_destra
        ),

        linea_bacino=Segment(
            landmarks.anca_sinistra,
            landmarks.anca_destra
        ),

        lato_tronco_sinistro=Segment(
            landmarks.spalla_sinistra,
            landmarks.anca_sinistra
        ),

        lato_tronco_destro=Segment(
            landmarks.spalla_destra,
            landmarks.anca_destra
        ),

        asse_tronco=Segment(
            centro_anche,
            centro_spalle
        ),

        #GAMBE

        femore_sinistro=Segment(
            landmarks.anca_sinistra,
            landmarks.ginocchio_sinistro
        ),

        femore_destro=Segment(
            landmarks.anca_destra,
            landmarks.ginocchio_destro
        ),

        gamba_sinistra=Segment(
            landmarks.ginocchio_sinistro,
            landmarks.caviglia_sinistra
        ),

        gamba_destra=Segment(
            landmarks.ginocchio_destro,
            landmarks.caviglia_destra
        ),

        #PIEDI

        asse_piede_sinistro=Segment(
            landmarks.tallone_sinistro,
            landmarks.avampiede_sinistro
        ),

        asse_piede_destro=Segment(
            landmarks.tallone_destro,
            landmarks.avampiede_destro
        ),

        asse_caviglia_piede_sinistro=Segment(
            landmarks.caviglia_sinistra,
            landmarks.avampiede_sinistro
        ),

        asse_caviglia_piede_destro=Segment(
            landmarks.caviglia_destra,
            landmarks.avampiede_destro
        )
    )

    angles = BodyAngles(

        #ANGOLO GOMITO SX E DX

        gomito_sinistro=calculate_angle(
            landmarks.spalla_sinistra,
            landmarks.gomito_sinistro,
            landmarks.polso_sinistro
        ),

        gomito_destro=calculate_angle(
            landmarks.spalla_destra,
            landmarks.gomito_destro,
            landmarks.polso_destro
        ),

        #ANGOLO SPALLA DX E SX

        spalla_sinistra=calculate_angle(
            landmarks.gomito_sinistro,
            landmarks.spalla_sinistra,
            landmarks.anca_sinistra
        ),

        spalla_destra=calculate_angle(
            landmarks.gomito_destro,
            landmarks.spalla_destra,
            landmarks.anca_destra
        ),

        # ANGOLO ANCA DX E SX

        anca_sinistra=calculate_angle(
            landmarks.spalla_sinistra,
            landmarks.anca_sinistra,
            landmarks.ginocchio_sinistro
        ),

        anca_destra=calculate_angle(
            landmarks.spalla_destra,
            landmarks.anca_destra,
            landmarks.ginocchio_destro
        ),

        #ANGOLO GINOCCHIO DX E SX

        ginocchio_sinistro=calculate_angle(
            landmarks.anca_sinistra,
            landmarks.ginocchio_sinistro,
            landmarks.caviglia_sinistra
        ),

        ginocchio_destro=calculate_angle(
            landmarks.anca_destra,
            landmarks.ginocchio_destro,
            landmarks.caviglia_destra
        ),

        #ANGOLO CAVIGLIA DX E SX

        caviglia_sinistra=calculate_angle(
            landmarks.ginocchio_sinistro,
            landmarks.caviglia_sinistra,
            landmarks.avampiede_sinistro
        ),

        caviglia_destra=calculate_angle(
            landmarks.ginocchio_destro,
            landmarks.caviglia_destra,
            landmarks.avampiede_destro
        ),

        #INCLINAZIONE TRONCO

        inclinazione_tronco=vertical_inclination(
            segments.asse_tronco
        ),

        #LATO INCLINAZIONE SPALLE

        inclinazione_spalle=horizontal_tilt(
            segments.linea_spalle
        ),

        #LATO INCLINAZIONE ANCHE

        inclinazione_bacino=horizontal_tilt(
            segments.linea_bacino
        ),

        # INCLINAZIONE VERTICALE GAMBE

        inclinazione_gamba_sinistra=vertical_inclination(
            segments.gamba_sinistra
        ),

        inclinazione_gamba_destra=vertical_inclination(
            segments.gamba_destra
        ),

        #INCLINAZIONE ORIZZONTALE PIEDI

        inclinazione_piede_sinistro=horizontal_inclination(
            segments.asse_piede_sinistro
        ),

        inclinazione_piede_destro=horizontal_inclination(
            segments.asse_piede_destro
        )
    )

    # SCHELETRO COMPLETO DEL CORPO

    return BodyModel(
        frame_number=frame_number,
        time_ms=time_ms,

        landmarks=landmarks,
        virtual_points=virtual_points,
        segments=segments,
        angles=angles
    )