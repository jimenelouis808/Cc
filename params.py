"""Parámetros centrales del proyecto de celdas in situ. Unidades: mm.

Valores marcados con 'MEDIR' son supuestos: confirmar con calibrador o ficha técnica.
"""

# --- Instrumentos -----------------------------------------------------------
XRD = {
    "model": "Bruker D8 Advance (theta-theta)",
    "goniometer_radius": 280.0,      # MEDIR: confirmar radio del goniómetro de tu equipo
    "divergence_deg": 0.3,           # MEDIR: ángulo de la rendija de divergencia usada
    "two_theta_min": 10.0,           # 2θ mínimo de interés (MoSe2 (002) ≈ 13.7°)
    "stage_max_height": None,        # MEDIR: altura libre máxima sobre la platina
    "stage_max_footprint": None,     # MEDIR: (ancho, largo) disponibles
}

RAMAN = {
    "model": "Renishaw inVia confocal",
    "objective": "50x LWD",          # MEDIR: leer grabado del objetivo
    "working_distance": 8.0,         # MEDIR: WD del objetivo de larga distancia
    "objective_tip_diameter": 25.0,  # MEDIR: diámetro de la punta del objetivo
    "clearance_margin": 1.0,
    "window_t": 0.5,                 # espesor de ventana (cuarzo / zafiro)
}

# --- Electrodos -------------------------------------------------------------
ELECTRODES = {
    "RE_body_d": 6.0,        # MEDIR: diámetro del cuerpo Ag/AgCl Camray
    "CE_wire_d": 0.5,        # MEDIR: alambre de Pt
    "CE_sleeve_d": 3.0,      # funda (PTFE/vidrio) por donde pasa el alambre
    "GC_disk_d": 3.0,        # disco glassy carbon
    "GC_body_d": 6.0,        # MEDIR: diámetro exterior del cuerpo del electrodo GC
}

WE_FLAT = {
    "sample_w": 15.0,        # lado del sustrato cuadrado
    "sample_t": 1.0,         # espesor del sustrato
    "exposed_d": 6.0,        # diámetro expuesto = ID del O-ring de cara
}

# --- Celda ------------------------------------------------------------------
CELL = {
    "thin_gap_raman": 0.3,   # capa de electrolito entre WE y ventana (Raman)
    "thin_gap_xrd": 0.05,    # capa durante medición XRD
    "reservoir_depth": 8.0,  # profundidad del reservorio lateral (RE/CE)
    "luggin_to_we": 2.0,     # distancia punta Luggin – borde del área expuesta
    "min_wall": 2.0,         # pared mínima mojada
    "bolt": "M3",
    "bolt_clearance_d": 3.4,
    "min_wall_dry": 1.2,     # pared mínima seca (entre agujeros, a caras exteriores)
    # huella común de base, anillo y tapas
    "body_w": 54.0,
    "corner_chamfer": 5.0,
    # tornillería anillo → base (cabeza arriba, tuerca embebida abajo)
    "bolt_inset": 6.0,       # centro del tornillo a la cara lateral
    "bolt_head_d": 5.5,      # M3 cabeza cilíndrica
    "bolt_head_h": 3.0,
    "nut_af": 5.5,           # M3, entre caras
    "nut_h": 2.4,
    # pines guía Ø3 (posiciones asimétricas: el anillo solo entra en una orientación)
    "pin_d": 3.0,
    "pin_depth": 4.5,       # por pieza: pin Ø3×8
    "pins": ((12.0, -22.0), (-22.0, 12.0)),
    # insertos térmicos M3 en la cara superior del anillo, para las tapas
    "insert_d": 4.0,         # MEDIR: Ø de agujero que pide el inserto
    "insert_depth": 5.0,
    "lid_insert_xy": 15.0,   # insertos en (±xy, ±xy)
}

BASE_FLAT = {
    "t": 6.0,                # espesor (de z = -t a z = 0)
    "pocket_clearance": 0.2, # holgura por lado del alojamiento de la muestra
    "corner_relief_d": 1.5,  # alivios en las esquinas del alojamiento
    "contact_w": 5.0,        # ranura para la lámina de contacto (Pt / Cu / Ni)
    "contact_t": 0.3,        # bajo la muestra, sale por la cara -x
    "eject_d": 3.0,          # agujero central para empujar la muestra desde abajo
}

CHAMBER_RING = {
    "h": 13.0,               # altura (de z = 0 a z = h)
    "floor_lip": 2.0,        # material sobre el escalón del O-ring (rigidez del piso)
    "well_d": 28.0,          # pozo/reservorio: ≥ punta del objetivo + 2·margen
}

# Puertos del anillo. face: cara por la que entra; offset: desplazamiento sobre la
# cara; z: altura del eje. kind 'fitting' = 1/4-28 de fondo plano (machuelar tras
# imprimir); kind 'radial' = vástago con O-ring radial (RE).
RING_PORTS = (
    {"name": "inlet", "face": "-x", "offset": 0.0, "z": 5.5, "kind": "fitting", "through_d": 2.0},
    {"name": "outlet", "face": "+x", "offset": 0.0, "z": 8.0, "kind": "fitting", "through_d": 2.0},
    {"name": "CE", "face": "+y", "offset": 0.0, "z": 6.0, "kind": "fitting",
     "through_d": ELECTRODES["CE_sleeve_d"]},
    {"name": "RE", "face": "-y", "offset": 0.0, "z": 7.0, "kind": "radial"},
)

# --- Sellos -----------------------------------------------------------------
ORING = {
    "face_cs": 1.5,          # MEDIR: sección del O-ring EPDM de cara
    "face_id": 10.0,         # MEDIR: diámetro interior del O-ring de cara
    # O-ring que sella sobre la muestra plana: su ID = WE_FLAT["exposed_d"]
    "sample_cs": 1.5,        # MEDIR
    # O-ring radial sobre el cuerpo de la RE
    "re_cs": 1.5,            # MEDIR
    "re_groove_from_face": 4.0,
}

# --- Puertos ----------------------------------------------------------------
# Las referencias (docs/referencias.md) usan conectores HPLC de fondo plano 1/4-28 UNF.
# Estandarizar todos los puertos en una misma rosca permite intercambiar RE/CE/flujo.
PORTS = {
    "thread": "1/4-28 UNF",
    "tap_drill_d": 5.5,      # broca para machuelo 1/4-28 (PTFE/PEEK maquinado)
    "flat_bottom_d": 7.0,    # MEDIR: asiento del conector
    "thread_depth": 7.0,     # profundidad roscada + asiento
    "fitting_pitch_min": 12.0,  # separación mínima entre conectores (llave / tuerca)
}

# --- Fabricación ------------------------------------------------------------
FDM = {
    "hole_offset": 0.25,     # calibrar con cupón
    "layer_h": 0.2,
}

COUPON = {
    "hole_nominals": (3.0, 6.0, 8.0),             # Ø nominales a calibrar
    "hole_offsets": (0.0, 0.1, 0.2, 0.3, 0.4),    # sobremedida modelada
    "groove_depth_factors": (0.70, 0.75, 0.80),   # profundidad / cs
    "pitch": 14.0,           # separación entre centros de agujeros
    "margin": 6.0,           # borde de la placa
    "thickness": 5.0,
    "mark_d": 1.2,           # hoyuelos que numeran las columnas
    "mark_depth": 0.4,
    "corner_chamfer": 4.0,   # esquina biselada = origen de la lectura
}

MATERIALS = {
    "body_print": "PP",
    "body_machined": ("PTFE", "PEEK"),
    "oring": "EPDM",
    "window_raman": {"H2SO4": "cuarzo", "KOH": "zafiro"},
    "window_xrd": {"H2SO4": "Kapton", "KOH": "pelicula PP/PE"},
}


# --- Ayudas -----------------------------------------------------------------
def oring_gland(cs, kind="face"):
    """Dimensiones de partida de ranura para O-ring de sección `cs`.

    kind='face'   -> sello axial estático (cara)
    kind='radial' -> sello radial estático sobre vástago
    Devuelve dict con profundidad y ancho de ranura. Verificar contra manual Parker.
    """
    if kind == "face":
        return {"depth": round(0.75 * cs, 3), "width": round(1.5 * cs, 3)}
    if kind == "radial":
        return {"depth": round(0.80 * cs, 3), "width": round(1.4 * cs, 3)}
    raise ValueError("kind debe ser 'face' o 'radial'")


def hole(d_nominal):
    """Diámetro a modelar para que el agujero impreso salga en d_nominal."""
    return d_nominal + FDM["hole_offset"]
