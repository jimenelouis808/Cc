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
}

# --- Sellos -----------------------------------------------------------------
ORING = {
    "face_cs": 1.5,          # MEDIR: sección del O-ring EPDM de cara
    "face_id": 10.0,         # MEDIR: diámetro interior del O-ring de cara
}

# --- Puertos ----------------------------------------------------------------
# Las referencias (docs/referencias.md) usan conectores HPLC de fondo plano 1/4-28 UNF.
# Estandarizar todos los puertos en una misma rosca permite intercambiar RE/CE/flujo.
PORTS = {
    "thread": "1/4-28 UNF",
    "tap_drill_d": 5.5,      # broca para machuelo 1/4-28 (PTFE/PEEK maquinado)
    "flat_bottom_d": 7.0,    # MEDIR: asiento del conector
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
