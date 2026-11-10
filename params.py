"""Parámetros centrales del proyecto de celdas in situ. Unidades: mm.

Valores marcados con 'MEDIR' son supuestos: confirmar con calibrador o ficha técnica.
"""

# --- Instrumentos -----------------------------------------------------------
XRD = {
    "model": "Bruker D8 Advance (theta-theta)",
    "goniometer_radius": 280.0,      # MEDIR: confirmar radio del goniómetro de tu equipo
    "divergence_deg": 0.3,           # MEDIR: ángulo de la rendija de divergencia usada
    "two_theta_min": 10.0,           # 2θ mínimo de interés (MoSe2 (002) ≈ 13.7°)
    "beam_length_y": 12.0,           # MEDIR: largo de la línea iluminada a lo largo de y
    "corridor_margin": 2.0,          # holgura del corredor del haz a cada lado en y
    "stage_max_height": None,        # MEDIR: altura libre máxima sobre la platina
    "stage_max_footprint": None,     # MEDIR: (ancho, largo) disponibles
}

RAMAN = {
    "model": "Renishaw inVia confocal",
    "objective": "50x LWD",          # MEDIR: leer grabado del objetivo
    "working_distance": 8.0,         # MEDIR: WD en aire (grabado)
    "numerical_aperture": 0.5,       # MEDIR: NA (grabado)
    # El objetivo se modela como cara frontal + nariz cónica + barril.
    "objective_tip_diameter": 12.0,  # MEDIR: Ø de la CARA FRONTAL (no del barril)
    "objective_nose_angle_deg": 35.0,  # MEDIR: semiángulo de la nariz, desde el eje
    "objective_barrel_d": 25.0,      # MEDIR: Ø del barril
    "clearance_margin": 1.0,
    "optical_margin": 0.2,           # holgura radial alrededor del cono de luz
    "window_t": 0.5,                 # espesor de ventana (cuarzo / zafiro)
    "n_window": 1.46,                # cuarzo (zafiro 1.77): el menor es el caso conservador
    "n_electrolyte": 1.33,
}

# --- Electrodos -------------------------------------------------------------
ELECTRODES = {
    # RE Ag/AgCl Gamry (medido con calibrador, 2026-11-10)
    "RE_body_d": 9.0,        # cuerpo de vidrio
    "RE_tip_d": 4.0,         # punta (frita)
    "RE_tip_l": 20.0,        # largo de la punta bajo el hombro
    "RE_l": 75.0,            # largo total
    # CE
    "CE_rod_d": 6.0,         # electrodo de Pt con cuerpo (medido)
    "CE_rod_wire_l": 35.0,   # alambre de Pt que sobresale del cuerpo (medido)
    "CE_wire_d": 0.406,      # alambre suelto Gamry 935-00056 (0.406 mm × 15 cm)
    "CE_sleeve_d": 3.0,      # funda (PTFE/vidrio) para el alambre suelto
    # WE glassy carbon convencionales (medidos)
    "GC_disk_d": 3.0,
    "GC_body_d": 7.0,        # Gamry 932-00022: disco 3 mm, cuerpo 7 mm
    "GC_options": ({"disk_d": 3.0, "body_d": 7.0, "l": None, "ref": "Gamry 932-00022"},
                   {"disk_d": 5.0, "body_d": 10.0, "l": 60.0, "ref": "Corrtest CS920"}),
}

WE_FLAT = {
    "sample_w": 15.0,        # lado del sustrato cuadrado
    "sample_t": 1.0,         # espesor del sustrato
    "exposed_d": 9.5,        # diámetro expuesto = ID del O-ring sobre la muestra.
                             # ≥ 9.5 para que la nariz de lid_raman deje pasar NA 0.5
                             # con pared de 2 mm (ver checks/walls.py: lid_checks)
}

# WE C: muestra en pinza de Pt (como la celda de PTFE de docs/referencias.md, R1).
# El portaelectrodo entra por un puerto lateral 1/4-28; la muestra (tira) queda
# horizontal, cara arriba, apoyada en un pedestal que fija su cara en z = 0.
# Portaelectrodos (medidos, 2026-11-10). La cabeza se describe como un bloque:
#   head_l × head_w, y head_up / head_down sobre y bajo el plano medio de la muestra
#   (que coincide con el eje del cuerpo). grip_l = cuánto entra la muestra en la cabeza.
CLAMPS = {
    # Corrtest CS945 (3.2.01.009): varilla de PEEK Ø6 × 60 con ranura en la punta y un
    # tornillo transversal; 10–11 mm de alto con el tornillo. Se monta con el tornillo
    # hacia abajo: sobre la muestra sólo queda el radio de la varilla. Sin el tornillo,
    # la varilla pasa por un puerto radial de Ø6.
    "CS945": {"rod_d": 6.0, "rod_l": 60.0, "head_l": 8.0, "head_w": 6.0,
              "head_up": 3.0, "head_down": 7.5, "grip_l": 5.0,   # grip_l: MEDIR (ranura)
              "max_sample_t": 1.5,                                # MEDIR: ancho de la ranura
              "mount": "through_port"},
    # Corrtest double knob (3.2.01.366): cuerpo Ø6, cabeza 18 × 21 × 15 mm con perillas.
    # No pasa por un puerto: se coloca desde arriba con la celda abierta.
    "double_knob": {"rod_d": 6.0, "rod_l": None, "head_l": 21.0, "head_w": 18.0,
                    "head_up": 10.0, "head_down": 5.0,            # MEDIR: dónde queda la muestra
                    "grip_l": 10.0, "max_sample_t": 2.0,          # MEDIR
                    "mount": "from_inside"},
}

WE_CLAMP = dict(CLAMPS["CS945"], **{
    "model": "CS945",
    "sample_w": 8.0,         # tira: ancho
    "sample_l": 14.0,        # tira: largo total (incluye el agarre)
    "sample_t": 1.0,         # MEDIR: espesor de tu muestra
    "clip_clearance": 0.5,   # cabeza ↔ émbolo de la tapa
    "pedestal_d": 4.0,       # apoyo bajo el punto de medición (0 = sin pedestal)
    "floor_clearance": 1.0,  # cabeza ↔ piso de la cámara
})

# Celda XRD (θ-θ): el haz va a lo largo de x. La muestra queda bajo una película
# (Kapton en ácido, PP/PE en KOH) sellada con O-ring bajo un marco tensor. La pinza
# de Pt (WE_CLAMP) toma una lengüeta flexible en un bolsillo del lado +y, 4–5 mm
# bajo la cara de la muestra: la propia muestra si es flexible, o una tira de
# Pt/Ti bajo la muestra si es rígida.
WE_XRD = {
    "sample_x": 18.0,        # a lo largo del haz: debe cubrir la huella a 2θ_min
    "sample_y": 18.0,        # perpendicular al haz: ≥ corredor del haz
    "sample_t": 1.0,         # MEDIR
    "pocket_clearance": 0.2,
    "tab_w": 5.0,            # lengüeta de contacto (Pt/Ti, o la muestra misma)
    "tab_t": 0.1,            # MEDIR
    "ramp_l": 6.0,           # tramo en que la lengüeta baja hasta la pinza
    "clip_depth": 6.0,       # eje de la pinza bajo la cara (CS945; double knob: 12)
}

CHAMBER_XRD = {
    "slot_w": 2.0,           # ranura reservorio → capa delgada, junto a la muestra (−y)
    "reservoir_l": 12.0,     # largo del reservorio interno en y
    "reservoir_margin": 1.5, # el reservorio cubre las bocas de los puertos −y con este margen
    "reservoir_z": (-14.5, -3.5),  # piso y techo: bajo el cordón quedan ≥ 2 mm de pared
    "port_wall": 8.0,        # pared entre reservorio/bolsillo y cara lateral (rosca 7 + 1)
    "cover_open": 3.0,       # tramo del bolsillo sin tapa junto a la muestra (la lengüeta baja)
    "cover_t": 1.5,          # tapa del bolsillo de la pinza, al ras
    "cover_ledge": 1.5,
    "floor_clearance": 1.0,
    "pocket_margin": 1.5,    # holgura lateral de la pinza en su bolsillo
}

LID_XRD = {
    # Marco tensor de la película. Los lados que cruzan el haz (±x) van biselados:
    # su borde interior es delgado y sube con la pendiente que permite el haz.
    "t": 5.0,                # espesor fuera del corredor (aluminio): rígido entre tornillos
    "edge_t": 0.3,           # espesor del borde interior biselado
    "bevel_deg": 4.5,        # pendiente del bisel en los lados ±x (< la del límite del haz)
    "groove_min_t": 1.0,     # el cordón va donde el marco ya tiene este espesor
    "clearance_margin": 0.2, # holgura vertical bajo el límite del haz
    "window_margin": 1.0,    # ventana libre alrededor de muestra, ranura y lengüeta
    "film_cs": 1.5,          # MEDIR: cordón EPDM bajo la película
    "screw_d": 3.0,          # M3 avellanado
    "screw_head_d": 6.0,
    "screw_pitch": 22.0,     # separación máxima entre tornillos del marco
    "edge_margin": 4.0,      # tornillo ↔ borde exterior
}

# Puertos de la celda XRD: sólo en caras ±y (nada sobresale a lo largo del haz).
XRD_PORTS = (
    {"name": "WE", "face": "+y", "offset": 0.0, "z": -WE_XRD["clip_depth"], "kind": "radial",
     "rod_d": WE_CLAMP["rod_d"]},
    {"name": "CE", "face": "-y", "offset": -18.0, "z": -8.0, "kind": "radial",
     "rod_d": ELECTRODES["CE_rod_d"]},
    {"name": "RE", "face": "-y", "offset": -6.0, "z": -8.0, "kind": "radial",
     "rod_d": ELECTRODES["RE_body_d"]},
    {"name": "inlet", "face": "-y", "offset": 6.0, "z": -8.0, "kind": "fitting",
     "through_d": 2.0},
    {"name": "outlet", "face": "-y", "offset": 18.0, "z": -8.0, "kind": "fitting",
     "through_d": 2.0},
)

# Electrodeposición acústica (Chladni): el WE es una placa cuadrada empotrada entre
# dos marcos atornillados, excitada por un piezo pegado por debajo fuera del centro.
# Plano z = 0 = cara superior de la placa (el WE).
CHLADNI = {
    "plate_material": "Ti grado 2",  # clave de sim/chladni_estimate.MATERIALS
    "plate_L": 35.0,         # lado libre entre líneas de empotramiento
    "plate_h": 0.3,          # MEDIR: espesor de la placa
    "clamp_w": 6.0,          # ancho de la franja empotrada
    "plate_preload": 0.05,   # el alojamiento de la placa es así de menos profundo que ella
    "tab_w": 8.0,            # lengüeta de contacto del WE (la toma la pinza de Pt, en seco)
    "tab_out": 12.0,         # cuánto sale la lengüeta del borde de los marcos
    "piezo_d": 15.0,         # MEDIR: disco piezo
    "piezo_t": 0.6,          # MEDIR
    "piezo_center": (-8.75, -8.75),  # centro del piezo (≈ L/4 de cada borde, fuera de los ejes)
    "piezo_margin": 1.0,     # piezo ↔ línea de empotramiento
    "seal_cs": 1.5,          # MEDIR: cordón EPDM sobre la placa
    "seal_inset": 0.5,       # cordón ↔ línea de empotramiento (zona casi sin desplazamiento)
    "seal_corner_r": 4.0,
    "relief_gap": 0.3,       # el marco superior no toca la placa dentro del empotramiento
    "bottom_t": 8.0,         # marco inferior (aluminio con Kapton, o PEEK): deja libre el piezo
    "electrolyte_h": 8.0,    # altura de electrolito sobre la placa (carga de masa: baja los modos)
    "headspace": 4.0,
    "ce_gap": 6.0,           # WE ↔ CE paralelo
    "ce_t": 1.0,             # MEDIR: malla de Pt o placa de grafito
    "ce_ledge": 1.5,         # escalón de la pared que sostiene el CE
    "re_corner": (1, 1),     # esquina (signos x, y) donde baja la RE con Luggin
    "re_inset": 4.0,         # punta de la RE ↔ esquina del área expuesta
    "re_seat_d": 14.0,       # resalte de la tapa donde apoya el hombro de la RE (Ø9)
    "luggin_h": 1.5,         # punta de la RE sobre la placa
    "bolts_per_side": 4,
    "bolt_margin": 4.0,      # tornillo ↔ borde de la placa
    "lid_t": 6.0,
}

# CE de la celda Chladni: el alambre de la malla sale por la pared −x con conector y funda.
CHLADNI_CE_PORT = {"name": "CE", "face": "-x", "offset": 0.0, "z": None, "kind": "fitting",
                   "through_d": ELECTRODES["CE_sleeve_d"]}   # z = plano medio del CE

# Puertos de la tapa de la celda Chladni (verticales, desde arriba).
CHLADNI_LID_PORTS = (
    {"name": "inlet", "xy": (-10.0, 10.0), "kind": "fitting", "through_d": 2.0},
    {"name": "vent", "xy": (10.0, -10.0), "kind": "fitting", "through_d": 2.0},
)

# EPD / DEP selectivo: chip de vidrio con electrodos interdigitados (sputtering con
# máscara de sombra) cuyo gap baja por segmentos a lo largo del canal. Plano z = 0 =
# cara del chip con los electrodos. El canal (a lo largo de x) lo define una junta
# entre el chip y un vidrio superior perforado; un marco con ventana lo aprieta.
EPD = {
    "slide": (75.0, 25.0, 1.0),     # MEDIR: portaobjetos (x, y, espesor)
    "cover": (48.0, 25.0, 1.0),     # MEDIR: vidrio superior, perforado en los extremos del canal
    "cover_hole_d": 1.0,            # MEDIR: agujeros del vidrio superior
    "gasket_t": 0.25,               # MEDIR: junta (PTFE o FFKM): altura del canal sin comprimir
    "gasket_compression": 0.10,     # fracción; el tope duro del marco la fija
    "channel_w": 3.0,
    "channel_l": 42.0,              # extremos redondeados; los agujeros van en los centros
    "gasket_seal_min": 2.0,         # junta mínima entre el canal y cualquier borde o bus
    # electrodos interdigitados por segmentos (dedos a lo largo de y, cruzando el canal)
    "gaps": (0.50, 0.35, 0.25, 0.15, 0.10),   # gap de cada segmento, de la entrada a la salida
    "segment_l": 4.0,
    "segment_spacing": 1.0,
    "finger_w": 0.20,
    "finger_overlap": 0.5,          # cuánto pasan los dedos bajo la junta a cada lado del canal
    "bus_w": 1.0,
    "pad": (10.0, 8.0),             # pads de contacto en el extremo +x (x, y)
    "pad_gap": 2.0,
    "clip_l": 8.0,                  # largo de pad que necesita la pinza
    # máscara de sombra (lámina cortada por láser): límites del proveedor
    "mask_min_feature": 0.10,       # MEDIR: abertura mínima
    "mask_min_web": 0.10,           # MEDIR: puente mínimo de lámina entre aberturas
    "mask_max_aspect": 45.0,        # MEDIR: largo / ancho máximo de un puente sin apoyo
    "mask_border": 5.0,
    # soporte
    "base_t": 5.0,
    "frame_t": 9.0,                 # ≥ rosca del conector (7) + asiento
    "frame_side": 6.5,              # franjas del marco fuera del ancho del portaobjetos
    "frame_chip_clearance": 0.3,    # el marco no toca el portaobjetos (sólo base y vidrio)
    "window_margin": 2.0,
    "port_oring": (2.0, 1.0),       # MEDIR: O-ring de cara sobre cada agujero (ID, cs)
    "screw_x": (-20.0, 0.0, 20.0),
    # operación (para la estimación de sim/epd_dep.py)
    "flow_ul_min": 10.0,
    "solvents": ("etanol", "IPA", "acetona"),
    "wetted_materials": ("vidrio", "Pt/Au", "PTFE", "FFKM", "PEEK"),
}

# Estimación de captura DEP (sim/epd_dep.py): orden de magnitud, no FEM.
EPD_SIM = {
    "medium": "etanol",
    "viscosity": 1.08e-3,     # Pa s (etanol, 25 °C)
    "eps_r": 24.5,
    "rho_medium": 789.0,      # kg/m³
    "v_rms": 10.0,            # MEDIR: tensión AC entre peines
    "re_k": 1.0,              # MEDIR: Re[K] del factor de Clausius-Mossotti (CNT metálico ≫ 1)
    "particle_l_um": 2.0,     # largo de los CNT
    "particle_d_nm": (5.0, 10.0, 20.0, 40.0),
    "rho_particle": 1300.0,
}

# Puertos del marco EPD: verticales, sobre los agujeros del vidrio superior.
EPD_PORTS = (
    {"name": "inlet", "x_sign": -1, "kind": "fitting", "through_d": 1.0},
    {"name": "outlet", "x_sign": 1, "kind": "fitting", "through_d": 1.0},
)

# Materiales que un disolvente ataca (no deben tocar el líquido)
SOLVENT_ATTACKS = {
    "acetona": ("PMMA", "ABS", "PC", "PS", "FKM"),
    "etanol": ("PMMA",),
    "IPA": ("PMMA",),
    "KOH": ("PLA", "PETG", "PC", "Kapton"),
}

# --- Celda ------------------------------------------------------------------
CELL = {
    "thin_gap_raman": 0.3,   # capa de electrolito entre WE y ventana (Raman), nominal
    "gap_range_raman": (0.1, 0.5),  # alcanzable con lainas bajo la tapa
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
    "lid_insert_xy": 16.0,   # insertos en (±xy, ±xy)
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
    "h": 15.0,               # altura (de z = 0 a z = h): el O-ring de la RE Ø9 cabe con pared
    "floor_lip": 2.0,        # material sobre el escalón del O-ring (rigidez del piso)
    "well_d": 32.0,          # pozo/reservorio: aloja el émbolo de lid_raman con holgura
}

LID_RAMAN = {
    # Tapa de una pieza: placa sobre el anillo + émbolo cónico hueco + nariz con
    # ventana pegada al ras. La placa apoya sobre lainas (arandelas DIN 988 3×6
    # en los 4 tornillos): gap = thin_gap_raman + (laina − shim_nominal).
    "plate_t": 5.0,
    "shim_nominal": 0.2,     # lainas con las que gap = thin_gap_raman
    "nose_clearance": 0.25,  # holgura radial nariz–agujero del piso
    "shoulder_clearance": 0.5,  # hombro de la nariz sobre el piso, a gap nominal
    "well_clearance": 0.5,   # holgura radial mínima émbolo–pozo
    "window_d": 6.0,         # MEDIR: ventana Ø6 × window_t, pegada al ras
    "window_recess_wall_min": 0.5,
    "window_land_min": 0.5,  # apoyo de la ventana fuera de la apertura
    "gland_width_factor": 1.7,  # ranura más ancha: la profundidad se reduce por la laina
    "oring_compression": (0.10, 0.40),  # rango aceptable en todo el recorrido de lainas
    "gland_fill_max": 0.85,
    "port_clearance": 1.0,   # émbolo ↔ bocas de los puertos del anillo
    "bolt_engagement_min": 3.0,
}

# Puertos de la cámara para pinza. Misma cara superior que el anillo (la tapa Raman
# sirve igual). El WE entra por −x a la altura del plano medio de la muestra.
CLAMP_PORTS = (
    {"name": "WE", "face": "-x", "offset": 0.0, "z": -WE_CLAMP["sample_t"] / 2, "kind": "radial",
     "rod_d": WE_CLAMP["rod_d"]},
    {"name": "inlet", "face": "+y", "offset": 6.0, "z": 6.5, "kind": "fitting", "through_d": 2.0},
    {"name": "CE", "face": "+y", "offset": -6.0, "z": 6.5, "kind": "radial",
     "rod_d": ELECTRODES["CE_rod_d"]},
    {"name": "outlet", "face": "+x", "offset": 0.0, "z": 9.0, "kind": "fitting", "through_d": 2.0},
    {"name": "RE", "face": "-y", "offset": 0.0, "z": 7.8, "kind": "radial",
     "rod_d": ELECTRODES["RE_body_d"]},
)

# Puertos del anillo. face: cara por la que entra; offset: desplazamiento sobre la
# cara; z: altura del eje. kind 'fitting' = 1/4-28 de fondo plano (machuelar tras
# imprimir); kind 'radial' = vástago con O-ring radial (RE).
RING_PORTS = (
    {"name": "inlet", "face": "-x", "offset": 0.0, "z": 5.5, "kind": "fitting", "through_d": 2.0},
    {"name": "outlet", "face": "+x", "offset": 0.0, "z": 9.0, "kind": "fitting", "through_d": 2.0},
    {"name": "CE", "face": "+y", "offset": 0.0, "z": 6.5, "kind": "radial",
     "rod_d": ELECTRODES["CE_rod_d"]},
    {"name": "RE", "face": "-y", "offset": 0.0, "z": 7.8, "kind": "radial",
     "rod_d": ELECTRODES["RE_body_d"]},
)

# --- Sellos -----------------------------------------------------------------
ORING = {
    "face_cs": 1.5,          # MEDIR: sección del O-ring EPDM de cara
    "face_id": 10.0,         # MEDIR: diámetro interior del O-ring de cara
    # O-ring que sella sobre la muestra plana: su ID = WE_FLAT["exposed_d"]
    "sample_cs": 1.0,        # MEDIR (9.5 × 1.0 EPDM: cabe en la muestra de 15 mm)
    # O-ring radial sobre el cuerpo de la RE
    "re_cs": 1.5,            # MEDIR
    "re_groove_from_face": 4.0,
    # O-ring de cara de la tapa Raman (sella contra la cara superior del anillo)
    "lid_id": 35.0,          # MEDIR
    "lid_cs": 1.5,           # MEDIR
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
