"""Generator for the bundled reference CIFs.

Kept in the repository so the provenance of every bundled structure is
auditable and so the library can be extended the same way it was built.
It is not imported by the package: the shipped ``.cif`` files are the
data, and this is how they were made.

Each structure is written with the CLOSED symmetry group, so the files
are complete CIFs that any other program can read. What is listed here
are only the generators, because listing the 192 operations of Fd-3m by
hand is how mistakes happen — and the group order is checked on closure,
which catches a wrong generator immediately.

Two independent checks were run on the output and both are cheap to
repeat: the crystallographic density of every entry against the
literature value (it validates the cell and the cell contents at once),
and the position of the strongest reflection against its powder card.

Usage:  python _generar_referencias.py
"""
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from ramancarbon.xrd.structure import Lattice, Site, Crystal
from ramancarbon.xrd.cif import write_cif

P4nmm  = ["-y+1/2,x,z", "-x,y+1/2,-z", "-x,-y,-z"]
P63mmc = ["x-y,x,z+1/2", "-x,-y,z+1/2", "y,x,-z", "-x,-y,-z"]
Pnnm   = ["-x,-y,z", "-x+1/2,y+1/2,-z+1/2", "-x,-y,-z"]
P3121  = ["-y,x-y,z+1/3", "y,x,-z"]
Pnma   = ["-x+1/2,-y,z+1/2", "-x,y+1/2,-z", "-x,-y,-z"]
Fd3m   = ["-x+3/4,-y+1/4,z+1/2", "-x+1/4,y+1/2,-z+3/4", "z,x,y",
          "y+3/4,x+1/4,-z+1/2", "-x,-y,-z", "x,y+1/2,z+1/2", "x+1/2,y,z+1/2"]
R3c    = ["-y,x-y,z", "y,x,-z+1/2", "-x,-y,-z", "x+2/3,y+1/3,z+1/3"]
Im3m   = ["z,x,y", "-y,x,z", "y,x,-z", "-x,-y,-z", "x+1/2,y+1/2,z+1/2"]
P21c   = ["-x,y+1/2,-z+1/2", "-x,-y,-z"]

REFS = [
 dict(name="grafito_2H", sg="P6_3/mmc", ops=P63mmc, formula="C",
      lat=(2.4640, 2.4640, 6.7110, 90, 90, 120),
      sites=[("C", (0.0, 0.0, 0.25), "C1"), ("C", (1/3, 2/3, 0.25), "C2")],
      conf="high",
      src="Trucano & Chen, Nature 258 (1975) 136",
      note="Grafito hexagonal. Como todo material laminar, sufre orientación preferente severa sobre 001 en un portamuestras de carga frontal; refínala antes de cuantificar. La 002 a 26.5° es la reflexión de referencia: su posición da el espaciado entre láminas (3.356 Å en grafito perfecto) y su anchura, el tamaño de cristalito a lo largo de c. En carbono turbostrático se desplaza a ángulos menores (d hasta 3.44 Å) y se ensancha muchísimo; eso NO es grafito con la celda cambiada, es desorden de apilamiento, y ajustarlo con esta fase da un parámetro c sin sentido físico."),
 dict(name="FeSe_tetragonal", sg="P4/nmm", ops=P4nmm, formula="FeSe",
      lat=(3.7734, 3.7734, 5.5258, 90, 90, 90),
      sites=[("Fe", (0.75, 0.25, 0.0), "Fe1"), ("Se", (0.25, 0.25, 0.2672), "Se1")],
      conf="high",
      src="Margadonna et al., Chem. Commun. (2008) 5607 (tipo PbO, beta-FeSe)",
      note="La fase superconductora (Tc ~ 8 K). Reflexión 101 a 28.6°, la más intensa; 001 a 16.0° y 112 a 47.4°. La z del selenio (0.2672) controla la altura del anión sobre el plano de hierro y es el parámetro que más afecta a las intensidades relativas: si refinas, refínalo."),
 dict(name="FeSe_hexagonal", sg="P6_3/mmc", ops=P63mmc, formula="FeSe",
      lat=(3.6370, 3.6370, 5.9570, 90, 90, 120),
      sites=[("Fe", (0.0, 0.0, 0.0), "Fe1"), ("Se", (1/3, 2/3, 0.25), "Se1")],
      conf="medium",
      src="Estructura tipo NiAs (delta-FeSe); parámetros de celda de la literatura de FeSe hexagonal",
      note="La fase NO superconductora. Es lo que zanja la duda que el Raman deja abierta: su 101 está a 32.4° y la del beta tetragonal a 28.6°, catorce veces la anchura de un pico. OJO: esta fase es deficiente en hierro (Fe(1-x)Se, y a composiciones concretas ordena como Fe7Se8 o Fe3Se4 con superestructura). Si el ajuste pide una ocupación de hierro por debajo de 1, no es un error del programa."),
 dict(name="FeSe2_marcasita", sg="Pnnm", ops=Pnnm, formula="FeSe2",
      lat=(4.8000, 5.7830, 3.5830, 90, 90, 90),
      sites=[("Fe", (0.0, 0.0, 0.0), "Fe1"), ("Se", (0.2136, 0.3733, 0.0), "Se1")],
      conf="medium",
      src="Tipo marcasita (FeS2 marcasita) con los parámetros de celda del FeSe2",
      note="Producto de exceso de selenio. Reflexiones calculadas más intensas: 111 a 34.9°, 101 a 31.1° y 120 a 37.6°. Caen encima de las del beta-FeSe (28.6 y 37.4°): si tu síntesis pudo dar los dos, hace falta el ajuste completo del patrón, no una comparación de picos."),
 dict(name="Se_trigonal", sg="P3_121", ops=P3121, formula="Se",
      lat=(4.3662, 4.3662, 4.9536, 90, 90, 120),
      sites=[("Se", (0.2254, 0.0, 1/3), "Se1")],
      conf="high",
      src="Cherin & Unger, Acta Cryst. B 28 (1972) 313",
      note="Selenio gris, la forma estable. Reflexión 100 a 23.5° y 101 a 29.7°, esta última la más intensa. Selenio elemental en un difractograma significa precursor segregado, no dopado."),
 dict(name="Fe_alfa", sg="Im-3m", ops=Im3m, formula="Fe",
      lat=(2.8665, 2.8665, 2.8665, 90, 90, 90),
      sites=[("Fe", (0.0, 0.0, 0.0), "Fe1")],
      conf="high",
      src="Parámetro de red del hierro alfa a temperatura ambiente",
      note="Hierro metálico (ferrita). 110 a 44.7°, 200 a 65.0°, 211 a 82.3°. Es Raman-inactivo en primer orden, así que DRX es la ÚNICA de las dos técnicas que lo ve: si el Raman no encuentra hierro, no significa que no lo haya."),
 dict(name="Fe3C_cementita", sg="Pnma", ops=Pnma, formula="Fe3C",
      lat=(5.0896, 6.7443, 4.5248, 90, 90, 90),
      sites=[("Fe", (0.036, 0.25, 0.852), "Fe1"),
             ("Fe", (0.186, 0.063, 0.328), "Fe2"),
             ("C", (0.877, 0.25, 0.444), "C1")],
      conf="medium",
      src="Estructura tipo cementita; coordenadas de la literatura de Fe3C",
      note="La fase que suele estar dentro de las nanopartículas encapsuladas en las puntas de los nanotubos crecidos con hierro. Su grupo intenso de reflexiones cae entre 42 y 46°, solapando con la 110 del hierro alfa a 44.7°: separarlos exige Rietveld, no lectura de picos."),
 dict(name="Fe3O4_magnetita", sg="Fd-3m", ops=Fd3m, formula="Fe3O4",
      lat=(8.3960, 8.3960, 8.3960, 90, 90, 90),
      sites=[("Fe", (0.125, 0.125, 0.125), "FeA"),
             ("Fe", (0.5, 0.5, 0.5), "FeB"),
             ("O", (0.2549, 0.2549, 0.2549), "O1")],
      conf="high",
      src="Espinela inversa, origen 2; Fleet, Acta Cryst. B 37 (1981) 917",
      note="Magnetita. 311 a 35.5°, la más intensa. AVISO: la maghemita gamma-Fe2O3 tiene prácticamente la misma celda y un patrón casi idéntico; con un difractómetro de laboratorio no se separan con fiabilidad. Si necesitas distinguirlas, hace falta la posición exacta del parámetro de red (8.396 frente a 8.35 Å) medida con patrón interno, o Mössbauer."),
 dict(name="Fe2O3_hematita", sg="R-3c", ops=R3c, formula="Fe2O3",
      lat=(5.0356, 5.0356, 13.7489, 90, 90, 120),
      sites=[("Fe", (0.0, 0.0, 0.3553), "Fe1"),
             ("O", (0.3059, 0.0, 0.25), "O1")],
      conf="high",
      src="Blake et al., Am. Mineral. 51 (1966) 123 (ejes hexagonales)",
      note="Hematita. 104 a 33.2° y 110 a 35.6°, las dos más intensas. Es el producto final de oxidación de casi cualquier residuo de catalizador de hierro."),
 dict(name="Si", sg="Fd-3m", ops=Fd3m, formula="Si",
      lat=(5.43094, 5.43094, 5.43094, 90, 90, 90),
      sites=[("Si", (0.125, 0.125, 0.125), "Si1")],
      conf="high",
      src="Parámetro de red del silicio, patrón NIST SRM 640",
      note="No es una fase de tu muestra: es el patrón de calibración. Su 111 a 28.44° y su 220 a 47.30° son de las posiciones mejor conocidas en cristalografía. Mide una mezcla con silicio y la desviación de esos ángulos es el cero de tu difractómetro."),
 dict(name="MoS2_2H", sg="P6_3/mmc", ops=P63mmc, formula="MoS2",
      lat=(3.1612, 3.1612, 12.2985, 90, 90, 120),
      sites=[("Mo", (1/3, 2/3, 0.25), "Mo1"), ("S", (1/3, 2/3, 0.6210), "S1")],
      conf="high",
      src="Bronsema, De Boer & Jellinek, Z. Anorg. Allg. Chem. 540 (1986) 15",
      note="Molibdenita 2H. La 002 a 14.4° es la reflexión de apilamiento y en material exfoliado se debilita hasta desaparecer: su AUSENCIA, no su posición, es lo que indica pocas capas. La 002 fuerte y estrecha significa cristal masivo mal exfoliado. Si la 002 medida es MUCHO más intensa que la calculada, eso es orientación preferente (las láminas tumbadas en el portamuestras), no más material: refina March-Dollase sobre el eje 001 antes de sacar fracciones en peso."),
 dict(name="MoSe2_2H", sg="P6_3/mmc", ops=P63mmc, formula="MoSe2",
      lat=(3.2890, 3.2890, 12.9270, 90, 90, 120),
      sites=[("Mo", (1/3, 2/3, 0.25), "Mo1"), ("Se", (1/3, 2/3, 0.6210), "Se1")],
      conf="medium",
      src="Estructura 2H del MoSe2, isoestructural con el MoS2",
      note="Como el MoS2, con la 002 a 13.7°. OJO CON LAS INTENSIDADES: en polvo verdaderamente aleatorio el cálculo da la 103 (37.9°) más intensa que la 002, porque el selenio pesa mucho más que el azufre y cambia el factor de estructura. En una muestra real prensada, las láminas se tumban y la 002 sale enormemente reforzada. Esa discrepancia NO es un error del cálculo ni de la estructura: es textura, y se corrige con el parámetro de orientación preferente (March-Dollase, eje 001). Es la causa número uno de que las intensidades no cuadren en un material laminar. La coordenada z del selenio está tomada por analogía con el sulfuro."),
 dict(name="WS2_2H", sg="P6_3/mmc", ops=P63mmc, formula="WS2",
      lat=(3.1532, 3.1532, 12.3230, 90, 90, 120),
      sites=[("W", (1/3, 2/3, 0.25), "W1"), ("S", (1/3, 2/3, 0.6220), "S1")],
      conf="medium",
      src="Estructura 2H del WS2, isoestructural con el MoS2",
      note="El tungsteno domina la dispersión (Z=74 frente a 16 del azufre), así que las intensidades son casi insensibles a la posición del azufre. Eso hace la identificación muy robusta y el refinamiento de la z, casi imposible."),
 dict(name="MoO2", sg="P2_1/c", ops=P21c, formula="MoO2",
      lat=(5.6109, 4.8562, 5.6285, 90, 120.95, 90),
      sites=[("Mo", (0.2316, 0.9917, 0.0164), "Mo1"),
             ("O", (0.1123, 0.2183, 0.2337), "O1"),
             ("O", (0.3897, 0.6934, 0.2971), "O2")],
      conf="medium",
      src="Tipo rutilo distorsionado (estructura del MoO2); Brandt & Skapski, Acta Chem. Scand. 21 (1967) 661",
      note="Óxido conductor. Su reflexión más intensa (-111) está a 26.0°, cerca de la 002 del grafito a 26.5°: en un compuesto MoO2/carbono las dos se solapan y hay que ajustar, no leer."),
]

out = pathlib.Path(__file__).resolve().parent
for ref in REFS:
    a, b, c, al, be, ga = ref["lat"]
    crystal = Crystal(
        name=ref["name"],
        lattice=Lattice(a, b, c, al, be, ga),
        sites=[Site(el, pos, label=lab) for el, pos, lab in ref["sites"]],
        operations=ref["ops"],
        space_group=ref["sg"],
        formula=ref["formula"],
        source=ref["src"],
        confidence=ref["conf"],
        notes=ref["note"],
    )
    path = write_cif(crystal, out / f"{ref['name']}.cif")
    # append the metadata CIF does not carry natively
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"\n_ramancarbon_confidence   '{ref['conf']}'\n")
        fh.write(f"_ramancarbon_source       '{ref['src']}'\n")
        fh.write(f"_ramancarbon_notes\n;\n{ref['note']}\n;\n")
    print(f"{ref['name']:20s} {crystal.order:3d} ops  {crystal.atoms_per_cell:3d} átomos  "
          f"rho={crystal.density:.3f}" if crystal.density else ref['name'])
