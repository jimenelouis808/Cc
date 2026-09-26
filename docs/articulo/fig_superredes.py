"""Figura: la jerarquía de superredes, cada arista un nanotubo."""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
warnings.simplefilter("ignore")
import figuras as F
from nanocarbon_lab.builders.supernetwork import build_supernetwork

# (nombre, parámetros, título, réplicas). Las celdas periódicas se
# repiten: dibujada sola, una celda se lee como un fragmento roto.
CASOS = [
    ("super-graphene",     dict(scale=34.0, tube_radius=5.0, blend=4.0), "Super-grafeno (3 tubos, 120°)", (2, 2, 1)),
    ("super-cubic",        dict(scale=34.0, tube_radius=5.0, blend=4.0), "Super-cúbica (6 tubos)", (2, 2, 2)),
    ("super-fcc",          dict(scale=40.0, tube_radius=3.0, blend=2.0), "Super-fcc (12 tubos)", (2, 2, 1)),
    ("super-icosahedron",  dict(scale=24.0, tube_radius=4.0, blend=3.0), "Jaula icosaédrica", None),
    ("super-hypercube",    dict(scale=20.0, tube_radius=3.5, blend=2.5), "Hipercubo (4-cubo)", None),
    ("superfullerene-C60", dict(scale=14.2, tube_radius=3.0, blend=2.0), "Superfulereno C$_{60}$", None),
]
paneles = []
for nombre, kw, titulo, rep in CASOS:
    t0 = time.time()
    a = F.construir(nombre, lambda n=nombre, k=kw: build_supernetwork(graph=n, seed=0, **k))
    extra = "" if rep is None else f"  ({rep[0]}×{rep[1]}×{rep[2]} celdas)"
    print(f"{nombre:20s} {F.censo(a)}  [{time.time()-t0:.0f}s]", flush=True)
    paneles.append((f"{titulo}{extra}\n{len(a)} átomos por celda", a,
                    dict(caras=True, corte_anillo=7.0, grosor=0.28, zoom=1.1,
                         repetir=rep)))
d = F.rejilla("fig4_superredes", paneles, ncols=3,
              pie="Naranja: pentágonos.  Azul: heptágonos.  Amarillo: octágonos.  "
                  "Nadie los coloca: salen de la curvatura de cada nodo.")
print("escrito", d, flush=True)
