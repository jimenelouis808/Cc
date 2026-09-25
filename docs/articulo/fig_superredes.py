"""Figura: la jerarquía de superredes, cada arista un nanotubo."""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
warnings.simplefilter("ignore")
import figuras as F
from nanocarbon_lab.builders.supernetwork import build_supernetwork

CASOS = [
    ("super-graphene",     dict(scale=34.0, tube_radius=5.0, blend=4.0), "Super-grafeno (3 tubos, 120°)"),
    ("super-cubic",        dict(scale=34.0, tube_radius=5.0, blend=4.0), "Super-cúbica (6 tubos)"),
    ("super-fcc",          dict(scale=40.0, tube_radius=3.0, blend=2.0), "Super-fcc (12 tubos)"),
    ("super-icosahedron",  dict(scale=24.0, tube_radius=4.0, blend=3.0), "Jaula icosaédrica"),
    ("super-hypercube",    dict(scale=20.0, tube_radius=3.5, blend=2.5), "Hipercubo (4-cubo)"),
    ("superfullerene-C60", dict(scale=14.2, tube_radius=3.0, blend=2.0), "Superfulereno C$_{60}$"),
]
paneles = []
for nombre, kw, titulo in CASOS:
    t0 = time.time()
    a = build_supernetwork(graph=nombre, seed=0, **kw)
    print(f"{nombre:20s} {F.censo(a)}  [{time.time()-t0:.0f}s]", flush=True)
    paneles.append((f"{titulo}\n{len(a)} átomos", a,
                    dict(caras=True, corte_anillo=7.0, grosor=0.3, zoom=1.15)))
d = F.rejilla("fig4_superredes", paneles, ncols=3,
              pie="Naranja: pentágonos.  Azul: heptágonos.  Amarillo: octágonos.  "
                  "Nadie los coloca: salen de la curvatura de cada nodo.")
print("escrito", d, flush=True)
