"""Figura: las familias que se construyen por red directa, no por campo."""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
warnings.simplefilter("ignore")
import figuras as F
from nanocarbon_lab.builders.cnt import build_cnt
from nanocarbon_lab.builders.fullerene import build_fullerene, build_nano_onion
from nanocarbon_lab.builders.nanocone import build_nanocone
from nanocarbon_lab.builders.haeckelite import build_haeckelite
from nanocarbon_lab.builders.toroid import build_polyhex_toroid
from nanocarbon_lab.builders.graphene import build_graphene_supercell

paneles = []
def add(titulo, atoms, **kw):
    print(f"{titulo:34s} {F.censo(atoms)}", flush=True)
    paneles.append((f"{titulo}\n{F.censo(atoms)}", atoms, kw))

add("Nanotubo (6,6)", build_cnt(6, 6, length=22.0),
    elev=14, azim=-68, zoom=1.0)
add("Fulereno C$_{60}$", build_fullerene(freq=1, family="C60"),
    caras=True, zoom=1.2)
add("Cebolla de carbono (3 capas)", build_nano_onion(n_shells=3, inner_freq=1),
    grosor=0.3, zoom=1.15)
add("Nanocono (1 pentágono, 112.9°)", build_nanocone(n_pentagons=1, radius=20.0),
    caras=True, elev=18, azim=-60, zoom=1.1)
add("Haeckelita R5,7 (plana)", build_haeckelite(nx=4, ny=4, pattern="r57"),
    caras=True, elev=88, azim=-90, zoom=1.25)
add("Toroide polihex (5,5)", build_polyhex_toroid(n=5, m=5, periods=60),
    elev=58, azim=-60, grosor=0.3, zoom=1.15)

d = F.rejilla("fig2_basicas", paneles, ncols=3,
              pie="Familias construidas tejiendo la red directamente. "
                  "Naranja: pentágonos.  Azul: heptágonos.")
print("escrito", d, flush=True)
