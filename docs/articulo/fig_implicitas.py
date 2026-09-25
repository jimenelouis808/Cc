"""Figura: las familias que nacen de un campo implícito.

Ninguna se arma pegando fragmentos. Se define f(x), se malla su nivel
cero y el censo de anillos sale de la curvatura.
"""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
warnings.simplefilter("ignore")
import figuras as F
from nanocarbon_lab.builders.junction import build_junction, build_schwarzite
from nanocarbon_lab.builders.nanocoil import build_nanocoil
from nanocarbon_lab.builders.network import build_nanotube_network

paneles = []
def add(titulo, fn, **kw):
    t0 = time.time()
    a = fn()
    print(f"{titulo:32s} {F.censo(a)}  [{time.time()-t0:.0f}s]", flush=True)
    paneles.append((f"{titulo}\n{F.censo(a)}", a, kw))

add("Unión Y (3 brazos, 120°)",
    lambda: build_junction(kind="Y", tube_radius=6.0, arm_length=22.0, blend=4.0),
    caras=True, grosor=0.28, zoom=1.15)
add("Unión X (4 brazos)",
    lambda: build_junction(kind="X", tube_radius=6.0, arm_length=20.0, blend=4.0),
    caras=True, grosor=0.28, zoom=1.15)
add("Unión 3D (6 brazos)",
    lambda: build_junction(kind="cross3d", tube_radius=5.0, arm_length=18.0, blend=3.5),
    caras=True, grosor=0.28, zoom=1.15)
add("Schwarzita P (Schwarz primitiva)",
    lambda: build_schwarzite(kind="primitive", cell=32.0),
    caras=True, corte_anillo=7.0, grosor=0.25, zoom=1.1)
add("Schwarzita giroide",
    lambda: build_schwarzite(kind="gyroid", cell=40.0),
    caras=True, corte_anillo=7.0, grosor=0.25, zoom=1.1)
add("Red cúbica de nanotubos",
    lambda: build_nanotube_network(kind="cubic", cell=40.0, tube_radius=6.0, blend=5.0),
    caras=True, corte_anillo=7.0, grosor=0.25, zoom=1.1)

d = F.rejilla("fig5_implicitas", paneles, ncols=3,
              pie="Superficies implícitas: campo → marching cubes → remallado → dual. "
                  "Naranja: pentágonos.  Azul: heptágonos.  Amarillo: octágonos.")
print("escrito", d, flush=True)
