"""Figura: la ruta de rodillas, donde el censo es el que la ley pide.

El contraste con `fig5_implicitas` es el punto de la lámina. La ruta
implícita deriva sus anillos del remallado y devuelve una pared sana y
amorfa -- 11-21% de anillos no hexagonales, las disclinaciones en el
lado correcto y muchísimas más de las necesarias. Aquí el nodo se arma
de brazos rectos y paga su curvatura en anillos COLOCADOS POR LA
CONSTRUCCIÓN, así que sale exactamente el presupuesto y ni uno más.
"""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
warnings.simplefilter("ignore")
import figuras as F
from nanocarbon_lab.jobs import Job, build

# (clave de caché, modo, parámetros, título, presupuesto, réplicas)
CASOS = [
    ("rod-toroide", "toroid (knees)", {},
     "Toroide, 6 rodillas", 0, None),
    ("rod-bobina", "coil (knees, periodic)", {},
     "Bobina, celda periódica", 0, (1, 1, 2)),
    ("rod-union-y", "junction (knees)", dict(kind="y"),
     "Unión Y (3 brazos, 120°)", -6, None),
    ("rod-union-x", "junction (knees)", dict(kind="x"),
     "Unión X plana (4 brazos)", -12, None),
    ("rod-union-td", "junction (knees)", dict(kind="tetrahedral"),
     "Nodo tetraédrico (109.47°)", -12, None),
    ("rod-lamina", "supernetwork (knees)", dict(net="super-graphene"),
     "Lámina: tubos en panal", -24, (2, 2, 1)),
    ("rod-schwarz-p", "schwarzite (knees)", dict(kind="primitive"),
     "Esquarcita Schwarz P", -24, (2, 2, 2)),
    ("rod-schwarz-d", "schwarzite (knees)", dict(kind="diamond"),
     "Esquarcita Schwarz D", -96, (2, 2, 2)),
    ("rod-giroide", "schwarzite (knees)", dict(kind="gyroid"),
     "Giroide (red srs)", -48, (2, 2, 2)),
]

paneles = []
for clave, modo, kw, titulo, presupuesto, rep in CASOS:
    t0 = time.time()
    a = F.construir(clave, lambda m=modo, k=kw: build(Job(m, dict(k))))
    medido = sum((6 - s) * c for s, c in a.info["ring_counts"].items())
    ley = "=" if medido == presupuesto else "≠"
    extra = "" if rep is None else f"  ({rep[0]}×{rep[1]}×{rep[2]} celdas)"
    print(f"{clave:16s} {F.censo(a)}  ley {medido:+d} {ley} {presupuesto:+d}"
          f"  [{time.time()-t0:.0f}s]", flush=True)
    paneles.append((f"{titulo}{extra}\n{len(a)} átomos · "
                    f"Σ(6−n) = {medido:+d}", a,
                    dict(caras=True, corte_anillo=7.0, grosor=0.28, zoom=1.1,
                         repetir=rep)))

d = F.rejilla("fig6_rodillas", paneles, ncols=3,
              pie="Naranja: pentágonos.  Azul: heptágonos.  Nadie los coloca "
                  "tampoco aquí, pero a diferencia de la ruta implícita salen "
                  "en el número exacto que el esqueleto fija antes de mallar: "
                  "Σ(6−n) = 6χ para un nodo, 12(V−E) para una red, 0 para un "
                  "toro.")
print("escrito", d, flush=True)
