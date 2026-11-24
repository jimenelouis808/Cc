# Estructura electrónica π de las superestructuras de carbono

`electronica.py` (modelo π Hückel de tbkit, t = −2.7 eV, primeros vecinos)
sobre las superestructuras de `nanocarbon_lab` usadas en el estudio de
adsorción. Las estructuras llegan como extxyz copiados en `estructuras/`; no se
importa ningún otro paquete. `figuras.py` dibuja `figuras/bandas.png` y
`figuras/dos.png`; los números están en `resultados.json`.

**Control**: el grafeno da el cono de Dirac exactamente en K.

| estructura | átomos | carácter | detalle |
|---|---|---|---|
| súper-grafeno (cristalina) | 920 | semimetal | cruce lineal en E_F sobre Γ–X (~0.6 de Γ a X) más una banda casi plana 0.05 eV por debajo |
| súper-cuadrado (cristalina) | 372 | semimetal | cono en M justo en E_F y una banda casi plana en E_F |
| Schwarz P | 968 | metal/semimetal | muchos cruces cerca de E_F |
| **Schwarz D** | 1440 | **semiconductor, gap 1.25 eV** | bordes de banda muy planos, a ±0.63 eV del centro del gap |
| giroide | 1744 | metal | DOS(E_F) 0.041 /eV/átomo |
| súper-grafeno (mallada) | 1180 | bandas planas | sin cono: los pares 5-7 dispersos localizan los estados |
| super-cube (jaula) | 1748 | molécula | HOMO-LUMO 0.043 eV |
| super-hypercube (jaula) | 6494 | molécula | HOMO-LUMO 0.007 eV |

- **Las superredes cristalinas reproducen la física de una red a escala de
  superred.** El súper-cuadrado tiene los nodos en los vértices de una red
  cuadrada y los tubos sobre sus aristas, que es la topología de la red de Lieb.
  Lieb da exactamente una banda plana y un cono de Dirac en M, y eso es lo que
  sale. El súper-grafeno es un panal de nodos con sitios en las aristas, y da un
  cono de Dirac con una banda casi plana pegada. Es una interpretación por
  topología, coherente con las bandas, no una demostración.
- **El desorden de la ruta mallada lo borra.** El mismo súper-grafeno construido
  por malla, con pares 5-7 dispersos, solo tiene bandas planas, sin cono. Para
  física de bandas la ruta cristalina es necesaria, no estética.
- **Schwarz D es la única con gap**: 1.25 eV en el modelo π. P y el giroide, de
  la misma familia de superficies mínimas, son metálicos.
- **Una banda plana en E_F invita al magnetismo** (inestabilidad de Stoner;
  teorema de Lieb para la red de Lieb). No se calculó aquí: el siguiente paso es
  `tbkit.hubbard` de campo medio sobre el súper-cuadrado y el súper-grafeno.

**Límites**: modelo π de primeros vecinos sin relajar con el modelo, sin σ y sin
interacción. Los gaps y las posiciones de banda son del modelo, no energías
ópticas. La curvatura de los nodos mezcla σ y π, y el modelo π no lo ve. Para
confirmar el gap de Schwarz D y los conos haría falta `c_bandas` o DFT.
