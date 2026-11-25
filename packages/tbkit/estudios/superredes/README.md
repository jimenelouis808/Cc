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
- **Una banda plana en E_F invita al magnetismo** (inestabilidad de Stoner).
  Se calculó (sección siguiente) y **no lo produce** a U realistas.

**Límites**: modelo π de primeros vecinos sin relajar con el modelo, sin σ y sin
interacción. Los gaps y las posiciones de banda son del modelo, no energías
ópticas. La curvatura de los nodos mezcla σ y π, y el modelo π no lo ve: la
sección de `c_bandas` lo mide.

## Comprobación s+p con `c_bandas`

`c_bandas.py` repite las bandas con `c_bandas` (s + p, ajustado a bandas GPAW PBE
de grafeno y tubos; solo electrónico), que sí ve la mezcla σ-π de la curvatura.
Reanudable por punto k (`cache_c_bandas/`, fuera de git); resultados en
`resultados_c_bandas.json`, figura `figuras/bandas_c_bandas.png` (π en gris,
s+p en color, cero en el centro del gap o en E_F).

| estructura | gap π | gap s+p | qué cambia |
|---|---|---|---|
| **Schwarz D** | 1.25 eV | **0.51 eV**, directo en Γ | sigue siendo semiconductor; la mezcla σ-π de los nodos curvos reduce el gap a menos de la mitad. Los bordes de banda siguen muy planos |
| súper-cuadrado | 0 | 0 | sigue el casi-contacto en M (~10 meV, entre bandas casi planas). Es la estructura en que más se mueven las bandas cercanas |
| súper-grafeno (cristalina) | 0 | 0.007 eV | prácticamente igual a π: cruce lineal en Γ–X (a 2/3 de Γ) y banda casi plana 0.07 eV por debajo de E_F |

- **El gap de Schwarz D se confirma, pero su valor no**: 1.25 eV en π y 0.51 eV
  en s+p. El número que valga dependerá de DFT; lo que es robusto es que D es
  semiconductor mientras P y el giroide no.
- **Los cruces de las superredes cristalinas son robustos** a la mezcla σ-π. El
  súper-grafeno casi no se mueve.
- **Validez**: `c_bandas` se ajustó con C-C de 1.37 a 1.48 Å y ±3 eV de E_F. Los
  nodos de las estructuras de codo llegan a 1.35-1.49 Å, ligeramente fuera de
  ese rango. Las geometrías no están relajadas con el modelo.

## Magnetismo de campo medio (Hubbard)

`magnetismo.py` resuelve el Hubbard de campo medio del modelo π
(`tbkit.hubbard.mean_field`, kT = 5 meV) con U = |t| = 2.7 eV, el valor por
defecto de tbkit (Yazyev 2010). Las redes no son bipartitas (heptágonos), así que
el teorema de Lieb no fija la respuesta y no existe un arranque antiferro de
subred. Cada estructura parte de tres estados (paramagnético, ferro y aleatorio)
y gana la energía más baja. Resultados en `magnetismo.json`.

| estructura | átomos | k | E (eV), los tres arranques | M, Σ\|m\| (μB) |
|---|---|---|---|---|
| grafeno (control) | 8 | 24×24 | −34.011 | 0, 0 |
| súper-cuadrado | 372 | 8×8 | −1574.185 | 0, 0 |
| súper-grafeno (cristalina) | 920 | 4×4 | −3901.478 | 0, 0 |
| Schwarz D (control con gap) | 1440 | 2×2×2 | −6089.332 | 0, < 10⁻³ |

**A U = |t| nada es magnético.** Los tres arranques convergen al mismo estado
no magnético. La banda plana es una sola entre cientos por celda: la DOS en E_F
es de ~0.05 estados/eV/átomo y U·DOS ≈ 0.1, lejos del criterio de Stoner (~1).

**Barrido de U** (`magnetismo_U.py`, `magnetismo_U.json`, figura
`figuras/magnetismo_U.png`). Arranque antiferro en el grafeno y aleatorio en las
superredes: un arranque ferro conserva la simetría uniforme y no puede llegar a
un estado escalonado.

| estructura | 1.5\|t\| | 2.0\|t\| | 2.5\|t\| |
|---|---|---|---|
| grafeno (control) | 0 | 0 | 0.21 μB/sitio |
| súper-cuadrado | 0 | 0 | 0.18 μB/sitio de media, máx. 0.26, M = 0 |
| súper-grafeno (cristalina) | 0 | 0.007 μB/sitio de media, máx. 0.036, M = 0 | 0.19 μB/sitio de media, máx. 0.24, M = 0 |

(entradas: ⟨|m|⟩ por átomo.)

- **El control pasa**: el grafeno se magnetiza entre 2.0 y 2.5 |t|. El umbral de
  campo medio del panal es 2.23 |t| (Monte Carlo cuántico: ~3.8 |t|).
- **Las superredes no adelantan el umbral.** El súper-cuadrado se ordena en el
  mismo intervalo que el grafeno, con un estado escalonado de momento total
  cero. `momentos.py` (`momentos.json`) reparte los momentos según la distancia
  en enlaces a un heptágono: 0.17 μB en los átomos de heptágono (104), 0.17 a
  1-2 enlaces (132) y 0.20 a 3-5 enlaces (136). Es la inestabilidad
  antiferro del panal, algo más fuerte lejos de los nodos, no un magnetismo de
  banda plana localizado en ellos.
- **El súper-grafeno parece adelantar el umbral, débilmente (en verificación: el grafeno con la malla 24×24 también da 0.006 μB a 2.0 |t|).** A 2.0 |t|, donde el
  grafeno aún no tiene momentos, ya tiene un estado escalonado (M = 0, máximo
  0.036 μB) localizado en los nodos: el 44 % de Σ|m| está en los 168 átomos de
  heptágono y el 39 % a 1-2 enlaces de ellos; a ≥ 6 enlaces no hay momento.
  Es lo esperable de la banda casi plana pegada al cruce, que vive en los nodos.
  Converge lento: 690 iteraciones (2.8 h), el frenado típico junto a un umbral.
  A 2.5 |t| ese estado de nodo queda absorbido por el antiferro del panal: los
  momentos se reparten (0.175 μB en heptágonos, 0.21 a ≥ 6 enlaces), como en el
  súper-cuadrado.
- **U > 2|t| no es físico para el carbono**: el U local efectivo del grafeno,
  descontando las repulsiones no locales, es ~1.6 |t| (Schüler et al., PRL 111,
  036601, 2013). El barrido mide la
  distancia a la inestabilidad; **no predice magnetismo**. Los momentos de
  campo medio son un parámetro de orden, no un estado fundamental correlacionado.
