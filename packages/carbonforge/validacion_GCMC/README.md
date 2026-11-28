# Validación del GCMC contra Levesque 2002

Reproducción de las isotermas publicadas por D. Levesque, A. Gicquel,
F. Lamari Darkrim y S. Beyaz Kayiran, *J. Phys.: Condens. Matter*
**14** (2002) 9285, tablas 2 y 3, sobre su propio adsorbente:
un haz hexagonal de nanotubos (17,0) de D = 13.3 Å separados
d = 6 Å, abiertos, 8704 carbonos en 175 900 Å³.

Nada de esto está ajustado. La geometría del haz, el reservorio y
el muestreo salen de parámetros publicados, y lo que se compara es
la columna `rho_o`: cinco decimales y ninguna convención. La
columna `wt%` del artículo **no** sirve de objetivo, porque no
concuerda con su propia `rho_o`: convertida en base de masa total, que
es la definición que escribe Kowalczyk (su ec. 5) citando a Levesque,
discrepa de la columna impresa un 3 % a 77 K — ver `bundle.py`.

## Resultado

### 77 K

| P (MPa) | ρ_o artículo | ρ_o aquí | dif | N |
|---|---|---|---|---|
| 0.1 | 0.03169 | 0.03041 | -4.0 % | 1598 ± 5 |
| 0.5 | 0.04029 | 0.03924 | -2.6 % | 2062 ± 3 |
| 1 | 0.04336 | 0.04244 | -2.1 % | 2230 ± 4 |
| 3 | 0.04694 | 0.04580 | -2.4 % | 2406 ± 4 |
| 5 | 0.04818 | 0.04740 | -1.6 % | 2491 ± 6 |
| 10 | 0.04996 | 0.04853 | -2.9 % | 2550 ± 4 |
| 15 | 0.05120 | 0.04891 | -4.5 % | 2570 ± 3 |
| 20 | 0.05128 | 0.04949 | -3.5 % | 2600 ± 7 |

8 puntos, diferencia absoluta media **3.0 %**, peor **4.5 %**.

### 293 K

| P (MPa) | ρ_o artículo | ρ_o aquí | dif | N |
|---|---|---|---|---|
| 0.1 | 0.00014 | 0.00015 | +3.7 % | 8 ± 0 |
| 0.5 | 0.00073 | 0.00072 | -1.7 % | 38 ± 0 |
| 1 | 0.00145 | 0.00140 | -3.5 % | 74 ± 0 |
| 3 | 0.00399 | 0.00394 | -1.3 % | 207 ± 2 |
| 5 | 0.00614 | 0.00612 | -0.3 % | 322 ± 2 |
| 10 | 0.01040 | 0.01067 | +2.6 % | 561 ± 2 |
| 15 | 0.01380 | 0.01446 | +4.8 % | 760 ± 4 |
| 20 | 0.01647 | 0.01742 | +5.8 % | 916 ± 3 |

8 puntos, diferencia absoluta media **3.0 %**, peor **5.8 %**.

## Qué significa ese ±6 %, y qué no

Es la precisión a la que este prototipo tiene derecho, y las tres
fuentes están medidas, no estimadas:

- El reservorio es un virial de dos términos ajustado a la tabla
  redondeada del propio artículo, y reproduce esa tabla al 1.2 %.
- La molécula es un solo sitio Lennard-Jones sin cuadrupolo.
- Las densidades publicadas están redondeadas a cinco decimales,
  lo que a 0.1 MPa y 293 K es **una cifra significativa**.

A 77 K las diferencias son todas del mismo signo, y el aviso de
deriva dice por qué: a 20 MPa la carga seguía subiendo +61
moléculas entre el primer bloque de producción y el último. Correr
más largo acerca, no aleja.

**Lo que esto valida es el muestreo, no el modelo.** Reproducir a
Levesque significa que el potencial, la corrección de Feynman-Hibbs,
la actividad del reservorio y el balance de las jugadas están bien
puestos. No dice que el modelo de un sitio sin cuadrupolo sea cierto
para el hidrógeno: dice que está implementado como él lo implementó.

## La misma validación con LAMMPS

Los dieciséis puntos otra vez, con `fix gcmc` de LAMMPS (22 jul 2025)
en lugar del Monte Carlo propio: el mismo haz, los mismos potenciales
de Feynman-Hibbs (LAMMPS los lee en tabla, generados por
`potentials.feynman_hibbs`), la misma fugacidad del virial, el mismo
corte truncado a 12 Å y el mismo número de jugadas por punto
(`carbonforge/adsorption/lammps.py`, `lammps_levesque.py`,
`datos_lammps.json`).

Antes de la isoterma, dos comprobaciones de que los dos motores
calculan lo mismo, no sólo números parecidos:

- **Caja vacía** (un carbono en 40 Å): 79.02 ± 0.69 moléculas en
  LAMMPS contra 79.08 ± 0.88 en el propio a 293 K y 5 MPa; 57.06 ±
  0.70 contra 57.05 ± 0.56 a 77 K y 1 MPa. El gas, la fugacidad y el
  potencial H₂–H₂ son el mismo.
- **Un H₂ dentro del haz**: 105 994.08 K en LAMMPS, 105 994.03 K en el
  propio. El potencial H₂–C es el mismo.

### 77 K

| P (MPa) | ρ_o artículo | ρ_o propio | ρ_o LAMMPS | LAMMPS − artículo | LAMMPS − propio | N (LAMMPS) |
|---|---|---|---|---|---|---|
| 0.1 | 0.03169 | 0.03041 | 0.03123 | -1.5 % | +2.7 % | 1655 ± 4 |
| 0.5 | 0.04029 | 0.03924 | 0.03953 | -1.9 % | +0.7 % | 2096 ± 5 |
| 1 | 0.04336 | 0.04244 | 0.04269 | -1.6 % | +0.6 % | 2263 ± 3 |
| 3 | 0.04694 | 0.04580 | 0.04609 | -1.8 % | +0.6 % | 2444 ± 4 |
| 5 | 0.04818 | 0.04740 | 0.04743 | -1.6 % | +0.1 % | 2514 ± 2 |
| 10 | 0.04996 | 0.04853 | 0.04861 | -2.7 % | +0.2 % | 2577 ± 1 |
| 15 | 0.05120 | 0.04891 | 0.04942 | -3.5 % | +1.0 % | 2620 ± 7 |
| 20 | 0.05128 | 0.04949 | 0.05004 | -2.4 % | +1.1 % | 2652 ± 7 |

LAMMPS contra el artículo: diferencia absoluta media **2.1 %**, peor 3.5 %. Contra el GCMC propio: media +0.9 %, de +0.1 a +2.7 %.

### 293 K

| P (MPa) | ρ_o artículo | ρ_o propio | ρ_o LAMMPS | LAMMPS − artículo | LAMMPS − propio | N (LAMMPS) |
|---|---|---|---|---|---|---|
| 0.1 | 0.00014 | 0.00015 | 0.00015 | +10.4 % | +3.0 % | 8 ± 0 |
| 0.5 | 0.00073 | 0.00072 | 0.00072 | -1.5 % | -0.1 % | 38 ± 0 |
| 1 | 0.00145 | 0.00140 | 0.00145 | -0.3 % | +3.3 % | 77 ± 1 |
| 3 | 0.00399 | 0.00394 | 0.00400 | +0.2 % | +1.5 % | 212 ± 2 |
| 5 | 0.00614 | 0.00612 | 0.00631 | +2.8 % | +3.2 % | 335 ± 3 |
| 10 | 0.01040 | 0.01067 | 0.01095 | +5.3 % | +2.6 % | 580 ± 4 |
| 15 | 0.01380 | 0.01446 | 0.01464 | +6.1 % | +1.3 % | 776 ± 4 |
| 20 | 0.01647 | 0.01742 | 0.01751 | +6.3 % | +0.5 % | 928 ± 6 |

LAMMPS contra el artículo: diferencia absoluta media **4.1 %**, peor 10.4 %. Contra el GCMC propio: media +1.9 %, de -0.1 a +3.3 %.

**Lo que dice.** Los dos motores coinciden dentro del 3.3 % en los
dieciséis puntos, y a 77 K, donde la corrección cuántica pesa un 20 %,
LAMMPS cae **más cerca del artículo** (2.1 % de media contra 3.0 %): el
propio quedaba bajo y su aviso de deriva decía que faltaba muestreo,
y LAMMPS, con las mismas jugadas, llega más lejos. A 293 K y alta
presión los dos se pasan del artículo lo mismo (+5 a +6 %), y que dos
motores independientes se pasen igual apunta al modelo (sin
cuadrupolo, reservorio virial) y no al muestreo. El +10 % a 0.1 MPa
son 8 moléculas contra un valor publicado con una cifra significativa.

LAMMPS es además rápido: con 300 intentos por paso, 90 000 jugadas en
43 s sobre las 8704 átomos del haz. Con 3 por paso tardaba media hora:
LAMMPS reconstruye la lista de vecinos de todo el armazón tras cada
paso que cambia el número de átomos, y eso era la corrida entera.

## Reproducirlo

```bash
carbonforge adsorcion isoterma haz.xyz salida/ \
    --temperatura 77 --presiones 0.1,0.5,1,3,5,10,15,20
python -m carbonforge.adsorption.run salida/
```

Son del orden de 6 horas por isoterma a 77 K. La carpeta reanuda:
los puntos que ya estén no se recalculan.

`datos.json` lleva los números de esta figura.
