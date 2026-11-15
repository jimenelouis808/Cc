# La rendija de grafeno: el ancho de poro, con todo lo demás quieto

Diez rendijas de grafeno, de 6 a 20 Å entre paredes, a 77 K y 1, 3.8 y
10 MPa. Treinta puntos de Monte Carlo gran canónico.

**Esta serie existe por una razón metodológica, no por la rendija.** Las
seis estructuras medidas antes —tres schwarzitas en
`validacion_schwarzitas/` y tres superestructuras en
`validacion_superestructuras/`— varían el área, el ancho de poro y la
curvatura **a la vez**, así que ninguna diferencia entre ellas se podía
atribuir a ninguno de los tres. Una rendija de grafeno fija la química y
fija la curvatura en cero, y su área accesible resulta ser **constante en
toda la serie**. Queda una sola variable.

![exceso contra ancho de poro](ancho_vs_exceso.png)

## El área constante es el control, y además calibra la medida

| H (centro a centro) | canal libre | área (m²/g) | vacío He |
|---|---|---|---|
| 6 Å | 2.97 Å | **1973** | 0.674 |
| 7 Å | 3.95 Å | 2670 | 0.809 |
| 8 Å | 4.93 Å | 2670 | 0.822 |
| 9 Å | 5.92 Å | 2670 | 0.834 |
| 10 Å | 6.91 Å | 2670 | 0.847 |
| 11 Å | 7.90 Å | 2670 | 0.860 |
| 12 Å | 8.89 Å | 2670 | 0.866 |
| 14 Å | 10.88 Å | 2670 | 0.885 |
| 16 Å | 12.88 Å | 2670 | 0.897 |
| 20 Å | 16.86 Å | 2670 | 0.919 |

De 7 Å en adelante el área no se mueve **ni un dígito**, que es lo que
tiene que pasar: el área de una lámina de grafeno con las dos caras
accesibles no depende de lo separadas que estén las láminas. A 6 Å cae a
1973 porque el canal (2.97 Å) deja de admitir la sonda con holgura y hay
superficie a la que no llega.

Y de paso es la **primera comprobación externa** que tiene esta rutina de
área: el valor teórico del grafeno con las dos caras es 2630 m²/g, y mide
2670 — un 1.5 % de diferencia. Hasta ahora sólo se había validado contra
sí misma (estable en el número de puntos, independiente de la semilla,
invariante al replicar la celda). Las tres pruebas siguen siendo
necesarias; ésta es la que dice que el número además es correcto.

El canal libre sale `H − 3.04 Å` en las diez, que es el diámetro de van
der Waals del carbono restado del centro a centro.

## Lo que mide: oscilaciones de capas, no una campana

Exceso adsorbido, en wt% de la masa total:

| H | canal | 1 MPa | 3.8 MPa | 10 MPa |
|---|---|---|---|---|
| 6 Å | 2.97 Å | 3.07 | 2.79 | 1.74 |
| 7 Å | 3.95 Å | 3.46 | 2.91 | 1.37 |
| 8 Å | 4.93 Å | 3.35 | 2.84 | 1.13 |
| **9 Å** | **5.92 Å** | 4.66 | 4.50 | 2.80 |
| 10 Å | 6.91 Å | **5.10** | 4.84 | 2.81 |
| 11 Å | 7.90 Å | 4.87 | 4.73 | 2.50 |
| **12 Å** | **8.89 Å** | 4.84 | 5.07 | **2.97** |
| 14 Å | 10.88 Å | 4.67 | **5.22** | 2.84 |
| 16 Å | 12.88 Å | 4.49 | 5.12 | 2.64 |
| 20 Å | 16.86 Å | 4.38 | 4.85 | 1.83 |

La curva no es una campana con un óptimo. **Los saltos caen en múltiplos
enteros del diámetro del hidrógeno**, σ = 2.958 Å:

- **1σ = 2.96 Å**, y el canal de H=6 mide **2.97**: una capa exacta.
- **2σ = 5.92 Å**, y el canal de H=9 mide **5.92**. Ahí está el salto
  grande, de 3.35 a 4.66 wt% a 1 MPa. El H=8, con 4.93 Å, se queda justo
  por debajo de poder meter la segunda capa: es el fondo del valle.
- **3σ = 8.87 Å**, y el canal de H=12 mide **8.89**. Ahí se recupera la
  curva tras el bache de H=11 (7.90 Å, justo por debajo), que sale
  **mínimo local en las tres presiones**: 4.87 / 4.73 / 2.50 contra sus
  dos vecinos a cada lado.

Es decir: cada vez que el canal cruza un múltiplo de σ entra una capa
nueva y la captación salta; justo antes de cruzarlo hay un hueco que no
da para la capa siguiente, y ese hueco es desperdicio. La rejilla
aterrizó en 1σ, 2σ y 3σ por casualidad — 6, 9 y 12 se eligieron para
acotar picos, no por esto — y es lo que hace la lectura inequívoca.

**El máximo se corre con la presión**, porque cada presión prefiere un
número de capas distinto: **H = 10 Å a 1 MPa**, **14 Å a 3.8 MPa**,
**12 Å a 10 MPa**. Comparar materiales a una sola presión, como se hizo
al principio de este estudio, puede invertir el orden.

**Y el valle se agrava con la presión.** H=8 pasa de 3.35 a 1.13 wt%
entre 1 y 10 MPa, mientras H=6 sólo baja de 3.07 a 1.74: cuanto más gas
hay disponible, más se nota el hueco que no se puede llenar.

## Lo que esto dice de las estructuras exóticas

A cada presión, el mejor de las diez rendijas contra el mejor de las seis
estructuras medidas antes:

| | mejor rendija | mejor estructura exótica |
|---|---|---|
| 1 MPa | **5.10** (H=10) | 4.32 (giroide) |
| 3.8 MPa | **5.22** (H=14) | 4.57 (giroide) |
| 10 MPa | **2.97** (H=12) | 2.38 (giroide) |

**A las tres presiones gana la rendija.** Ni los 48 heptágonos por celda
del giroide, ni los 360 pentágonos del superfulereno, ni la curvatura
negativa de una superficie mínima compran nada frente a dos láminas
planas a la distancia correcta. Y en la figura se ve lo mismo punto a
punto: a igualdad de ancho de canal, las estructuras exóticas caen **sobre
o por debajo** de la curva de la rendija.

Eso **no** dice que la topología no sirva para nada. Dice que, para
fisisorción de H₂ a 77 K con este modelo, la variable que manda es la
geometría del poro, y que una superestructura sólo puede ganar si llega a
anchos que una rendija no alcanza — o aportando algo que este modelo no
tiene: química distinta en la pared, que es exactamente el siguiente
paso (dopantes O, N, B, P, Se), y que necesita σ/ε por elemento.

## Se intentó llevar una superestructura a la banda buena, y no basta

La pregunta obvia tras lo anterior: si lo que importa es el ancho, ¿y si
se encoge una superestructura hasta ese ancho? Dos cosas lo impiden en
parte y una tercera lo impide del todo.

**La escala sola no es un parámetro libre.** A escalas 20, 24 y 28 con el
tubo de 5 Å de la súper-grafeno original, el constructor **rechaza** la
red: cada nodo se come unos `radio + blend` = 9 Å de *cada* extremo de un
strut, y por debajo de escala 38 no queda tubo entre nodos. El hueco y el
grosor de la pared están acoplados — `escala ≥ 2√3·(radio + blend)` — así
que estrechar la celda obliga a adelgazar el tubo a la vez. Eso sí
construye:

| | átomos | canal libre | área (m²/g) | vacío He | veredicto |
|---|---|---|---|---|---|
| escala 22, tubo 4, blend 2 | 2192 | **9.15 Å** | 1963 | 0.732 | TENSO |
| escala 26, tubo 5, blend 2 | 1588 | **10.57 Å** | 1995 | 0.744 | TENSO |
| escala 34, tubo 5, blend 4 | 2360 | 18.46 Å | 2250 | 0.808 | TENSO |

(La de escala 18 y tubo 3 también construye y sale ROTA: enlace de
1.585 Å. Las tres cumplen el presupuesto `sum(6-n) = −24` exacto y no
tienen un solo contacto por debajo de 2 Å.)

Los canales de 9.15 y 10.57 Å caen justo donde la rendija es mejor. Pero
**el área baja** — 2250 → 1995 → 1963 — que es el precio de estrechar una
red de tubos: más fracción de masa en la pared.

**La predicción se registró antes de correr**, para que pudiera fallar:
si el ancho es lo único que manda, la superestructura debería dar lo de
la rendija al mismo ancho multiplicado por el cociente de áreas. Falló.

| | canal | P (MPa) | wt% exc | rendija igual ancho | predicho × área | medido/predicho |
|---|---|---|---|---|---|---|
| escala 22 | 9.15 Å | 1.0 | 3.11 | 4.82 | 3.54 | **0.88** |
| | | 3.8 | 3.06 | 5.09 | 3.74 | **0.82** |
| | | 10.0 | 1.69 | 2.96 | 2.17 | **0.78** |
| escala 26 | 10.57 Å | 1.0 | 3.18 | 4.69 | 3.51 | **0.91** |
| | | 3.8 | 3.05 | 5.20 | 3.88 | **0.79** |
| | | 10.0 | 1.68 | 2.86 | 2.14 | **0.78** |

**Entre un 9 y un 22 % por debajo de lo que explica el área, y el déficit
crece con la presión.** Encoger ayuda algo —la eficiencia por área sube
de 0.75 en la escala 34 a 0.79 en la 22— pero no alcanza a la rendija.

### Y la explicación que se dio primero era falsa

Al medir esto se propuso que **igualar el diámetro de esfera libre fuese
una forma ingenua de igualar poros**: en una rendija de 9 Å todo punto
está a menos de 4.5 Å de una pared, mientras que un canal hexagonal con
la misma esfera inscrita tendría esquinas y un centro lejos de toda
pared, que se llenaría de gas casi libre. Encajaba con que el déficit
creciera con la presión.

**Se midió, y es falsa.** `analyse_porosity` reporta ahora
`mean_wall_distance`, la distancia media de un punto del volumen
accesible a la superficie del átomo más cercano. Para una rendija vale
`probe_radius/2 + canal/4` en forma cerrada, y las diez rendijas caen
sobre su valor analítico a ±0.04 Å o menos. Las superestructuras leen
**más bajo**, no más alto:

| | canal | pared media | rendija del mismo canal | exceso |
|---|---|---|---|---|
| súper-grafeno e22 | 9.15 Å | 2.77 Å | 3.01 Å | **−0.24** |
| súper-grafeno e26 | 10.57 Å | 2.87 Å | 3.37 Å | **−0.49** |

Su volumen accesible está **más cerca** de las paredes que el de una
rendija equivalente, no más lejos. La causa es que los tubos son huecos:
un tubo de radio 4 Å tiene un interior con radio libre de 2.4 Å, que el
H₂ alcanza, y ese volumen pegado a la pared arrastra la media.

Mirando la cola en vez de la media, que es lo que la bimodalidad pide,
la conclusión no cambia. Fracción del volumen accesible a más de una
distancia dada de la pared, contra la rendija del mismo canal:

| | > 2 Å | > 3 Å | > 4 Å | > 5 Å |
|---|---|---|---|---|
| súper-grafeno e22 | 0.711 | 0.336 | 0.146 | 0.044 |
| rendija igual | 0.823 | 0.503 | 0.184 | 0.000 |
| súper-grafeno e26 | 0.686 | 0.362 | 0.193 | 0.069 |
| rendija igual | 0.855 | 0.595 | 0.335 | 0.074 |

Hay algo de cola más allá de lo que una rendija puede tener —un 4.4 %
del volumen de la e22 está más lejos de una pared que **cualquier** punto
de su rendija equivalente, que es el centro de los hexágonos— pero un
4 % no explica un déficit del 9 al 22 %, y en todo lo demás la
superestructura tiene **menos** volumen lejos de la pared, no más.

Así que **la geometría del poro no explica el déficit**, y las dos
explicaciones que se propusieron están descartadas por medición. El
candidato que quedaba era la **curvatura de la pared**: la superficie
de la superestructura es sobre todo exterior de tubo, que es convexa, y
una pared convexa tiene menos carbonos al alcance de un punto dado que
una lámina plana, así que su pozo de potencial es menos profundo. Eso
haría que cada m² valiese menos, independientemente del ancho de poro —
lo que encaja con que el déficit aparezca en la eficiencia **por área**.

## Y la curvatura sí lo explica: 24 % del pozo

Ya se puede medir: `analyse_porosity` informa ahora la distribución de
energía potencial del armazón sobre el volumen accesible, y
`curvatura.py` aísla la variable. Una pared cada vez, misma densidad de
carbono, misma sonda, misma temperatura, y la sonda **se barre** hacia
afuera para leer cada pared en SU propio mínimo y no en el del plano:

| (n,n) | R (Å) | pozo (K) | distancia (Å) | % del plano |
|---|---|---|---|---|
| (5,5) | 3.39 | −333.5 | 3.22 | 73.9 % |
| (6,6) | 4.07 | −344.5 | 3.22 | **76.4 %** |
| (8,8) | 5.42 | −361.2 | 3.22 | 80.1 % |
| (10,10) | 6.78 | −373.2 | 3.22 | 82.7 % |
| (15,15) | 10.17 | −392.2 | 3.22 | 87.0 % |
| (20,20) | 13.56 | −403.7 | 3.22 | 89.5 % |
| (30,30) | 20.34 | −416.7 | 3.22 | 92.4 % |
| plana | ∞ | −451.0 | 3.22 | 100.0 % |

El exterior de un (6,6) —que es el puntal de estas superestructuras—
vale **el 76 % del pozo de una lámina plana**. El déficit crece al
cerrarse el radio y **se va a cero cuando el radio crece**, que es el
límite que convierte esto en una medida y no en un relato: una pared de
radio 20 Å ya está al 92 % del plano.

Y la distancia del mínimo **no se mueve**: 3.22 Å en las ocho filas. O
sea que no es que la molécula se siente en otro sitio, es que desde el
mismo sitio ve menos carbono. Que es exactamente el mecanismo
propuesto, y no otro compatible con el mismo número.

### Lo que NO dice

Un haz de tubos medido entero sale **más** hondo que una rendija
(−493 K de media contra −347 K sobre el volumen accesible, a 77 K), y
eso no contradice lo anterior: en un haz, un sitio del surco ve
carbono de **tres** tubos a la vez, y la superposición recupera con
creces lo que la convexidad cuesta por pared. El déficit de las
superestructuras es que tienen superficie convexa **abierta** —poros
grandes, sin surcos— así que pagan el 24 % sin cobrar la superposición.
Esa es la consecuencia de diseño: lo que hay que buscar no es más área,
son surcos.

## Límites

Los mismos que `validacion_superestructuras/`, y uno propio:

- Marco **rígido**, H₂ **esférico** (sin cuadrupolo ni rotaciones), **sin
  cargas**. Áreas geométricas, no BET.
- **Paredes de una sola capa**, elegido así a propósito: las paredes de
  las superestructuras son de una capa, así que ésta es su control y no
  otro material. Una pared de grafito de tres capas tiene un pozo de
  potencial algo más profundo y daría captaciones algo mayores.
- **Apilamiento AA.** A estas separaciones el registro entre capas es
  invisible para una molécula de H₂, y AA mantiene todos los huecos
  idénticos, que es el propósito de la serie.
- La rejilla no resuelve dónde está el máximo **dentro** de cada régimen
  de capas, sólo entre cuáles dos puntos cae.
- Las superestructuras encogidas se comparan con la rendija igualando la
  esfera libre mayor. Medida la distancia media a la pared (ver arriba),
  esa manera de igualar resultó **no** ser la causa del déficit, y la
  causa sigue sin establecerse.
- Tres puntos (H = 6, 9 y 16) traen el aviso de deriva del muestreo. Ese
  aviso está señalado como falso positivo en
  `validacion_superestructuras/README.md`: compara el primer bloque de
  producción con el último, que es la comparación más ruidosa posible con
  esos datos, y triplicar el equilibrado invirtió su signo sin mover la
  media.

## Reproducir

```bash
python construye.py <carpeta>              # las diez rendijas
python revisa.py   <carpeta>               # enlaces bajo PBC + porosidad
carbonforge adsorcion isoterma <carpeta>/rendija_10.extxyz <carpeta>/iso_10 \
    --temperatura 77 --presiones 1,3.8,10 \
    --equilibrado 1500000 --produccion 1500000 --instantaneas 1
python -m carbonforge.adsorption.run <carpeta>/iso_10
python informe.py <carpeta>                # la tabla, con las seis anteriores
python figura.py                           # ancho_vs_exceso.png
```

## El pozo de un surco: `surco.py`

`curvatura.py` cerró el *por qué* del déficit por m² de las
superestructuras — su superficie es sobre todo el EXTERIOR de un tubo, y
una pared convexa tiene menos carbonos al alcance de un punto que una
lámina plana de la misma densidad: **76.4 %** del pozo plano para un
(6,6). Eso deja abierta la pregunta de diseño, porque el adsorbente de
una superestructura no es una pared: donde dos tubos corren paralelos
dejan un **surco**, y un punto en la hendidura ve dos paredes convexas a
la vez. `surco.py` mide las tres geometrías con lo único distinto siendo
la forma (mismo carbono, misma sonda, 77 K, mismo barrido).

| hueco (Å) | rendija (K) | surco (K) | surco / rendija | surco / pared plana |
|---|---|---|---|---|
| 3.0 | vacía | -700.9 | -- | 155.4 % |
| 3.4 | vacía | -696.6 | -- | 154.5 % |
| 4.0 | vacía | -691.0 | -- | 153.2 % |
| 5.0 | vacía | -691.9 | -- | 153.4 % |
| 6.0 | -766.1 | -703.9 | 91.9 % | 156.1 % |
| **7.0** | **-817.4** | -611.6 | 74.8 % | 135.6 % |
| 8.0 | -604.5 | -413.6 | 68.4 % | 91.7 % |
| 10.0 | -488.5 | -174.8 | 35.8 % | 38.8 % |
| 12.0 | -463.9 | -81.9 | 17.7 % | 18.2 % |

Pared plana suelta: **-451.0 K**. Pared convexa (6,6) suelta: -344.5 K.

Dos conclusiones, y van en direcciones opuestas:

* **El surco paga la convexidad con creces.** -700.9 K contra los
  -344.5 de la pared desnuda: **2.03 veces**, y **155 %** de una lámina
  plana. El déficit del 24 % que `curvatura.py` midió no es el final de
  la cuenta; dos paredes convejas enfrentadas superan a una plana.
* **Y sigue sin ganarle a una rendija bien elegida.** El mejor surco
  (-703.9 K a 6 Å) es el **86 %** del mejor pozo de rendija (-817.4 K a
  7 Å). La superestructura no es un adsorbente mejor que el carbón de
  rendijas; es uno comparable con una red abierta, que es otro
  argumento.

**Dos cotas de barrido eran el resultado antes de arreglarlas**, y las
dos daban números de la superficie equivocada:

* El barrido de la rendija salía por la pared opuesta y encontraba su
  mínimo en el **exterior de una bicapa**. Así una rendija de 3.0 Å
  —que no admite nada— se informaba a -503.7 K. Acotado al interior,
  sale `vacía`, que es la respuesta.
* El del surco empezaba a 0.5 Å del eje de simetría, y con hueco ancho
  el sitio más profundo **es el eje**: todos los huecos de 7 Å en
  adelante informaban su mínimo exactamente en el suelo del barrido. El
  surco se barre desde 0.

```bash
uv run python validacion_rendijas/surco.py
```

## ¿A qué presión llena cada geometría? `llenado.py`

`curvatura.py` y `surco.py` miden la **profundidad** de un pozo. Eso no
es la respuesta de diseño: lo que se elige es cuánto se adsorbe a la
presión disponible, y a baja presión eso es la **pendiente inicial** de
la isoterma — el coeficiente de Henry, que es una media de Widom sobre
la celda vacía y cuesta segundos donde una isoterma cuesta días
(`adsorption/henry.py`).

Es el límite del **mismo** GCMC, no un segundo modelo: el mismo
reservorio (`VirialGas.activity`) y el mismo par H₂-C con corrección de
Feynman-Hibbs. `tests/test_adsorption_henry.py` fija las dos mitades de
esa afirmación: a 1 kPa el GCMC cae sobre la tangente (6.03 moléculas
frente a 6.19, un 2.5 %) y a 10 kPa ya se ha separado un 25 %, que es
donde acaba el régimen lineal.

Dos geometrías, un mando cada una, misma química:

| | mejor pendiente | K_H (mol/kg/MPa) | pozo más profundo | q_st (kJ/mol) |
|---|---|---|---|---|
| rendija | H = 6.6 Å (canal 3.5 Å) | **8660 ± 50** | H = 6.4 Å | 7.81 |
| haz de tubos (surcos) | hueco = 4.0 Å | 4252 ± 107 | hueco = 3.8 Å | **8.96** |

**Las dos columnas no premian a la misma geometría, y eso es el
resultado.**

* **El surco gana en profundidad de pozo**, 8.96 contra 7.81 kJ/mol — y
  gana por más de lo que `surco.py` sugería. La razón es geométrica y
  corrige el encuadre de esa sección: el surco de un haz triangular es
  una cuña de **tres** tubos, no de dos, y el punto en ella ve tres
  paredes convexas. La hipótesis del surco queda confirmada en su mejor
  versión.
* **La rendija gana en pendiente por gramo**, 8660 contra 4252: un
  factor **2.04**. Y el
  motivo no es el pozo: `K_H` es **por kilogramo de armazón**, y un haz
  gasta la mayor parte de su carbono en paredes de tubo que miran a un
  interior vacío, mientras que en una rendija cada carbono da a un poro
  que trabaja. **Sitios mejores, y menos de ellos por gramo.**

Así que la respuesta a "¿llena antes una red abierta?" es **no** — pero
por un factor 2, no por el 86 % que sugería comparar pozos de dos tubos,
y el límite real no es la convexidad sino la **densidad de sitios por
gramo**. Que es un problema distinto y con otra cura: no ensanchar el
poro, sino adelgazar el armazón.

Tres cosas que el barrido enseñó y que un barrido grueso habría
escondido:

* **`K_H` va como `exp(q_st/RT)` y `RT` es 0.64 kJ/mol a 77 K**, así que
  un pozo 1.8 kJ/mol más profundo es un factor 16. La serie gruesa de
  `construye.py` (6, 7, 8, 10, 14, 20 Å) **se salta el óptimo de la
  rendija**: da 760 a 6.0 y 717 a 8.0 con 8660 en medio. Un barrido
  demasiado grueso no desenfoca el óptimo, informa de otro adsorbente.
* **El 6.0 Å del haz que usa la referencia no es su óptimo.** Levesque
  probó 3.4 y 6.0 y encontró 6.0 mejor de los dos, lo cual es cierto;
  el máximo está en 4.0 Å, con `K_H` 5.5 veces el de 6.0 (4252 contra 775). Las
  isotermas ya corridas en este repositorio están todas en 6.0.
* **El error propio de la media de Widom subestima el del haz.** Las
  semillas coinciden al 1 % en una rendija y al 6 % en un haz cerca de
  su pico, porque allí los sitios profundos son una fracción pequeña de
  una celda grande. `llenado.py` corre dos semillas y publica la
  separación entre ellas, no el error interno.

Lo que esto **no** mide: super-grafeno ni supertubo. Esos llevan nodos,
y un nodo no es ni surco ni rendija. El haz es el adsorbente de surcos
en su forma idealizada —todo surco, ningún nodo— y, a diferencia de
cualquier superestructura de este repositorio, tiene una isoterma
publicada detrás (`validacion_GCMC/`).

Tampoco mide una presión de media carga: eso necesita una carga de
saturación, que sale de una isoterma y no del límite de cobertura cero.
`HenryReport.half_filling_pressure` la calcula si se le da una, y dice
que el supuesto de Langmuir es un supuesto.

```bash
uv run python validacion_rendijas/llenado.py
```

## Y las superredes de verdad: `llenado_super.py`

`llenado.py` compara una pared plana con un surco **idealizado** —todo
surco, ningún nodo— porque una superred lleva nodos y un nodo no es ni
surco ni rendija. Esta es la cosa real: súper-cuadrado, súper-grafeno,
súper-cúbico y súper-fcc, construidas con `nanocarbon_lab` en sus
presets de catálogo y entregadas por fichero (ningún paquete importa al
otro), empotradas en celda legal aquí.

A 77 K, un millón de inserciones por dos semillas:

| estructura | átomos | canal (Å) | área (m²/g) | K_H (mol/kg/MPa) | q_st (kJ/mol) |
|---|---|---|---|---|---|
| **rendija, H = 6.6 Å** | 960 | 3.55 | — | **8660 ± 50** | 7.81 |
| súper-cúbico | 836 | 20.00 | 2314 | 4956 ± 101 | **8.90** |
| haz de tubos, hueco 4.0 Å | 8704 | 10.21 | — | 4252 ± 107 | **8.96** |
| súper-cuadrado | 1328 | 20.22 | 2351 | 4225 ± 5 | 8.47 |
| súper-grafeno | 2360 | 18.46 | 2249 | 4057 ± 35 | 8.56 |
| súper-fcc | 2982 | 7.33 | 1760 | **633 ± 6** | 7.30 |

Cuatro cosas, y ninguna era la esperada:

* **El nodo no estropea nada.** Tres de las cuatro caen exactamente
  donde cae el haz idealizado (4000–5000), y el súper-cúbico queda por
  encima de él. Los nodos eran la razón para no extrapolar del haz a la
  superred; medido, la extrapolación valía. El nodo es una minoría de
  la pared y no estrangula el canal.
* **El súper-fcc es la excepción, y pierde por un factor 7.** Su canal
  es de 7.33 Å contra 18–20 Å de las otras y su área 1760 contra
  2250–2350: doce tubos en un nodo no dejan sitio para una pared entre
  ellos, que es lo que el catálogo de `nanocarbon_lab` ya dice de él
  («se informa, no se recomienda»). Aquella advertencia era sobre la
  geometría de la pared; **esto es la misma conclusión llegando desde
  la adsorción, que es una comprobación independiente.**
* **La rendija óptima sigue ganando la pendiente por gramo, 1.75×, con
  el pozo MÁS SOMERO de la tabla salvo el fcc.** Esa es la frase que
  resume todo el estudio: `K_H` ≈ densidad de sitios × `exp(q/RT)`, y
  la rendija tiene sitios mediocres **en todas partes** mientras las
  superredes tienen sitios excelentes en unos pocos. Gana el que tiene
  más sitios por gramo, no el que tiene los mejores.
* **El orden no sigue ni al área ni al canal.** El súper-cuadrado tiene
  el área mayor (2351 m²/g) y el canal más ancho (20.22 Å) y no es el
  mejor; el súper-cúbico, con menos área y canal más estrecho, lo
  supera. `K_H` sigue a `q_st`, no al área — que es por qué el área
  geométrica no es la cifra de mérito a baja presión y por qué
  `curvatura.py` buscaba en el sitio correcto al mirar pozos.

Dos notas de método que costaron una medida cada una:

* **Estas celdas son casi todas vacío, así que su media de Widom
  converge más despacio que la de una rendija** y a 200 000
  inserciones el propio informe avisaba (5–7 % de error en el
  súper-cúbico y el fcc). A 10⁶ el aviso desaparece y los valores
  apenas se mueven (4862 → 4956, 640 → 633), o sea que el aviso era
  conservador y aun así había que atenderlo. El número de muestras es
  un argumento del script precisamente porque la respuesta al aviso es
  subirlo.
* **A 10⁶ puntos el proceso moría sin traza**, porque `query_ball_point`
  sobre todos los puntos a la vez guarda un millón de listas de
  vecinos. Es la misma clase de fallo que el campo de red sin trocear
  de `implicit.py` y la misma cura: la evaluación va en bloques
  (`henry.CHUNK`). Los puntos se sortean de una vez, así que el tamaño
  de bloque no puede cambiar la respuesta, y un test lo fija a 1e-12.

El súper-cúbico se construye con el rescate de pared colapsada activo
(`nanocarbon_lab` avisa y lo reconstruye con la pared sujeta a su
superficie); lo que se mide aquí es la estructura rescatada, no la
colapsada.

```bash
# en nanocarbon_lab, que escribe los extxyz:
#   python construye_super.py <carpeta>
uv run python validacion_rendijas/llenado_super.py <carpeta> 1000000
```
