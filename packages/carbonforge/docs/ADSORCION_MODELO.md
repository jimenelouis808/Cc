# Hasta dónde llega este modelo, y qué hace falta para lo siguiente

El prototipo de `adsorption/` reproduce las isotermas de Levesque en
dieciséis puntos y dos temperaturas dentro de ±6 %. Eso acota lo que
está bien y, por lo mismo, acota lo que no: **lo validado es el
muestreo, no el modelo.** Que salgan sus números significa que el
potencial, la corrección cuántica, la actividad del reservorio y el
balance de las jugadas están bien puestos — no que un hidrógeno de un
sitio sin cuadrupolo sea cierto.

Este documento dice qué hay que cambiar para carbonos dopados y
funcionalizados (O, N, B, P, Se) y para otros adsorbatos, en el orden en
que conviene hacerlo. Cada punto dice **qué medida lo delataría**, para
que nada entre sin una comprobación.

## El estado actual, en una tabla

| Pieza | Qué es hoy | Dónde se rompe |
|---|---|---|
| H₂ | un sitio Lennard-Jones, sin carga | un dopante cambia el campo eléctrico local y el modelo no lo ve |
| Armazón | un solo σ, ε para todo átomo | no hay forma de escribir un N o un B |
| Electrostática | **no existe** | es el mecanismo principal del dopado |
| Cuántica | Feynman-Hibbs orden ħ² | bien para H₂ ≥ 77 K; dudoso por debajo |
| Armazón | rígido | correcto para grafito; no para un grupo funcional colgante |
| Reservorio | virial de dos términos a 77/150/293 K | sólo H₂, y sólo tres temperaturas |

## 1. Parámetros por elemento (lo que desbloquea O, N, B, P, Se)

Hoy `H2_C` es una pareja fija. Hace falta un **catálogo por elemento**
con σ y ε propios, y mezcla de Lorentz-Berthelot, que es como ya está
construido el H₂–C: `σ_sf = (σ_ff + σ_ss)/2`, `ε_sf = sqrt(ε_ff·ε_ss)`.
El carbono ya está medido en este repositorio como σ = 3.40 Å, ε/k =
28 K, y una prueba comprueba que mezclarlo con el H₂ devuelve el par que
usa Levesque — así que el mecanismo está verificado antes de añadir
nada.

Esto es barato y mecánico. **No basta**, y el punto siguiente es por qué.

## 2. Electrostática: sin esto, el dopado no se modela, se decora

Un nitrógeno grafítico no adsorbe más hidrógeno porque sea *más grande*.
Lo hace porque redistribuye la carga y crea un campo que interacciona
con el **cuadrupolo** del H₂. Un modelo que sólo cambia σ y ε mueve la
isoterma unos puntos porcentuales y por la razón equivocada, y lo peor
es que seguirá pareciendo razonable.

Lo que hace falta, en orden:

1. **El cuadrupolo del H₂ como tres cargas puntuales.** Levesque lo
   describe y no está implementado: `q = 0.4829 e` en los protones y
   `−2q` en el centro. Esto convierte la molécula de una esfera en algo
   con orientación, y por tanto añade jugadas de **rotación** al Monte
   Carlo.
2. **Cargas parciales en el armazón.** `forcefields/charges` ya existe
   en este paquete para derivarlas del DFT del usuario, y la CLAUDE.md
   ya advierte que las de carbono dopado son representativas y no están
   asentadas. Esa advertencia es exactamente la que aplica aquí.
3. **Sumas de Ewald**, porque la electrostática no se puede cortar a
   12 Å como el Lennard-Jones: decae como 1/r y la suma truncada
   depende del corte.

**La medida que lo delataría:** el cuadrupolo tiene que **bajar** poco
la adsorción en un armazón apolar y subirla en uno dopado. Si al
añadirlo el haz de SWNT de Levesque se mueve más de un par de por
ciento, está mal implementado — y ese haz es precisamente el control,
porque su respuesta correcta ya está medida.

## 3. Otros adsorbatos: el salto es de un sitio a varios

| Adsorbato | Modelo mínimo creíble | Qué añade |
|---|---|---|
| N₂ | dos sitios + cuadrupolo | orientación |
| CO₂ | tres sitios + cuadrupolo grande | orientación, y la electrostática pasa a dominar |
| CH₄ | un sitio (unido) sirve de verdad | casi nada: es el siguiente después del H₂ |
| H₂O | carga + enlace de hidrógeno | otro problema; no lo trate este módulo sin decirlo |

**CH₄ es el que conviene hacer primero** y no por facilidad: es
esférico y apolar, así que valida el camino del reservorio y del
muestreo para una molécula que no es H₂ **sin** meter electrostática a
la vez. Si CH₄ sale y H₂ seguía saliendo, el siguiente fallo que
aparezca es de la electrostática y no de la plomería.

Cada adsorbato nuevo necesita **su propio reservorio**: `bulk.py` ajusta
un virial a densidades medidas de hidrógeno, y para CH₄ o CO₂ hacen
falta las suyas. El módulo ya refuerza esto rechazando una temperatura
sin tabla en vez de extrapolar.

## 4. Un armazón que se mueve

Rígido es correcto para grafito y para un nanotubo. Deja de serlo para
un grupo funcional colgante: un −OH o un −COOH tiene grados de libertad
propios que cambian a qué distancia puede acercarse el adsorbato. La
salida sensata no es relajar todo el armazón en el GCMC, sino **dejar
flexibles sólo los grupos** y mantener rígida la red.

## 5. Lo que hay que arreglar antes que nada de lo anterior

**El coste.** `fluid_energy` es O(N) por movimiento, con dos copias de
array por llamada, así que un punto a 77 K cuesta de 31 a 47 minutos.
Todo lo de arriba lo empeora: el cuadrupolo multiplica por tres los
sitios y añade rotaciones, y Ewald es más caro que el Lennard-Jones.
Antes de añadir física hay que pagar esto:

- **lista de celdas** para el fluido, que baja la energía de un
  movimiento de O(N) a O(vecinos);
- **cachear la energía de armazón por molécula**, que ahorra la consulta
  al árbol en cada borrado y la mitad de cada traslación;
- **rejilla de energía del armazón**, que es la optimización grande y la
  que hay que hacer con cuidado: interpola en la parte empinada de la
  pared, que es justo donde se decide una inserción. La isoterma de
  Levesque es la prueba de si la interpolación se nota.

Las tres son comprobables contra los dieciséis puntos ya medidos: si una
de ellas mueve la isoterma, es un error, no una aceleración.

## Orden recomendado

1. Coste (lista de celdas, caché de armazón). Sin resultados nuevos,
   pero sin esto lo demás no es practicable.
2. Catálogo de σ, ε por elemento. Barato, y necesario para todo.
3. CH₄, con su reservorio. Valida la plomería sin electrostática.
4. Cuadrupolo del H₂ + jugadas de rotación, con el haz de Levesque como
   control de que no se movió lo que no debía.
5. Cargas del armazón y Ewald. Aquí empieza el dopado de verdad.
6. Grupos flexibles.

Nada de esto convierte el prototipo en un código de producción, y ese
tampoco es el objetivo. El objetivo es que cada número que salga venga
con la medida que lo respalda, que es lo que el caso de Levesque
estableció y lo que cada paso de arriba tiene que mantener.
