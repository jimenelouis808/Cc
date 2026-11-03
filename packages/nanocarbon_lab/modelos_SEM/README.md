# Modelos estructurales para las morfologías SEM de FeSe@C

Construidos con `nanocarbon_lab` a partir de los tres CIF de `cif/`
(procedencia en `cif/PROCEDENCIA.md`).

```bash
uv run python modelos_SEM/construir.py
```

Todo sale en `xyz/` (coordenadas) e `img/` (dos vistas por modelo: una
oblicua y otra por el eje, con sufijo `_eje`).

## Lo primero: qué es una hipótesis aquí y qué no

**Cuál fase es cuál sigue sin decidirse, y estos modelos no lo deciden.**
Lo que hacen es ponerlas una al lado de la otra con el hábito que cada
grupo espacial da de forma natural, para que la comparación con el SEM
sea una comparación y no un recuerdo:

- la **placa hexagonal** se corta del FeSe hexagonal (P6₃/mmc, tipo
  NiAs). Un prisma hexagonal es el hábito que esa simetría regala, y es
  exactamente tu razonamiento — queda escrito como lo que es, una
  sospecha hecha geometría.
- la **barra** se corta del FeSe tetragonal (P4/nmm, tipo PbO).

Nada en el modelo prueba la asignación. **Lo que la zanja es la
difracción**: la 101 del tetragonal está a 28.6° y la del hexagonal a
32.4°, catorce veces la anchura de un pico. Si tienes DRX de la muestra
de 900 °C, eso lo cierra en una tarde; el SEM solo lo sugiere.

## Los modelos

| modelo | átomos | fórmula |
|---|---|---|
| `placa_H_nucleo_desnudo` | 1626 | Fe813Se813 |
| `placa_H_C` | 4976 | C3350Fe813Se813 |
| `placa_H_NSe` | 4976 | C3065Fe813N201Se897 |
| `placa_H_NSeO` | 5014 | C3068H15Fe813N204O13Se901 |
| `placa_H_delgada_desnuda` | 1086 | Fe546Se540 |
| `barra_T_nucleo_desnudo` | 504 | Fe252Se252 |
| `barra_T_C` | 2322 | C1818Fe252Se252 |
| `barra_T_NSe` | 2322 | C1663Fe252N109Se298 |
| `barra_T_NSeO` | 2337 | C1664H6Fe252N110O5Se300 |
| `particula_Fe3C_nucleo_desnudo` | 578 | C156Fe422 |
| `particula_Fe3C_C` | 2838 | C2416Fe422 |
| `particula_Fe3C_NSe` | 2838 | C2224Fe422N136Se56 |
| `particula_Fe3C_NSeO` | 2863 | C2226H10Fe422N138O9Se58 |
| `particula_Fe3C_interior` | 55 | C17Fe38 |
| `nanotubo_C` | 3000 | C3000 |
| `nanotubo_NSe` | 3000 | C2745N180Se75 |
| `nanotubo_NSeO` | 3030 | C2747H12N182O10Se79 |
| `brazo_particula_dentro` | 3055 | C3017Fe38 |
| `escena_anemona` | 24440 | C24136Fe304 |
| `escena_barras` | 37152 | C29088Fe4032Se4032 |
| `escena_profundidad` | 10353 | C8185Fe1103Se1065 |
| `escena_anemonas` | 54990 | C54306Fe684 |
| `brazo_decorado` | 3565 | C3017Fe548 |
| `escena_anemona_decorada` | 28520 | C24136Fe4384 |

### Las tres versiones

Cada objeto de carbono viene en tres, y **la geometría de abajo es la
misma en las tres**: el núcleo se corta una vez y la coraza se malla una
vez, así que lo único que cambia entre dos versiones es la química. Si se
construyeran por separado, el mallado diferiría y la comparación no
mediría nada.

| sufijo | qué lleva |
|---|---|
| `_C` | carbono puro. El control. |
| `_NSe` | coraza codopada N + Se, sustitucional, **en una sola pasada** |
| `_NSeO` | lo anterior más grupos de superficie O/N/Se en la cara externa |

**Por qué una sola pasada**: las dos fracciones son del mismo
denominador. Sustituir N y luego Se haría que la segunda fracción fuera
una fracción de lo que dejó la primera.

**Las fracciones**: N al 6 %, Se al 2.5 %. No son libres — el catálogo de
`dopants/chemistry.py` pone el techo del Se en el 3 % («más mismatch que
el S y correspondientemente más raro») y el del N en el 20 %. El Se es el
que manda y 2.5 % está justo debajo de su techo. Correlación `avoid`,
así que no quedan dos heteroátomos enlazados: N y Se no son una pareja
dador-aceptor como B–N, no hay razón para que se junten, y juntarlos
concentraría la tensión que el Se ya trae (es un 48 % más grande que el
carbono y saca su sitio del plano).

**Los grupos**, en el orden en que se injertan y con su cobertura:
hidroxilo 5 %, carboxilo 3 %, carbonilo 3 %, amina 3 %, **selenol** 2 %,
**selenona** 2 %. Cada pasada ve los sitios que dejó la anterior, así que
la cobertura cae sola — que es lo que hace un carbono oxidado de verdad,
unos pocos por ciento y no una monocapa.

**La selenona y el selenol no están en el catálogo**, y no hacía falta
añadirlos: `functionalize.groups.substitute` cambia el elemento y
**reconstruye las longitudes de enlace**, porque la biblioteca solo
guarda ángulos. El selenol es el hidroxilo con O→Se y la selenona es el
tiocarbonilo con S→Se, así que el C=Se sale con la longitud del selenio y
no con la del azufre.

## Lo que el programa avisó, y que hay que leer antes de usar esto

**1. La interfase Fe–C no está decidida por nadie.** El aviso sale en los
tres núcleos con hierro:

> el núcleo contiene Fe, que forma carburos estables: el carbono sobre
> ellos quimisorbe a unos 2.1 Å en vez de quedarse en el contacto de van
> der Waals de 3.45 Å. Las dos estructuras existen —el contacto es lo
> que una coraza grafítica tiene en la mayor parte de su área— pero son
> interfases distintas y nada aquí calcula cuál prefiere tu sistema.

Estos modelos usan el contacto de van der Waals. **Para tu pregunta de
si la partícula está dentro o fuera del tubo eso importa**: una partícula
que nucleó el tubo y quedó en su punta estuvo en contacto quimisorbido
mientras crecía.

**2. El Fe₃C sale 6.2 % fuera de estequiometría** y el programa lo llama
`strained`. No es un fallo del corte: el carbono de la cementita es
intersticial y está más adentro, así que cualquier superficie que cortes
termina en hierro. Baja al 5 % si agrandas la partícula y nunca llega a
cero. El aviso habla de «cluster cargado», que es el caso iónico; la
cementita es metálica, así que ahí la advertencia no aplica tal cual —
pero el desbalance sí es real y hay que declararlo si publicas una
composición.

**3. La estequiometría del FeSe se forzó quitando átomos.** 92 en la
placa. «Los menos coordinados» es la receta habitual y es una heurística:
menos enlazado por número de vecinos no es menos enlazado por energía.

**4. En la placa, unos pocos carbonos quedan a 1.68 Å del núcleo.** El
percentil cinco está en 2.33 Å, así que son unos pocos átomos y no la
pared: es el borde afilado del prisma hexagonal. Un borde real se
reconstruye ahí y este modelo no. Ya subí el `wall_anchor` de 0.7 a 1.0
por eso; si te molesta, lo honrado es redondear el canto del núcleo, no
dejar que la pared se desplome.

**5. El hexagonal es deficiente en hierro en la vida real** (Fe₁₋ₓSe, y a
composiciones concretas ordena como Fe₇Se₈ o Fe₃Se₄ con superestructura).
Estos modelos lo cortan 1:1, que es la celda ideal.

## La anémona, y lo que cambió en la segunda versión

| | v1 | v2 |
|---|---|---|
| átomos | 32 730 | **22 860** |
| brazos | 5 | **10** |
| brazos | rectos | **curvados** |
| partícula | fuera, pegada a la punta | **dentro de la punta** |
| radio del tubo | 7.85 Å | **11.76 Å** |

**La partícula dentro obligó a ensanchar el tubo, no a encoger la
partícula**, y esa dirección es un resultado: por debajo de unos 126
átomos una partícula de cementita deja de ser cementita — 70 átomos
vuelve `broken` y 36 también — así que la pieza pequeña es la que no se
puede encoger. Un tubo de 7.85 Å de radio deja 4.5 Å de hueco y la
partícula más pequeña que sigue siendo cementita llega a 6.57 Å del eje.
Con `freq = 6` el tubo tiene 11.76 Å de radio y 8.41 Å de hueco, y la
holgura medida es **1.84 Å**. El ensamblaje la mide y se niega si no
cabe, en vez de meterla igual.

**Los brazos se curvan con `shape="arc"`, no con `bend_angle`.** El
segundo impone el arco después de relajar y lo paga en tensión
(1.372–1.529 Å de enlace a 0.6 rad); el primero pasa por el presupuesto
de deformación de `centerline.py`, que recorta la amplitud hasta que la
pared aguanta, y vuelve en **1.393–1.447 Å**. La curva es gratis si se
pide por el camino correcto.

**Y arrancan a cuatro radios del centro.** A uno, cuatro pares de brazos
quedaban a 2.4–3.1 Å, que en un render se lee como fundidos. Van
repartidos sobre la esfera por el método de Fibonacci y no en un plano:
una anémona no es una flor.

Sigue siendo una **ilustración**, y el programa lo dice en cada una:

> esto es una ILUSTRACIÓN, no una estructura: los objetos se colocaron a
> mano, no hay ninguna interfase relajada entre dos de ellos, y nada en
> los huecos es física. No lo cuantifiques, no lo relajes y no se lo des
> a un código DFT.

## La placa: 8.9 Å, y el techo lo pone la coraza

| grosor | ancho | relación | qué pasa |
|---|---|---|---|
| 3.0 Å (media celda) | 45 Å | 15:1 | **no es FeSe**: termina Se–Se, sale Fe110Se200 |
| 6.0 Å (una celda) | 58 Å | 9.8:1 | el núcleo está bien; **la coraza se rompe** |
| **8.9 Å** | **58 Å** | **6.5:1** | **lo que se entrega** |
| 11.9 Å (dos celdas) | 58 Å | 4.9:1 | la primera versión |

**Lo que impide bajar de 8.9 Å no es el cristal, es el recubrimiento.**
A 6.0 Å la coraza grafítica vuelve rota con todos los huecos y rigideces
probados —enlaces hasta 1.73–1.76 Å contra el techo sp² de 1.55— y el
mensaje del propio constructor dice por qué: *«la superficie tiene
detalles más finos que un anillo de carbono»*. Un canto de 6 Å es más
fino que la pared que tendría que envolverlo. En una muestra real ese
borde es donde el recubrimiento está defectuoso o simplemente abierto.

Por eso va también `placa_H_delgada_desnuda`: 6.0 Å de grosor y 58 Å de
ancho, **9.8:1**, sin cubrir. Es la más delgada que el cristal permite y
la que hay que comparar con la micrografía.

## Todo es un parámetro

```bash
uv run python modelos_SEM/construir.py --help
uv run python modelos_SEM/construir.py --barra-semilargo 60 --brazos 14
```

Cada entrada de `DEFAULTS` es una opción con el mismo nombre, así que
añadir una perilla es añadir una línea a esa tabla. Lo que se puede
barrer sin tocar el código: radio y grosor de la placa, radio y largo de
la barra, los semiejes y el número de capas de la partícula de
cementita, la partícula que va dentro del tubo, la frecuencia (= el
radio, cuantizado por la red) y los anillos y la curvatura del tubo, el
número de brazos y su separación, las fracciones de N y de Se, y del
recubrimiento el hueco, el ancla, la rejilla, el remallado, la
relajación y la semilla.

De las escenas: `--barras` (cuántas), `--barras-separacion` (en unidades
del largo de la propia barra), `--barras-holgura` (los Å que cada barra
debe dejar libres) y `--barras-intentos` (cuántas tiradas antes de
quedarse con la mejor); `--anemonas-separacion`; y de la decoración
`--fe-radio`, `--fe-particulas` y `--fe-contacto`.

## Qué átomo es sólido y cuál no

Cada modelo escribe en su `bundle/<nombre>.json` una lista `solid` con
los índices del núcleo. La hoja de contacto y el render la leen, y por
eso el Se del FeSe sale en relleno de van der Waals mientras el Se
sustituido en la coraza sale en bolas y palos como sus vecinos.

**Es una REGIÓN y no un elemento, y tenía que serlo**: es el mismo
selenio en dos sitios —parte de un sólido en el núcleo, un átomo en una
red en la coraza— y ninguna regla por elemento los separa. Lo mismo vale
para el azufre y el oxígeno de los grupos funcionales. `--spacefill-element`
sigue existiendo para una estructura que no lleve la lista.

## La coraza lisa cuesta enlaces, y es una elección

La pared sale de mallar un sobre y relajarla. Dejada libre llega a la
longitud de enlace correcta **y se arruga**, porque el campo de fuerzas
no tiene término de flexión: ondular no le cuesta casi nada. El ancla la
sujeta sobre su propio sobre, que es lo que la hace parecer grafeno.

Medido sobre la barra, la rugosidad RMS contra el sobre y el peor enlace:

| ancla | rugosidad RMS | máx | enlace máx | |
|---|---|---|---|---|
| 0.0 | 0.580 Å | 3.01 | 1.523 | sp² OK |
| 1.0 | 0.381 | 2.37 | 1.592 | roto |
| **3.0** | **0.335** | **2.10** | **1.656** | roto |
| 8.0 | — | — | — | el constructor la rechaza |

**Va en 3.0.** Una pared 42 % más lisa a cambio de enlaces fuera del
rango sp². Es la elección que pediste para la portada, y el programa la
dice en voz alta al terminar cada vez: esos modelos sirven como
ilustración y no para relajar ni para DFT.

**El recocido se queda apagado**, y ahora con número: 40 pasadas mueven
la rugosidad de 0.335 a 0.327 y los no-hexágonos del 8.2 % al 8.6 %.
Compra un dos por ciento.

**La rejilla no entra en esto, y vale la pena saberlo** porque es lo
primero que uno toca cuando una malla sale mal. Barridas las doce
combinaciones de hueco (3.4 / 3.0) x ancla (3 / 1 / 0) x rejilla
(128 / 160) sobre la barra, **el veredicto no depende de la rejilla en
ninguna**: 128 y 160 dan el mismo enlace máximo hasta la centésima. El
hueco mueve poco (1.69 -> 1.63 Å al cerrarlo de 3.4 a 3.0) y **el ancla
lo mueve todo**: las cuatro combinaciones limpias de las doce son
exactamente las cuatro con ancla 0. Afinar la malla no arregla una
pared que está tensa porque se la está sujetando.

**Y si una rigidez no construye, se baja sola.** El constructor *levanta
una excepción* en vez de devolver una pared mala, así que una llamada
sin guardia pierde la corrida entera por una familia. Se prueba 3, luego
1, luego 0, y la que se usó se dice.

## La tabla puede mentir, y mintió

`modelos.json` se escribe entero en cada corrida, pero una corrida
parcial (`--solo`) **conserva las filas de las familias que no
construyó**, o perdería la mitad del catálogo. Eso tiene un modo de
fallo propio y pasó: los modelos de la familia Fe3C se reconstruyeron a
2838 átomos y su fila siguió diciendo 2770, porque ninguna corrida
posterior tocó esa familia. **Las estructuras estuvieron bien todo el
tiempo y la tabla fue la que mintió**, que es peor que equivocarse
cualquiera de los dos por separado: un `.xyz` malo se ve en el dibujo,
una tabla mala se cita en un artículo.

Ahora cada fila conservada se comprueba contra la primera línea de su
`.xyz` —que es el número de átomos— y se corrige diciéndolo en voz
alta. Verificado adrede: metiendo un 1234 a mano en la fila de
`particula_Fe3C_C`, la siguiente corrida parcial avisa y la devuelve a
2838.

## Para comparar: las hojas de contacto

```bash
uv run python modelos_SEM/contacto.py
```

Tres hojas en `img/`, y lo que las hace útiles es que **todos los
paneles están a la misma escala**:

| hoja | qué enseña |
|---|---|
| `_contacto.png` | todos los modelos de frente, 130 Å de lado |
| `_contacto_perfil.png` | los mismos de perfil — el grosor |
| `_contacto_escenas.png` | la anémona y el brazo, 380 Å de lado |

Las imágenes sueltas de `img/<modelo>.png` autoescalan cada una a su
propio sujeto, así que una partícula de 30 Å y una anémona de 300 llenan
el mismo cuadro y parecen del mismo tamaño. Eso está bien para mirar una
estructura y **mal para decidir si la barra es bastante larga**, que es
para lo que sirven éstas.

Dos decisiones de dibujo que no son cosméticas:

- **El carbono va pequeño y el metal a radio de van der Waals**, el
  mismo convenio que el render de Blender. Dibujado todo a vdW, la
  coraza tapa el núcleo entero y **todos los modelos recubiertos salían
  como una mancha negra** — justo la comparación que estas hojas existen
  para permitir.
- **De frente y de perfil son dos hojas.** Una placa de frente enseña su
  hábito y esconde su grosor, que es el número en discusión.

## La barra: 85 Å y 9 : 1

Era 30.6 × 9.4 Å, **3.2 : 1** — rechoncha, no una aguja. Ahora 85.0 ×
9.4 Å, **9 : 1**, con 504 átomos de núcleo contra 192.

**El grosor se queda en 9.4 Å por lo mismo que el de la placa**: por
debajo de unos 9 Å la coraza grafítica no tiene qué envolver. Una
`rod` (cápsula) y no un `cylinder`: el canto vivo de un cilindro deja al
sobre nada por donde pasar y el recubrimiento lo rechaza, que es
además lo que hace un canto real — se reconstruye.

## Las cinco escenas de portada, y la cámara

```bash
uv run python modelos_SEM/construir.py --solo escena   # las escenas
uv run python modelos_SEM/camara.py                    # lo que encuadraría
```

| escena | átomos | fórmula | qué es |
|---|---|---|---|
| `escena_barras` | **37 152** | C29088Fe4032Se4032 | 16 barras recubiertas, como el montón del SEM a 900 °C |
| `escena_profundidad` | 10 353 | C8185Fe1103Se1065 | placa, barra y tubo de pie y de perfil, en fila hacia el fondo |
| `escena_anemona` | 24 440 | C24136Fe304 | una anémona de 8 brazos |
| `escena_anemonas` | **54 990** | C54306Fe684 | tres anémonas juntas |
| `escena_anemona_decorada` | **28 520** | C24136Fe4384 | una anémona con 34 partículas de Fe por brazo, en ruido azul |

**Son ilustraciones, no estructuras.** `scene.compose` coloca piezas ya
terminadas: no hay interfase relajada entre dos de ellas y nada de los
huecos es física. El propio módulo lo avisa en cada escena y hay que
dejarlo dicho en el pie de figura.

### Las barras se sortean y se miden, no se colocan

Ajustar a mano el paso de la rejilla y el temblor contra el informe de
pares de la composición llevó el campo de dos pares interpenetrados
(0.26 y 0.98 Å) a uno (0.38 Å) **y de ahí no pasó**: con orientaciones
al azar a esta densidad siempre hay una tirada que cae encima de un
vecino, y separar más el campo es justo lo contrario de lo pedido.

Así que cada barra se **dibuja y se mide antes de quedarse**: hasta
`--barras-intentos` tiradas de rotación y temblor, y se acepta la
primera que deje `--barras-holgura` (2.6 Å) contra todas las anteriores;
si ninguna lo logra, se queda la mejor y el programa lo dice. Resultado
medido: **ningún par por debajo de 2.0 Å** —el umbral con el que
`scene.py` llama interpenetración— y el más cercano a 3.37 Å, que es un
contacto grafítico y es exactamente lo que un montón denso debe
aparentar.

### `auto` encuadra mal una escena de profundidad, y es por diseño

`camera.best_view` elige la dirección cuya silueta tiene **más área**,
que es la que enseña más estructura — correcto cuando el sujeto es la
forma, y al revés cuando la composición es una fila: la silueta más
grande de tres objetos en línea es la que los pone uno al lado del otro,
que es lo único que esa escena no quiere. Por eso hay una vista nueva,
`depth` = (0.35, −1.0, 0.30), que mira **casi a lo largo de la línea**,
inclinada lo justo para ver los de atrás por un lado del de delante en
vez de eclipsados. Es un nombre y no una búsqueda porque ninguna medida
de área la devolvería nunca. Llega sola a `render_cnt --view depth`.

Y la escena que la usa es ahora una **fila de póster**: cada pieza de
pie, de perfil, una detrás de otra. Las tres rotaciones no se escriben
a mano sino que se **miden** (`standing_angles`): se toman los ejes
principales de cada pieza y se mandan el más largo a +z, para que se
ponga de pie, y el más corto a −y, para que su cara ancha mire a la
cámara. Para la placa eso enseña el hexágono y para la barra y el tubo
su longitud — en los tres casos la silueta por la que se reconocen. Lo
contrario de tres ángulos de tres cuartos distintos, que es lo que se
leía como tres objetos sueltos en el espacio.

| escena | vista | cámara a | relleno | en cuadro |
|---|---|---|---|---|
| `escena_barras` | auto | 585 Å | 1.18 | 99.7 % |
| `escena_profundidad` | **depth** | 267 Å | 1.10 | 100 % |
| `escena_anemona` | auto | 679 Å | 0.95 | 100 % |
| `escena_anemonas` | auto | 986 Å | 1.12 | 97.9 % |
| `escena_anemona_decorada` | auto | 655 Å | 1.00 | 100 % |

**Un relleno por encima de 1 recorta a propósito**: el sujeto se encuadra
más grande que el cuadro y sus bordes se salen, que es lo que «más zoom»
significa en una portada — la composición se sale de la página. La vista
previa dice qué porcentaje de átomos queda dentro, así que lo que se
corta es un número y no una sorpresa.

**Y la vista previa no es un render**: sin luces, sin materiales, sin
profundidad de campo. Usa las mismas llamadas que el render (`best_view`
o `named_direction`, `frame_camera`, `basis_for`) con la misma lente y el
mismo relleno, así que contesta «¿está el sujeto en cuadro, del tamaño
correcto, desde un ángulo que lo enseñe?» y nada más. Un render de
25 000 átomos cuesta minutos; esta pregunta no cuesta ninguno.

### Las partículas de la anémona decorada: tres repartos, dos malos

Lo que la micrografía enseña no es ni azar ni red. Es **ruido azul**:
ni grumos, ni filas, ni ángulo preferido.

1. **Azar uniforme.** Se agrupa. A esta cantidad los huecos son tan
   grandes como las partículas, el ojo encuentra los grumos primero y
   se lee como suciedad.
2. **Aros a lo largo del cañón**, desfasados medio paso. Arregla los
   grumos y mete un artefacto peor: las partículas se alinean en
   **anillos** visibles alrededor del tubo, que la micrografía no
   tiene. Una red es *pareja*, y pareja no es lo mismo que uniforme.
3. **Ruido azul sobre los átomos de la pared**, que es lo que hay
   ahora: para cada asiento se sortean `DECORATION_CANDIDATES` átomos
   de pared y se queda el más lejano de todos los asientos ya tomados.
   Aleatorio, así que lleva `--semilla`; determinista dada una.

**Los asientos son átomos de la pared y no puntos de un cilindro**, y
ésa es la parte que estaba mal primero. El brazo está **doblado**:
parametrizarlo por altura y ángulo alrededor de su propio eje
principal es la parametrización de un cilindro, y sobre un tubo curvo
estira por fuera del codo y pliega por dentro. Las separaciones
medidas eran honestas y el dibujo seguía saliendo con grumos, porque
se estaban midiendo en el espacio equivocado. La distancia entre dos
átomos no depende de cómo esté doblado el tubo.

La normal de cada asiento se mide igual de local: es la dirección de
menor varianza del vecindario del átomo, con el signo fijado contra el
vector que sale del centro de un tramo corto de tubo. Medido en la
versión que se entrega: 34 partículas, **9.2 Å** entre las dos más
cercanas y la más hundida a **3.37 Å** de la pared, que es el contacto
grafítico.

## Elegir el estilo sale barato; renderizar la escena no

```bash
BPY=<python con bpy> bash modelos_SEM/pruebas_estilo.sh            # el brazo
BPY=... RES="700 875" SAMPLES=24 bash modelos_SEM/pruebas_estilo.sh
```

**El coste está en el número de átomos, casi no en las muestras ni en
la resolución.** Medido en este contenedor, los siete presetes sobre el
brazo de 3565 átomos a 700×875 y 24 muestras tardaron **entre 10 y 11
minutos cada uno**, con una dispersión de un minuto entre el más barato
y el más caro. La escena de 10 353 átomos **no cabe en la media hora**
que el ejecutor en segundo plano permite por defecto y hay que lanzarla
con el tope ampliado. La malla se construye una icoesfera y un cilindro a la
vez desde Python, así que una escena de 10 353 átomos tarda lo mismo
pida lo que pida, y una de 55 000 es horas. Por eso las pruebas de
estilo van sobre `brazo_decorado` —3565 átomos— que lleva exactamente
los tres materiales en discusión: la coraza grafítica, las partículas
de Fe encima y la partícula de la punta. Se elige el estilo ahí y se
gastan las horas una sola vez.

### `core_shell_cover`, el preset escrito para esta muestra

Los 37 presets del catálogo están pensados para otras cosas —una jaula,
una red periódica, un cristal transparente—. Éste parte de lo que la
muestra es: **una coraza oscura de carbono con metal brillante dentro o
encima**, que es lo que enseña la micrografía y lo que una portada de
ella tiene que conservar.

- **Fondo casi negro, pero no negro.** Un fondo muerto se traga el lado
  oscuro de un sujeto oscuro, y el rim por sí solo no sostiene un borde
  que no tiene nada detrás.
- **El rim es la pieza que carga con todo.** Ocho brazos del mismo gris
  oscuro se superponen; sin un borde en cada silueta se funden en una
  sola mancha. El rim no es una lámpara: es emisión por Fresnel, así
  que dibuja borde en *todas* las siluetas visibles y no en un lado.
- **Glare por encima de la imagen terminada, con el umbral entre el
  grafito y el metal.** Sólo florece el metal. Es justo el aspecto de
  la micrografía, y no cuesta ni una muestra porque corre sobre el PNG.
- La separación metal/carbono la llevan **el color por elemento y el
  modelo `auto`** (sólido el metal, abierto el carbono), no la luz.

El estilo entra en el catálogo como uno más, así que llega solo a la
ventana, al CLI y a `render_cnt --style core_shell_cover`.

### Lo que el barrido contestó

Siete presetes sobre el mismo brazo, la misma cámara y las mismas
banderas: lo único que cambia es el estilo.

| preset | fondo | qué pasa |
|---|---|---|
| `core_shell_cover` | casi negro | el rim separa cada silueta; el glare enciende sólo el metal |
| `electron_micrograph` | casi negro | el más parecido al SEM: la coraza se lee como tejido |
| `nature_nano` | negro, sin suelo | limpio, una luz dura, nada compite con el sujeto |
| `clusters_on_sheet` | azul con suelo | el reflejo del suelo añade una segunda copia del sujeto |
| `advanced_materials` | azul medio | **el peor**: coraza y fondo con el mismo valor, el borde se pierde |
| `science_cover` | pálido, alto clave | sobre claro el carbono gris desaparece y quedan las partículas solas |
| `acs_nano_vivid` | magenta | el color del fondo se come el del metal |

La conclusión no es de gusto: **esta muestra necesita fondo oscuro**,
porque lo que hay que ver es una coraza gris oscuro, y sobre un fondo
claro o de valor medio esa coraza es justo lo que se pierde. Entre los
oscuros gana el que más borde ponga en cada silueta, que es para lo que
está escrito `core_shell_cover`.

### Y el render enseñó un defecto que ningún número había dicho

En el primer render las partículas de la decoración salieron
**flotando**. El empuje era media extensión máxima de la partícula más
el contacto grafítico, y el programa reportaba un mínimo de 3.37 Å que
era el de la mejor asentada y no decía nada de las demás. Medido
después por partícula: mediana 3.47 Å y **máximo 5.65**.

Dos reglas más baratas se probaron y las dos dejan partículas en el
aire:

- **El átomo más bajo a lo largo de la normal, más el hueco.** Mejor, y
  todavía mal por la parte lateral: ese átomo queda a `hueco` medido a
  lo largo de la normal, así que su distancia real es
  `hypot(hueco, lateral)`.
- **Contra el átomo de asiento y no contra la pared.** En el interior
  del codo la pared sube a buscar a la partícula, y un empuje calculado
  desde un solo átomo no se entera.

Lo que se entrega resuelve el empuje por **bisección contra el árbol de
la pared**: el menor empuje con el que ningún átomo de la partícula
baja de `--fe-contacto`. Medido: mínimo y mediana **2.80 Å**, que es el
contacto pedido, en las 34.

**Y el contacto pedido es 2.8 Å, por debajo de los 3.4 de van der
Waals.** Eso es una decisión de dibujo y no de química, así que es un
parámetro con su valor dicho y no una constante callada por debajo de
la física. La razón es medible: con el metal a radio de van der Waals y
el carbono en bolas y palos finos, una partícula apoyada en el contacto
real tiene el centro a unos 6 Å sobre una pared cuya superficie dibujada
es una malla delgada; sobre un tubo de 9.8 Å de radio, una partícula en
el limbo proyecta a 15.8 Å contra los 9.8 del tubo —un 60 % más
afuera— y se lee como despegada aunque esté tocando.

**Cada partícula va girada al azar.** Un cristal cortado de una red
tiene caras planas y orientación, y treinta y cuatro copias de la misma
orientación se leen como un sello repetido y no como partículas
sueltas. El giro es una rotación uniforme de SO(3) por el método del
cuaternión de Shoemake, **no tres ángulos de Euler al azar**, que se
agolpan hacia los polos y dejarían a todo el campo compartiendo un eje
preferido —que es justo el artefacto que el giro existe para quitar—.
Verificado: determinante exactamente +1, ortogonalidad a 1.3e-15, y el
eje de rotación repartido uniformemente sobre la esfera (8 cajas en z,
447 a 529 contra las 500 que tocan).

## Las escenas como .blend, para ajustarlas a mano

```bash
BPY=<python con bpy> bash modelos_SEM/escenas_blend.sh
BPY=... bash modelos_SEM/escenas_blend.sh escena_barras
```

Deja los cinco `.blend` en `modelos_SEM/blend/`. Llevan dentro la
geometría, los materiales, las luces y la cámara —la misma que la
vista previa reporta: misma lente, mismo relleno, mismo nombre de
vista—, y **nada renderizado**: `--no-render` construye la escena y
para.

Eso ahorra el muestreo entero y **no ahorra lo caro**, que es construir
la malla: una icoesfera y un cilindro a la vez desde Python. Por orden
de coste, que es el orden en que el script los escribe:

| escena | átomos |
|---|---|
| `escena_profundidad` | 10 353 |
| `escena_anemona` | 24 440 |
| `escena_anemona_decorada` | 28 520 |
| `escena_barras` | 37 152 |
| `escena_anemonas` | 54 990 |

Dentro de Blender, mover la cámara no toca nada más: la escena no
depende de dónde esté. El estilo sí está horneado en los materiales,
así que cambiar de preset es volver a correr el script con `ESTILO=`.

**Y aquí sólo cabe la primera.** Medido en este contenedor,
`escena_profundidad` tardó **88 minutos de CPU** en construir su
malla, que son **8.5 min por cada 1000 átomos**. De ahí:

| escena | átomos | CPU estimada |
|---|---|---|
| `escena_profundidad` | 10 353 | **88 min (medido)** |
| `escena_anemona` | 24 440 | ~3.5 h |
| `escena_anemona_decorada` | 28 520 | ~4 h |
| `escena_barras` | 37 152 | ~5 h |
| `escena_anemonas` | 54 990 | ~8 h |

El ejecutor en segundo plano corta a las 2 horas, así que las cuatro
grandes **no se pueden construir aquí**: la cola murió a mitad de la
anémona. Se construyen en una máquina con Blender instalado, que es
para lo que el script toma `BPY`:

```bash
cd packages/nanocarbon_lab
BPY="blender -b -P" bash modelos_SEM/escenas_blend.sh                  # las cinco
BPY="blender -b -P" bash modelos_SEM/escenas_blend.sh escena_anemonas  # una
```

Los `.xyz` y sus sidecars están versionados, así que no hace falta
nada más del contenedor, y **no hay que instalar este paquete**:
`render_cnt.py` corre dentro del Python de Blender y sólo importa sus
vecinos de `nanocarbon_lab/blender/`, más numpy, que Blender ya trae.

**Y ya no hace falta scipy.** Se usaba en un único sitio —adivinar
enlaces cuando el sidecar no los trae— y hacía falta porque una
partícula recubierta salía con **cero** enlaces: cada coraza registra
los suyos y se perdían al pegar núcleo y capas en una sola estructura.
Arreglado (`coat.stacked_bonds`), las cinco escenas los traen:

| escena | enlaces antes | ahora |
|---|---|---|
| `escena_barras` | **0** | 43 632 |
| `escena_profundidad` | 4 500 (sólo el tubo) | 12 252 |

Y por modelo suelto: `placa_H_C` 5025, `barra_T_C` 2727,
`particula_Fe3C_C` 3390. Antes, los tres a cero.

## El render de portada

`modelos_SEM/renderizar.sh` produce las imágenes de `render/`, y lleva
el estilo en una sola variable: `ESTILO=core_shell_cover bash
modelos_SEM/renderizar.sh`. Tres banderas hacen el aspecto que pediste:

```
--model auto            metales en relleno de van der Waals,
                        carbono en bolas y palos
--spacefill-element Se  ...y el calcógeno también
--atom-scale 0.55 --bond-scale 0.5    bolas y palos finos
```

**`--spacefill-element` es nuevo en el paquete y hacía falta.** El modelo
`auto` pregunta por el elemento y su tabla son **metales**, porque un
metal no tiene enlaces direccionales que dibujar. Un núcleo de FeSe
también es un sólido y el selenio no es un metal, así que salía abierto y
con palos hacia el hierro. Ahora se puede declarar por render.

**No se declara en las versiones dopadas**, y esa asimetría es el punto:
ahí el selenio está en dos sitios a la vez —en el núcleo de FeSe y
sustituido dentro del carbono— y **ninguna regla por elemento puede
distinguirlos**. En `placa_H_NSeO` se queda abierto, que es lo correcto
para el selenio del grafito y lo incorrecto para el del núcleo. Si hace
falta la portada con el núcleo sólido *y* el dopaje visible, el camino es
colorear por estructura y no por elemento.

El grafito sale más liso que en la primera versión porque la coraza se
malla y se relaja mucho más fino: `grid_resolution` 80 → **128**,
`remesh_iterations` 25 → **60**, `relax_iterations` 3000 → **8000**, y
`roughness` explícitamente a 0. Cuesta tiempo, no fidelidad.

**Blender no está en este contenedor como aplicación**; el render corre
con la rueda `bpy` en un entorno aparte, porque `bpy` fija numpy 1.26 y
el workspace necesita numpy 2 —el `pyproject.toml` raíz declara ese
conflicto justo por esto—. El script lo explica arriba del todo y vale
igual con un Blender instalado.

## Lo que falta y es fácil de añadir cuando decidas

- **Tubos de pared múltiple**, con `builders/assemblies.py`, si lo
  recubierto de grafito es el tubo entero y no sólo la punta.
- **N piridínico y pirrólico de verdad.** El dopaje sustitucional de aquí
  da el N **grafítico**; los otros dos motivos son N en el borde de una
  vacante y hay que pasar por `defects/`. Si tu XPS dice que el
  piridínico domina, este modelo todavía no lo representa.
- **La partícula fuera de la punta**, que era la v1 y sigue siendo la
  otra lectura de tu micrografía: cambiar el signo del desplazamiento en
  `arm_with_particle_inside`.
