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

## La anémona ligera: una sola pieza de carbono

```bash
uv run python modelos_SEM/anemona.py
uv run python modelos_SEM/anemona.py --brazos 6 --semilla 3 --fe_particulas 20
```

`escena_anemona_decorada` (arriba) son ocho tubos tapados colocados
alrededor de un punto: **28 520 átomos**, brazos que no se tocan, y el
carburo escondido en una punta no más ancha que el tubo.
`escena_anemona_ligera` se construye al revés y pesa **5810 átomos**
(C5338Fe472):

| pieza | qué es | átomos |
|---|---|---|
| pared de carbono | UNA superficie cerrada: núcleo, 7 brazos de 36–62 Å, 5 ramas, un bulbo en cada punta | 5256 C |
| Fe₃C encapsulado | una partícula DENTRO de cada bulbo, tallada a su cavidad | 28–41 por bulbo |
| Fe decorativo | 33 perlas de α-Fe de 9 átomos sobre los brazos | 297 Fe |

La pared sale de `builders.anemone.build_anemone`, que es nuevo: el
esqueleto es un **árbol**, y un árbol engrosado es una esfera tenga los
brazos que tenga (`V − E = 1`), así que el presupuesto de anillos es
+12 exacto y el constructor lo comprueba — censo
`{5: 145, 6: 2356, 7: 125, 8: 4}`, enlaces 1.322–1.535 Å, cero
contactos. Es la ruta implícita, o sea pared sana y amorfa (pares 5-7
donde la curvatura los pide), no una red enrollada.

**Qué está relajado y qué está colocado.** La pared es una estructura
relajada y comprobada. El metal está **colocado**: nada aquí calcula si
la cementita llena un bulbo ni si el hierro moja un tubo, y los 2.8 Å
de contacto son la misma convención de dibujo que `construir.py` usa
para el hierro sobre una pared. Es un modelo para mirar y para partir
de él, no una interfase que nadie haya relajado.

Cuatro cosas que costaron una vuelta cada una:

* **Los átomos se van a los bulbos si se les deja.** Con brazos de
  28–44 Å y bulbos de 10 Å, 3800 de 5400 carbonos eran bulbo y salía
  un racimo de bolas sobre muñones. Una anémona es sobre todo brazo.
* **El centro de un bulbo se MIDE, de grueso a fino.** La punta de un
  brazo largo y fino se desplaza varios Å al relajar; tomar el
  desplazamiento entre cajas envolventes puso tres de ocho centros
  sobre su propio tallo, y una captura del tamaño del bulbo dejó otro
  sin átomos alrededor. Ahora: desplazamiento entre centroides, captura
  ancha que se estrecha, y al final una esfera por mínimos cuadrados.
* **Una esfera rígida desperdicia la cavidad.** El bulbo relajado está
  abollado y la abolladura más honda fija el radio libre: cabían 16
  átomos donde hay sitio para el doble. La partícula se talla como la
  tallaría su cáscara — esfera de cementita hasta el radio MEDIO de la
  pared menos el contacto, menos todo átomo a menos del contacto, menos
  los que ese corte deja sueltos. **El precio: ya no es 3:1 exacto**
  (sale Fe/C ≈ 1.8–2.5), y la fórmula impresa es la del fichero.
* **El asiento de cada perla se COMPRUEBA.** Cerca de una rama el signo
  de la normal local puede salir hacia dentro, y una perla empujada al
  interior de un tubo de 3.3 Å no puede librar su pared: una quedó a
  1.1 Å. La que no descansa al contacto no se dibuja.

El `.extxyz` lleva una columna `grupo` (0 pared, 1 Fe₃C, 2 Fe) para
colorear u ocultar cada pieza. La pared se guarda en una caché junto a
los xyz: cambiar el contacto o el número de perlas la reutiliza (2 s)
en vez de reconstruirla (5 min).

## La anémona de nanotubos cristalinos (la buena)

```bash
uv run python modelos_SEM/anemona_tubos.py
uv run python modelos_SEM/anemona_tubos.py --brazos 24 --semilla 4 --fe_cada 300
```

`escena_anemona_ligera` (arriba) es topológicamente pulcra y **fea**:
su pared es la de la ruta mallada, grumosa y con pares 5-7 por todas
partes, nada que ver con los tubos de hexágonos limpios de un esquema
de micrografía. `escena_anemona_tubos` vuelve a los brazos cristalinos
de `construir.py` y los lleva más lejos, sin presupuesto de átomos:
**68 260 átomos** (C61691Fe6569).

| pieza | qué es | cuánto |
|---|---|---|
| brazos | 20 nanotubos tapados construidos sobre su red (`build_capped_cnt`): radios 5.9, 7.9 o 9.8 Å, 10–26 anillos, cada uno curvado distinto | 52 760 C con las ramas |
| ramas | 14 tubos finos (4.0 o 5.9 Å) que salen de la pared de un brazo, inclinados hacia la punta | — |
| núcleo | la partícula de Fe₃C en su cáscara grafítica (`xyz/particula_Fe3C_C.xyz`) de la que nacen los brazos | C2416Fe422 |
| Fe₃C en las puntas | un tapón de cementita cortado por el propio tubo, en 28 de 34 puntas | 2712 átomos |
| Fe decorativo | 344 perlas de α-Fe de dos tamaños, asentadas a 2.80 Å de la pared | 4110 Fe |

**Es una ESCENA.** Cada tubo es una estructura relajada y comprobada;
donde se encuentran están colocados a contacto de van der Waals, no
fundidos, y el metal está colocado y no relajado. Toda colocación se
comprueba contra una holgura medida —un brazo que no libra se vuelve a
tirar más lejos, una rama o una perla que no libra se omite— y los
recuentos impresos son los que entraron. Las seis puntas sin tapón son
las de los tubos de 4 Å, donde no cabe nada a 2.8 Å de la pared.

Tres cosas de método:

* **El tapón no es 3:1 exacto** (sale C675Fe2037, Fe/C ≈ 3.0 en
  conjunto pero no en cada punta), porque lo corta la cavidad.
* **La primera tirada asentó 1077 perlas** —una cada 60 Å²— y los tubos
  desaparecían debajo. La decoración de una micrografía es una
  salpicadura, no un recubrimiento: ahora va una cada 420 Å².
* **El «fuera» de cada átomo se toma de la línea central de su propio
  tubo**, no de un parche local, que cerca de un vecino da el signo al
  revés y mete la perla dentro de un tubo ancho, donde además cabe.

El par más cercano de la escena, 1.26 Å, **no es de la colocación**:
son 55 enlaces cortos dentro de la cáscara del núcleo, que ya venían en
`particula_Fe3C_C.xyz`. El `.extxyz` lleva la columna `grupo` (0 tubos,
1 núcleo, 2 Fe₃C de las puntas, 3 Fe).

## Escenas limpias y sus proyectos de Blender

| guion | escena | qué es |
|---|---|---|
| `anemona_tubos.py` | `escena_anemona_tubos` | 11 brazos con 16 ramas **fundidas** a la pared (`build_branched_cnt`: seis heptágonos por unión) |
| `anemona_tubos.py --brazos 26 --ramas_max 0 --finos 0.4 --nombre escena_anemona_fina` | `escena_anemona_fina` | 26 brazos finos sin ramas, 77 722 átomos |
| `escena_fese.py` | `escena_placas` | la placa hexagonal de FeSe **envuelta por completo** (la de `construir.py`), cinco veces en columna, girando de frente a canto |
| `escena_fese.py` | `escena_barras_tubo` | 9 barras de FeSe tetragonal talladas a la cavidad de un nanotubo tapado, sin tocarse |

En las anémonas el Fe **moja** la pared: cada partícula se hunde, la
pared la corta al contacto Fe-C (2.1 Å) y se asienta. El metal está
colocado, no relajado, y los brazos se apoyan en la partícula central a
distancia de van der Waals.

### `portada_blend.py`: el mismo estilo, en segundos

```bash
<python con bpy> modelos_SEM/portada_blend.py -- \
    --extxyz modelos_SEM/xyz/escena_placas.extxyz \
    --blend modelos_SEM/blend/escena_placas.blend [--png vista.png]
```

Aplica `core_shell_cover` con las funciones del propio `render_cnt.py`
(materiales, fondo, luces, cámara medida), pero cada átomo y cada enlace
es una **instancia**: la escena de 77 722 átomos se monta en segundos y
el `.blend` pesa 7.6 MB, donde `escenas_blend.sh` midió 88 minutos para
10 353 átomos. Añade profundidad de campo enfocada al centro y deja el
suelo fuera (`--floor` lo devuelve): visto desde arriba refleja las
luces como rectángulos brillantes detrás del sujeto. El *glare* del
estilo es un pase sobre el PNG, así que va en `--png` y no en el
`.blend`.

Los tres proyectos están en `blend/` a 1600×2000, y sus pruebas a
800×1000 y 48 muestras en `render/portada/`. La anémona usa
`--atom-scale 1.0 --bond-scale 1.3 --fill 1.2` para que los tubos se
lean a esa distancia.

### El estilo `revista` y la escena conjunta

`portada_blend.py --style revista` es un estilo propio del guion para
portada de alto impacto: fondo índigo profundo degradado a casi negro,
carbono perla con borde cian, **Fe cobre pulido y Se verde azulado a
radio de van der Waals** (Fe 2.00 Å, Batsanov; Se 1.90 Å, Bondi), luz
principal cálida, contraluz fría fuerte y un acento magenta bajo para
que las sombras tengan color. `--efecto` añade cualquier efecto de
`render_cnt` (p. ej. `dust:color=amber,count=260,strength=4`). El
`halo` volumétrico no sirve aquí: su densidad está en metros y a escala
de ångström llena el cuadro de niebla.

| proyecto | escena | opciones |
|---|---|---|
| `revista_anemona_fina` | 26 brazos finos | `--atom-scale 1.0 --bond-scale 1.3 --fill 1.2` |
| `revista_barras` | 20 barras (de 26 pedidas, las que caben sin tocarse) | bokeh cian y polvo ámbar, `--fill 1.08` |
| `revista_conjunta` | `escena_conjunta.py`: anémona compacta de 18 brazos, arco de 4 placas envueltas que giran de canto a frente, abanico de 7 barras | `--view front --fill 0.93`, polvo ámbar |

`escena_conjunta.py` coloca cada pieza, la comprueba contra todo lo ya
colocado y, si se acerca más que `holgura` (6 Å), la aparta del centro
del cuadro e imprime cuánto. La composición final no aparta ninguna.
La anémona compacta sale de
`anemona_tubos.py --brazos 18 --ramas_max 0 --finos 0.5 --anillos_min 12
--anillos_max 22 --fe_cada 520 --semilla 3 --nombre anemona_conjunta`.

### Placas con superficie pareja, brazos hundidos y suelo espejo

- **`placa_mixta.py` → `xyz/placa_H_mixta.xyz`.** En el FeSe tipo NiAs
  cada plano basal es todo Fe o todo Se, así que la placa original
  mostraba un elemento en el centro de cada cara y el otro en el canto
  (un disco verde azulado con anillo cobre). El núcleo nuevo se talla
  con la normal de la placa a lo largo de a* (planos {10-10}): cada cara
  es un plano completo con Fe y Se a la misma altura, 326 átomos al
  49 % de Fe en los 0.4 Å externos. {11-20} también promedia ~48 %,
  pero su cara sale en terrazas (77 átomos en la capa externa) y se ve
  a franjas. La coraza es la de `placa_H_C`, ya validada; recubrir este
  núcleo desde cero salió roto en 12 intentos (mejor enlace 1.60 Å). Se
  quitan los átomos del núcleo a menos de 2.2 Å de la coraza (la
  original tenía su par más cercano a 1.5 Å). En cualquier cara del
  FeSe, Fe y Se se alternan por filas, así que de cerca se ve un rayado
  fino: es el cristal, no el corte.
- **`anemona_tubos.py --hundir 4`.** La base de cada brazo entra 4 Å
  en la coraza de la partícula central, los brazos se comprueban solo
  entre sí y contra el metal (≥ 2.2 Å), y se quitan los carbonos de la
  coraza a menos de 2.6 Å de un brazo: queda una boca alrededor de
  cada tubo y los brazos parecen nacer de la partícula. Es una
  apariencia: las dos redes no están enlazadas.
- **Suelo espejo en `revista`.** Negro, metálico y pulido, con
  `--elev/--azim` para colocar la cámara. Tres cosas fueron necesarias:
  estirarlo 60 veces para que su borde no corte el cuadro como un
  horizonte; *light linking* (cada lámpara ilumina todo menos el
  suelo), porque si no cada lámpara se reflejaba como un rectángulo
  blanco bajo las piezas; y no ocultar las lámparas en los reflejos,
  que apaga el metal (su brillo ES el reflejo de las lámparas). Sin
  efectos de partículas.
- **`escena_orbita.py`.** Segunda escena conjunta: la anémona en el
  centro con un anillo inclinado de placas (mirando hacia fuera) y
  barras (a lo largo del anillo) alternadas, reflejado en el suelo.

| proyecto | cámara |
|---|---|
| `revista_orbita` | `--elev 30 --azim 15 --fill 1.0` |
| `revista_conjunta` | `--elev 14 --fill 0.93` |
| `revista_anemona_fina` | `--elev 20 --azim 25 --atom-scale 1.0 --bond-scale 1.3 --fill 1.1` |
| `revista_placas` | `--elev 12 --azim 20 --fill 0.95` |
| `revista_barras` | `--elev 32 --azim 10 --fill 1.05` |

### Campo de anémonas y vórtice

- **`escena_campo.py`** (180 344 átomos): la protagonista
  (`anemona_conjunta`, 18 brazos) bien adelantada; tres copias giradas
  de `anemona_mini` (12 brazos cortos, 14 019 átomos, construida con
  `anemona_tubos.py --brazos 12 --ramas_max 0 --finos 0.6 --anillos_min
  10 --anillos_max 16 --fe_cada 800 --semilla 7 --hundir 4 --nombre
  anemona_mini`) a los lados y detrás; y en el suelo 36 barras tumbadas
  y 5 placas planas o inclinadas, todas sobre un mismo plano, que es
  donde el render pone el espejo. Las de fondo van abiertas a los lados
  a propósito: con lente corta y de cerca, lo que queda detrás de la
  protagonista a menos de unos dos radios del eje queda tapado.
- **`escena_vortice.py`** (172 188 átomos): 40 placas y barras en una
  hélice que se abre al subir (130 → 300 Å de radio, 560 Å de alto, 2.2
  vueltas) saliendo de la anémona; las barras siguen la tangente de la
  hélice y las placas miran hacia fuera.

`portada_blend.py` gana `--lente` (mm; corta para que lo de delante
crezca frente a lo de atrás) y `--encuadrar GRUPO...`, que encuadra solo
esos grupos del `.extxyz` y amplía el recorte lejano de la cámara a toda
la escena (si no, todo lo que está detrás de la protagonista se corta).

| proyecto | cámara |
|---|---|
| `revista_campo` | `--elev 9 --lente 28 --fill 0.72 --encuadrar 0` |
| `revista_vortice` | `--elev 24 --azim 10 --fill 1.05` |

### Solo anémonas, vórtice sobre un campo, y la síntesis

- **`escena_anemonas`** = `escena_campo.py --fondo 7 --barras 0 --placas
  0 --nombre escena_anemonas` (141 541 átomos): la protagonista y siete
  anémonas pequeñas alternando `anemona_mini` y `anemona_mini_b` (14
  brazos, 18 623 átomos: `anemona_tubos.py --brazos 14 --ramas_max 0
  --finos 0.4 --anillos_min 10 --anillos_max 18 --fe_cada 700 --semilla
  11 --hundir 4 --nombre anemona_mini_b`). Primer plano con `--lente 50
  --fill 1.1 --encuadrar 0`: la protagonista llena el cuadro y las demás
  quedan detrás, desenfocadas.
- **`escena_vortice.py`** ahora pone `campo` anémonas pequeñas (7) en el
  suelo alrededor de la base (284 133 átomos).
- **`escena_sintesis.py`** (14 885 átomos): arriba acetonitrilo,
  ferroceno y selenourea enteros, construidos átomo a átomo; más abajo
  los mismos rotos en sus fragmentos (CH3 + CN, Fe + dos C5H5, Se +
  C(NH2)2), cada vez más separados; luego átomos sueltos de Fe y Se; y
  abajo los productos: a 900 °C la placa con la coraza dopada y
  funcionalizada (`placa_H_NSeO`) alrededor del núcleo de corte parejo,
  con una ventana abierta en media cara para ver el FeSe, y dos barras
  `barra_T_NSeO`; a 1020 °C dos `nanotubo_NSeO` con más grupos de
  oxígeno injertados. Los fragmentos son un dibujo de la química, no una
  simulación: qué enlace se rompe primero no se calcula. **Moléculas,
  fragmentos y átomos sueltos van ampliados ×5**, cada uno sobre su
  propio centro, y un rótulo lo dice: a escala real una molécula de 5 Å
  junto a una placa de 70 Å es una mota. Los rótulos se escriben en
  `xyz/escena_sintesis.rotulos.json` y `portada_blend.py` los convierte
  en texto 3D luminoso orientado a la cámara; van sobre el suelo, porque
  bajo él el espejo les cortaba el pie (y "1020" se leía "1030").

`portada_blend.py` ahora dibuja enlaces entre todos los no metales en
tres clases de longitud (X-H, X-X, X-Se), con colores para N, O y H en
`revista`. Un átomo enlazado se dibuja como bola y uno sin enlaces
(núcleo cristalino) como esfera de van der Waals, así el Se dopante de
un tubo es una bola y el del FeSe una superficie. Opciones nuevas:
`--bola GRUPO...`, `--ampliar F` (los enlaces de los grupos `--bola` se
buscan a escala real y bolas y varillas se dibujan F veces más gruesas),
`--dopantes F` (agranda N, O, H y Se de las paredes) y los rótulos.

| proyecto | cámara |
|---|---|
| `revista_anemonas` | `--elev 12 --lente 50 --fill 1.1 --encuadrar 0` |
| `revista_vortice` | `--elev 16 --azim 10 --fill 1.22` |
| `revista_sintesis` | `--elev 6 --fill 0.92 --bola 4 5 6 --ampliar 5 --dopantes 2.2` |

### Primeros planos, paletas y temperatura sin texto

- **Anémonas juntas.** `escena_campo.py --compacto 0.4` acerca el fondo
  y la protagonista; con 8 anémonas de fondo la escena cabe en 745 × 962
  Å (antes 1558 × 1917). Si una no cabe en su sitio, se desplaza más
  cuanto más intentos lleva, en vez de descartarse. Con teleobjetivo
  (`--lente 105`) la profundidad se comprime: la red hexagonal de la
  protagonista se ve de cerca y las demás quedan agrupadas y
  desenfocadas detrás.
- **Vórtice.** El campo se acerca al eje (`campo_cerca` 250 Å,
  `campo_lejos` 420 Å, antes 380–640).
- **Síntesis simétrica.** Productos en espejo respecto al centro: placa
  de frente con ventana y dos barras en X detrás (izquierda), un tubo de
  frente y dos en X detrás (derecha). Las moléculas se generan en la
  mitad izquierda y se reflejan en la derecha. **Sin rótulos**: la
  temperatura la dice la luz. `portada_blend.py --calor` pone un fondo
  que pasa en horizontal de ámbar a casi negro y a azul frío, y un foco
  sobre los productos de cada mitad (ámbar a la izquierda, blanco
  azulado a la derecha). Dos luces anchas desde los lados se probaron
  antes: iluminaban todo con los dos colores y sumaban blanco.
- **Paletas** (`--paleta`, solo con `--style revista`). Todas ponen el
  color o la luz en el ENLACE, para que se lean los hexágonos y no las
  bolas:
  - `revista`: carbono perla, Fe cobre, Se verde azulado.
  - `neon`: átomos de vidrio negro y enlaces cian que emiten luz (a
    emisión 5 el tubo se quemaba en blanco; va a 1.2), Fe oro, Se
    magenta.
  - `oro`: laca negra y enlaces de oro, Fe oro rosa, Se esmeralda.
  - `iridiscente`: plata bajo una película delgada (420–520 nm), cuyo
    color cambia con el ángulo; Fe naranja, Se violeta.

| proyecto | cámara |
|---|---|
| `revista_anemonas_{revista,neon,oro,iridiscente}` | `--paleta … --elev 10 --lente 105 --fill 1.2 --encuadrar 0 --bond-scale 1.6` |
| `revista_vortice` | `--elev 14 --azim 10 --lente 85 --fill 1.3 --bond-scale 1.4` |
| `revista_sintesis` | `--calor --elev 10 --lente 85 --fill 0.98 --bola 4 5 6 --ampliar 5 --dopantes 2.0 --bond-scale 1.4` |
| `revista_sintesis_cerca` | igual, con `--elev 14 --fill 0.86 --encuadrar 0 1 2 3` |

### Primer plano de la superficie de FeSe

`escena_superficie.py` (3113 átomos) es el primer plano de la síntesis,
**sin coraza de carbono**: el núcleo de corte parejo de
`placa_H_mixta` tumbado con su cara {10-10} hacia arriba, centrado, y
dos barras desnudas (`barra_T_nucleo_desnudo`) en V detrás, una espejo
de la otra. Encima, a **escala real** (de cerca no hace falta ampliar),
precursores enteros, fragmentos y adátomos de Fe, Se, C y N posados a
2.4 Å de la superficie, todos al azar: la simetría es de la
composición, no de las moléculas. En `escena_sintesis.py` los
precursores vuelven a caer al azar por todo el ancho; solo los
productos están en espejo.

| proyecto | cámara |
|---|---|
| `revista_superficie` | `--calor --elev 26 --lente 85 --fill 0.92 --encuadrar 0 --bola 4 5 6 --bond-scale 1.3` |
| `revista_superficie_v` | igual, con `--elev 30 --fill 0.97 --encuadrar 0 2` (placa y barras) |

### Primer plano de una barra y de un nanotubo con partícula

- **`escena_barra.py`** (3973 átomos): una barra de FeSe tetragonal
  tallada nueva como cápsula (radio 11 Å, 110 Å de largo, Fe1250Se1250,
  *clean*), más gruesa que `barra_T_nucleo_desnudo` para que su
  superficie llene el cuadro, en diagonal ascendente; detrás, de frente
  y a 72 Å, la placa de corte parejo. Sin precursores.
- **`escena_tubo_particula.py`** (4873 átomos): nanotubo tapado de
  frecuencia 5 (9.8 Å de radio, `{5: 12, 6: 1740}`), con grupos
  hidroxilo, carboxilo y carbonilo por fuera (663 O) y dopado con N y Se
  (220 N, 91 Se), y una partícula de cementita tallada por la cavidad
  (C48Fe124, a 3.09 Å de la pared como mínimo). **Los grupos van antes
  que el dopaje**: injertados después de `codope`, los tres pasos solo
  encontraron sitio para 17 O, frente a 210 hidroxilos solos en el tubo
  sin dopar.

`portada_blend.py`: `--invertir-calor` pone el frío a la izquierda y el
calor a la derecha; `--desenfoque F` multiplica la profundidad de campo;
y con `--encuadrar` el foco va al centro de lo encuadrado (antes iba al
centro de toda la escena, entre la barra y la placa, y dejaba la placa
nítida).

| proyecto | cámara |
|---|---|
| `revista_barra` | `--calor --invertir-calor --elev 10 --lente 85 --fill 0.88 --encuadrar 2 --desenfoque 20` |
| `revista_tubo_particula_{revista,neon}` | `--paleta … --elev 14 --lente 85 --fill 0.3 --encuadrar 1 --dopantes 1.5 --bond-scale 1.4 --desenfoque 2` |

### Punta del nanotubo y anémonas de tubos cortos y anchos

- **`escena_tubo_particula.py`** ahora pone la partícula en la punta
  (`punta 1`): una esfera de cementita de 12 Å centrada en la tapa y
  tallada por la cavidad, que la llena y baja un poco por el tubo
  (C43Fe123). El tubo va vertical con la punta arriba. Se al 3 % (el
  máximo que el módulo de dopantes considera realista; al 4 % avisa) y
  resaltado en el render con `--se-grande 1.6`, que agranda solo las
  bolas de Se enlazadas.
- **Anémonas cortas y anchas.** `anemona_tubos.py` acepta
  `--freq_fino/--freq_grueso` (radio 1.96 × freq Å). Con 4 y 5 (7.8 y
  9.8 Å) y pocos anillos: `anemona_corta` (12 brazos, 19 750 átomos,
  Fe3C en las 12 puntas), `anemona_mini_corta` (8), `anemona_mini_corta_b`
  (9) y `escena_anemona_corta` (16). `escena_campo.py` acepta `--heroe` y
  `--fondos`, y `escena_conjunta.py`/`escena_orbita.py` aceptan
  `--anemona` y `--nombre`, así que las escenas `_corta` salen de los
  mismos guiones sin tocar las anteriores. El vórtice no cambia.

| proyecto | escena |
|---|---|
| `revista_tubo_particula_{revista,neon}` | `--elev 8 --lente 85 --fill 0.42 --encuadrar 1 --dopantes 1.4 --se-grande 1.6 --bond-scale 1.4 --desenfoque 2` |
| `revista_anemona_corta` | `escena_anemona_corta`, `--elev 20 --azim 25 --bond-scale 1.3 --fill 1.05` |
| `revista_anemonas_corta_{revista,neon,oro,iridiscente}` | `escena_campo.py --heroe anemona_corta --fondos anemona_mini_corta,anemona_mini_corta_b --fondo 8 --barras 0 --placas 0 --compacto 0.4` |
| `revista_campo_corta` | igual con `--fondo 3` y barras y placas |
| `revista_conjunta_corta`, `revista_orbita_corta` | `--anemona anemona_corta` |

### Brazos tallados en la partícula y Se proporcional

- **`anemona_tubos.py --tallar 1`.** Con brazos anchos (7.8 y 9.8 Å)
  sobre una partícula de 18.5 Å, `--hundir` solo no bastaba: los brazos
  chocaban entre sí junto a la superficie, la comprobación los alejaba y
  quedaban separados de la partícula. Con `--tallar` los brazos se
  comprueban entre sí solo fuera de la partícula (más allá de su radio
  − 1 Å), entran 14 Å (`--hundir 14`), se quitan los átomos de los
  brazos que quedan ocultos dentro, y la partícula pierde su coraza y su
  metal donde entra cada tubo. Las anémonas `_corta` se reconstruyeron
  así; ningún par baja del 1.26 Å que ya traía la coraza de la partícula.
- **Se en el tubo.** Todo el Se está enlazado (1.42 Å) y se dibuja como
  bola, no a radio de van der Waals; se veía tan grande por el factor
  `--se-grande 1.6` sobre `--dopantes 1.4` (1.48 Å de radio). Se renderiza
  sin `--se-grande`, proporcional a N, O y H. Se quitó la versión neón.
