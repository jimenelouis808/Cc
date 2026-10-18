# Novedades

Lo que ha cambiado en esta ronda, escrito para quien va a usarlo y no para
quien lo ha escrito. Cada apartado dice qué hace ahora, qué NO hace, y dónde
está la trampa.

---

## nanocarbon_lab · esferas sin facetas, presets de superestructura, y la ventana que se explica

### Las bolas facetadas: era 80 triángulos, no un ajuste de gusto

`sphere_subdiv` estaba fijo en 2. Una icoesfera a subdivisión 2 son **80
triángulos** — no 320, que es la 3; la cuenta es `20·4ⁿ` — y eso se ve facetado
en cuanto un átomo ocupa más de un pequeño porcentaje del cuadro, que es
siempre el caso en un primer plano.

No se puede simplemente subir: el coste es lineal en el número de átomos, y
2000 átomos a subdivisión 5 son diez millones de triángulos antes de dibujar un
solo enlace. Ahora hay cuatro niveles y **se elige por el número de átomos**:

| nivel | caras por átomo | lados del enlace |
|---|---|---|
| `low` | 80 | 8 |
| `normal` | 320 | 12 |
| `high` | 1280 | 20 |
| `ultra` | 5120 | 32 |

Una jaula de 60 átomos recibe `ultra`, una red de 1200 `normal`, una de 7500
`low`. `--detail` manda si quieres fijarlo.

### Cuatro presets para superestructuras (900–2000 átomos)

A ese tamaño el sujeto es una **red**, no un objeto, y tres valores tienen que
cambiar a la vez — por eso son presets y no una nota diciéndote que muevas tres
controles:

- **Los átomos tienen que encoger.** Todo lo que engorda un átomo cierra los
  huecos por los que el ojo lee profundidad, y la imagen se vuelve un muro de
  esferas.
- **La profundidad de campo tiene que abrirse.** Una red fotografiada a f/2 es
  un borrón con una franja nítida: se lee como un error, no como una decisión.
- **El color no puede venir del elemento.** Una red de puro carbono coloreada
  por elemento es una masa gris; lo que varía es la curvatura.

| estilo | qué es |
|---|---|
| `lattice_dark` | iluminada por detrás contra negro, los puentes se separan por su borde |
| `lattice_depth` | neblina en el aire: el fondo se desvanece — la perspectiva atmosférica es la única pista de profundidad que sobrevive a la repetición |
| `lattice_blueprint` | **sin esferas** (licorice) sobre rejilla luminosa: para una figura sobre conectividad |
| `lattice_studio` | luz plana sobre blanco, sin acento ni bloom |

### Tres FX más, y todos opcionales

`dust` (motas en el aire — lo más común en un render de estudio para que el
aire parezca aire y el objeto deje de leerse como un recorte), `grid` (rejilla
luminosa en el suelo) y `wave` (un anillo alrededor del sujeto: un frente de
onda, una intercara). Con los de antes son **diez**: laser, arc, resonance,
orbit, halo, bokeh, dust, grid, wave, fog, glare.

Todos siguen siendo opcionales: cada uno tiene su casilla en la ventana,
`--no-style-effects` quita los que trae un estilo, y ninguno ilumina la escena
por defecto — un objeto emisivo es una lámpara en Cycles, y un campo de motas
lo bastante brillante como para iluminar algo es un campo que nadie se creería.

### La ventana dice qué hace cada preset

Un preset es un paquete de una docena de decisiones y su nombre no dice
ninguna. Ahora, debajo del menú de estilo, la ventana imprime las que cambian
lo que ves:

```
backdrop   gradient
colour by  ring -- tamaño de anillo: dónde están la curvatura y los defectos...
model      licorice -- sólo varillas, para una red densa cuyos átomos taparían
           la topología
camera     42 mm, sin profundidad de campo
render     Cycles a 320 muestras
scale      atoms x1, bonds x0.75
effects    grid, glare  (las casillas pueden quitarlos)
```

Y debajo, **qué has sobrescrito tú**. Cada opción de modelo, de modo de color y
de fondo tiene su línea de ayuda; hay un test que falla si alguna se queda sin
ella, porque un control cuyas opciones no se explican es un control que nadie
usa.

### Y la elección de dónde renderizar, explícita

Dos botones lado a lado, porque son una elección y no una secuencia:

- **Render here →** aplica todo lo de arriba y escribe la imagen, sin abrir
  Blender.
- **Open in Blender →** monta la misma escena —luces, cámara, materiales,
  fondo, efectos—, guarda el `.blend` al lado del bundle y te entrega la
  ventana viva para que ajustes y renderices tú.

---

## nanocarbon_lab · ball-and-stick o space-filling (o los dos a la vez)

### El control que pedías, y el caso mixto es el importante

```bash
nanocarbon render jaula.xyz --model ballstick   # una red
nanocarbon render np.xyz    --model spacefill   # una partícula metálica
nanocarbon render np.xyz    --model auto        # los dos, en una imagen
```

| modelo | qué dibuja |
|---|---|
| `ballstick` | esferas a radios de dibujo unidas por varillas — lo correcto para una red, donde la conectividad **es** el tema |
| `spacefill` | esferas a su radio de **van der Waals** real, sin varillas — la forma que la estructura ocupa de verdad |
| `auto` | **space-filling para los metales, ball-and-stick para el resto** |
| `licorice` / `wireframe` | sólo varillas, gruesas o finas |

**`auto` es el que importa**, y es justo tu segunda imagen de referencia: una
partícula de oro recubierta de carbono sale como **núcleo metálico sólido
dentro de una jaula de carbono abierta**, que es lo que es. Ni el modelo puro
de bolas ni el puro de esferas acierta en más de la mitad de esa estructura.

Tres reglas detrás:

- **Los radios de vdW son los reales** (Bondi; Alvarez 2013 para los metales),
  no radios de dibujo escalados a gusto. Un modelo space-filling se **define**
  por átomos que se tocan en su contacto de vdW, así que ajustarlos a ojo
  dejaría de serlo.
- **`auto` no dibuja ninguna varilla hacia un átomo space-filling.** Empezaría
  dentro de una esfera de su propio radio y saldría por el otro lado — y entre
  dos metales es una invención: los átomos de un cúmulo de oro están
  empaquetados, no enlazados por pares.
- **Sólo los metales** van a space-filling. El carbono, los dopantes y los
  grupos siguen en ball-and-stick, porque la red es el tema.

Un estilo puede declarar su modelo preferido (`clusters_on_sheet` trae `auto`),
y `--model` manda por encima.

### Cinco presets sacados de tus referencias

Cada uno copia **una** clase de imagen concreta, y su descripción dice cuál —
un preset cuya procedencia es «quedaba bonito» no se puede corregir después.

| estilo | qué copia |
|---|---|
| `model_kit` | un kit físico de bolas y varillas sobre una mesa: esferas moldeadas brillantes, **varillas blancas mates**, fondo muy desenfocado |
| `clusters_on_sheet` | cúmulos metálicos sobre una red: oro cálido empaquetado contra lámina azul fría, suelo brillante. Trae `--model auto` |
| `clear_glass` | cristal azul-gris translúcido sobre fondo claro cálido, la red perdiéndose en desenfoque |
| `neon_violet` | azul-violeta profundo, borde magenta, suelo reflectante y **bokeh real** detrás |
| `teal_lab` | esmalte rojo y cian sobre turquesa plano — el registro de gráfico de portada de índice |

Lo que añaden además del color:

- **Un suelo que refleja** (`floor_metallic`). Una foto de producto apoya el
  sujeto en un espejo; un recolector de sombras no es lo mismo.
- **Bokeh como geometría, no como textura.** Son esferas emisivas de verdad
  detrás del plano de enfoque, así que las desenfoca la apertura de la propia
  cámara. Unos discos pintados tendrían el mismo tamaño y forma mirara donde
  mirara la cámara; éstos crecen, se mueven y se solapan al cambiar el objetivo
  o el diafragma. El estilo que los trae trae también una apertura abierta,
  porque bokeh con todo enfocado es un campo de puntos.
- **Varillas mates bajo bolas brillantes** (`model_kit`). Las bolas de un kit
  son plástico inyectado brillante y las varillas plástico blanco mate, y
  equivocarse en eso es la mayor parte de por qué un render se lee como
  diagrama y no como fotografía de un objeto. Éste es el caso para el que
  existe el material de enlace separado.
- **`atom_scale` / `bond_scale` por estilo.** Las bolas de un kit están
  deliberadamente fuera de escala; eso es lo que lo hace un kit. Con un tope:
  una bola de kit real mide alrededor de medio enlace de ancho, y pasado eso
  las esferas se fusionan en un bulto sin varillas visibles — un modelo
  space-filling por accidente. Lo medí subiendo demasiado y bajándolo.

Dos cosas honestas: `model_kit` está pensado para un **sujeto pequeño** —un
kit modela una molécula o un fragmento, y una jaula de 240 átomos a esos
radios es una bola densa de esferas haga lo que haga la luz—, y `clear_glass`
lleva 640 muestras porque el cristal es lo más lento de aquí: cada esfera
refracta dos veces y lo que hay **detrás** es la imagen.

---

## nanocarbon_lab · materiales separados, efectos que son luz, y presets de revista

### Primero: sirve para todo, no sólo carbono

El renderizador lee un **bundle**, no un tipo de estructura. Nanopartículas,
composiciones, dicalcogenuros — todo pasa por el mismo camino. Lo comprobé
renderizando un núcleo de Au₄₉₀ dentro de una corteza grafítica dopada con N,
y una composición de tres piezas:

```bash
nanocarbon render np.xyz   --style cosmos  --colour-by element
nanocarbon render trio.xyz --style plasmon --colour-by structure
```

Y una nanopartícula es justo donde `--colour-by element` se gana el sueldo:
oro, carbono y nitrógeno se leen de un vistazo, donde en una jaula de carbono
puro ese modo da un solo color.

### Átomos y enlaces ya son materiales separados

No lo eran: cuando el color pertenecía al átomo, los enlaces **reutilizaban la
lista de materiales de los átomos**, así que pedir átomos de cristal daba
enlaces de cristal y no había forma de decir otra cosa.

```bash
nanocarbon render jaula.xyz --colour-by element \
  --finish glass --bond-finish metal --bond-colour neutral
```

| `--bond-colour` | qué hacen los enlaces |
|---|---|
| `auto` | partidos donde el color es del átomo, el del estilo donde no — el comportamiento de antes, y el de por defecto |
| `atoms` | cada mitad toma el color de su átomo |
| `style` | el material de enlace del estilo |
| `neutral` | un gris, para que el color lo lleven sólo los átomos |

### Los efectos ahora son luz, no líneas

Tenías razón en que se veían básicos. Tres reescrituras, y las tres eran el
mismo error de fondo: dibujar una forma donde debería haber luz.

- **Un trazo de ancho constante parece alambre.** Ahora cada filamento lleva
  perfil: fino en los dos extremos para un arco, fino en uno para una chispa,
  con cintura para un haz.
- **Un solo tubo emisivo es un alambre que brilla, no un brillo.** Ahora cada
  trazo es un **núcleo caliente fino dentro de una vaina ancha casi
  transparente**, porque un solo material no puede ser opaco en el centro y
  tenue en el borde. El núcleo **conserva el tono**: subirlo casi a blanco,
  que fue la primera versión, convertía cada filamento de color en un alambre
  blanco.
- **Un paseo aleatorio no es un rayo.** Ahora es **desplazamiento fractal del
  punto medio** — el método con el que se genera un relámpago: curvas grandes
  con quiebros pequeños encima y quiebros más pequeños sobre ésos. El
  desplazamiento de un paseo es la *suma* de sus pasos, así que deriva suave
  y el resultado es un meandro: exactamente lo que parecían los arcos.

El láser además **se detiene en la superficie**, suelta **chispas** y deja un
destello en el impacto.

**Y el halo tiene que ser un volumen.** Un casquete de superficie conserva la
silueta de la esfera por mucho que lo afines: emisión uniforme da un disco
esmerilado, y Fresnel en el limbo da un anillo duro — un círculo dibujado
alrededor. Construí los dos y los dos se descartaron al verlos. Un volumen no
tiene silueta. Con una trampa que costó un fotograma en blanco con la
estructura de fantasma dentro: **la emisión volumétrica se acumula a lo largo
del rayo**, así que un casquete el doble de ancho brilla el doble. Ahora se
divide por la longitud del camino.

### Luz de borde por Fresnel

`rim` añade emisión **sólo donde la superficie se gira respecto a la cámara**.
Es lo único de una portada que una lámpara no puede hacer: una lámpara de
contorno dibuja un borde brillante de *un* lado, y esto lo dibuja en **todas
las siluetas que el espectador ve**, incluida la pared lejana de una jaula a
través de la cercana. Es lo que hace que una portada oscura se lea en
miniatura. Encendido en los estilos oscuros, apagado en los de figura.

### Cinco presets de revista más (veintidós en total)

Cada uno es una **respuesta distinta** a qué premia la dirección de arte de esa
revista, no cinco maneras de decir «oscuro y brillante»:

| estilo | qué es |
|---|---|
| `science_cover` | clave alta, sujeto único, poca profundidad de campo sobre fondo claro degradado — se lee en miniatura y bajo una cabecera |
| `nature_nano` | casi negro, una luz dura desde atrás, cuerpo tan oscuro que **el borde es el dibujo** |
| `advanced_materials` | saturado, brillante, lleno de color; colorea por elemento, porque esas portadas suelen ir de una composición |
| `acs_nano_studio` | la foto de producto: metal cepillado sobre ciclorama, sin color de acento, manda la geometría |
| `matter_section` | figura de página interior: luz plana, fondo blanco, enlaces neutros, sin bloom ni desenfoque — para cuando un revisor cuenta átomos |

### Resolución y muestras, una sola decisión

```bash
nanocarbon render jaula.xyz --quality print   # 2400 px, 512 muestras
nanocarbon render jaula.xyz --quality cover   # 4096 px, 1024 muestras
```

`print` son 300 dpi a lo ancho de una columna de revista con margen para
recortar; `cover`, 300 dpi a lo ancho de una página. Escala el lado **largo**
y conserva la proporción de `--resolution`, así que una portada pedida a
2000×2400 sigue siendo 5:6. Van juntas porque un fotograma de 6000 px a 64
muestras es una imagen grande con ruido, y uno de 600 px a 1024 es una
miniatura lenta.

---

## nanocarbon_lab · acabados, motor a elegir, y el .blend editable

### Dos respuestas primero

**Sí, los renders que te enseñé eran modo `ring`** — los cinco estilos de
portada traen `default_colour_by="ring"`. Y hay una razón por la que no
usé elemento para enseñarlos: un C240 o un nanotubo **coloreado por
elemento es de un solo color**, correctamente, porque todos sus átomos
son carbono. El modo elemento luce en una estructura dopada,
funcionalizada o encapsulada. El ejemplo de abajo es una jaula al 8 % N,
8 % B con hidroxilos: cinco elementos, y se lee de un vistazo.

**No, no era EEVEE: era Cycles**, y siempre lo ha sido — los diecisiete
estilos declaran `engine="CYCLES"`. Lo que faltaba era poder *elegir*, y
eso ya está.

### Acabados: un cuarto eje

`--style` es el carácter de la superficie, `--colour-by` es lo que
significa el color, `--background` es el fondo, y ahora `--finish`
**sustituye el carácter conservando el color**:

```bash
nanocarbon render jaula.xyz --colour-by element --finish glass
nanocarbon render jaula.xyz --style cosmos --finish iridescent
nanocarbon render --list-finishes
```

Dieciséis: `glass`, `crystal`, `glow`, `neon`, `metal`, `gold`,
`brushed`, `chrome`, `plastic`, `ceramic`, `clay`, `matte`, `wax`,
`iridescent`, `toon`, y `style` (dejar el del estilo, que es lo que hace
por defecto).

Lo importante: colorear por elemento y pedir cristal da **un carbono de
cristal y un oxígeno de cristal, cada uno con su color**. Ése es el punto
de mantener los ejes separados.

Un número ahí no es gusto sino medida: el cristal **aclara un color
oscuro** hacia el blanco. La luz transmitida se multiplica por el color
base en cada rebote, así que cristal sobre el gris del carbono,
atravesando la pared cercana y la lejana de una jaula, sale casi negro
— hollín, no cristal.

### Cycles o EEVEE, a elegir

```bash
nanocarbon render jaula.xyz --engine eevee    # segundos, para comprobar
nanocarbon render jaula.xyz                   # Cycles, para el final
```

Tres cosas que costó averiguar y que conviene no volver a descubrir:

- **Cycles es un add-on**, así que en un `bpy` de PyPI no aparece en la
  lista de motores — esa lista se construye antes de que los add-ons se
  registren. Asignarlo funciona igual. No concluyas de la lista que no
  está.
- **El identificador de EEVEE se movió**: `BLENDER_EEVEE_NEXT` en la 4.x
  y `BLENDER_EEVEE` en la 5. Se prueban los dos.
- **EEVEE es un rasterizador y necesita contexto de GPU.** Sin EGL no da
  error: **aborta el proceso** con SIGABRT tras imprimir una línea sobre
  `libEGL.so.1`, y eso no se puede capturar desde Python. Ahora se
  comprueba antes y se cae a Cycles avisando. Comprueba **EGL en
  concreto**: la primera versión aceptaba EGL, GL u OpenGL, y este
  contenedor tiene `libGL.so.1` y no tiene libEGL, así que pasaba la
  comprobación directo al abort que existía para evitar.

### El .blend con todo desmontado

`--save-blend` / `--open` ya existían; lo que no estaba claro es qué cae
dentro. Guardé uno y lo volví a abrir para comprobarlo:

```
engine   : CYCLES | samples: 320
objects  : ['Atoms', 'Bonds', 'Camera', 'Light0', 'Light1']
materials: mat_element_B, mat_element_C, mat_element_H, mat_element_N, mat_element_O
world    : Background, Output, TexCoord, TexVoronoi, ColorRamp, MixRGB, TexNoise, ...
```

Un material por elemento (o por tamaño de anillo, o por pieza) con su
árbol de nodos, las luces en sus posiciones escaladas, la cámara en el
ángulo elegido y el grafo entero del campo de estrellas en el mundo.
Todo datos normales de Blender para desmontar.

---

## nanocarbon_lab · Blender para portada de revista

### La cámara ya busca el mejor ángulo

Antes miraba siempre desde el mismo vector fijo `(1, -1.4, 0.6)`: un buen
tres cuartos para un objeto compacto y un ángulo arbitrario para todo lo
demás. Miraba **a lo largo** de una fila de estructuras y veía un
nanotubo **de punta**.

`--view auto` (lo que hace por defecto) lo **mide**: proyecta los átomos
en cada dirección candidata y se queda con la que da la silueta de mayor
área, que es la que enseña más estructura. Los ángulos con nombre
(`iso`, `hero`, `front`, `top`…) siguen estando para cuando quieras uno
concreto.

Y el encuadre se hacía con la **esfera envolvente**. Para una jaula da
igual; para una composición es un desastre: una fila de 77 Å de largo y
12 Å de fondo tiene una esfera de radio 39 Å, así que la cámara se
alejaba lo suficiente para que cupiera una bola de 78 Å y el sujeto
ocupaba **0,56 del ancho del cuadro frente al 0,86 pedido**. Ahora se
encuadra la **caja proyectada**, por separado en horizontal y en
vertical, y se suma el fondo del objeto porque su cara cercana está más
cerca que su centro.

### Efectos: láser, descargas, resonancias, halos, niebla y bloom

Siete, cada uno con sus parámetros, escritos
`nombre:clave=valor,clave=valor`:

| efecto | qué es |
|---|---|
| `laser` | un haz que llega a la estructura, con el punto donde incide |
| `arc` | filamentos dentados tipo descarga alrededor del objeto |
| `resonance` | capas concéntricas, cada una más tenue, para un modo que *suena* |
| `orbit` | anillos a distintos ángulos, como un átomo de libro de texto |
| `halo` | una cáscara suave de luz |
| `fog` | neblina, para que los haces se vean en el aire |
| `glare` | bloom sobre la imagen terminada |

```bash
nanocarbon render jaula.xyz --style nature_dark \
  --effect "laser:color=laser-green,azimuth=150" \
  --effect "arc:count=9" --effect resonance --effect glare
nanocarbon render --list-effects     # todos los parámetros de todos
```

**Todas las distancias van en radios del sujeto, nunca en Å**, así que un
efecto significa lo mismo en una jaula de 3 Å y en una red de 90 Å — la
misma propiedad que ya tenían las luces y por la misma razón.

Cuatro cosas salieron mal primero y cada una es ahora una regla:

- **Un objeto emisivo es una lámpara.** Doce arcos brillantes no sólo
  *parecen* un rayo: **iluminan todo**, y con algo de niebla en el aire
  lavan el cuadro entero. Ahora se les apagan los rayos difusos y
  especulares, así que brillan sin iluminar. El láser es la excepción:
  tiene que iluminar lo que golpea.
- **Dos puntos al azar en una esfera son antipodales de media**, así que
  cada arco salía tan largo como ancha es la estructura — en una
  composición, un zigzag que cruzaba toda la imagen. Ahora un parámetro
  `span` limita el ángulo entre los dos extremos y el arco se pega a la
  superficie, que es además lo que hace una descarga real.
- **Un haz apuntado al centro atraviesa la estructura** y sale por el
  otro lado, que se lee como una raya pintada encima. Ahora para en la
  **superficie**.
- **Los efectos se escalan con lo que la cámara ENCUADRA**, no con la
  esfera envolvente. Escritos como «1,1 radios» salían más grandes que
  el cuadro en una composición.

### Fondos: el universo

`space` y `nebula`, generados **proceduralmente** — ni hay que
distribuir un HDRI ni que encontrarlo al renderizar, y una portada de
6000 px sigue teniendo estrellas que son puntos y no manchas. Las
estrellas son un campo de **Voronoi** recortado justo por encima de
cero, no ruido: el ruido da borrones que se leen como polvo, y Voronoi
tiene un mínimo nítido por celda, así que el corte deja una dispersión
de puntos de brillos distintos.

### Materiales de verdad, y cinco estilos de portada

El Principled BSDF de Blender 4.3+ trae cosas que no estaban
aprovechadas, y ahora sí: **anisotropía** (metal cepillado, cuyo brillo
se estira en vez de ser un punto redondo — ninguna rugosidad sola hace
eso), **interferencia de película fina** (iridiscencia real que *cambia
con el ángulo de vista*, porque se calcula como interferencia y no se
pinta con una rampa de color), tinte del barniz y tinte del brillo
especular (el oro refleja oro).

Cinco estilos nuevos, y cada uno **trae sus propios efectos**, porque un
estilo que trata de un láser debería llegar con el láser puesto:

| estilo | qué es |
|---|---|
| `cosmos` | la estructura sola en el espacio profundo: estrellas, nebulosa, oro pulido con barniz tintado |
| `plasmon` | metal cepillado anisótropo con capas de onda estacionaria alrededor |
| `laser_probe` | un haz incidiendo en una sala oscura con neblina, el rayo visible en el aire |
| `electric_storm` | descarga: filamentos sobre la estructura contra negro, con halo |
| `iridescent` | película fina — pompa de jabón o titanio anodizado |

Diecisiete estilos en total. `--no-style-effects` quita los que trae el
estilo.

### El bloom se hace sobre la imagen, y eso está medido

Blender 5 movió el compositor a `scene.compositing_node_group`, y
asignarle uno en un `bpy` sin interfaz **devuelve un fotograma negro**:
0,0003 de media frente a 0,2125 para la misma escena sin él,
reproducido sobre una esfera emisiva sola en la escena. Una función que
deja el render en negro en silencio es peor que no tenerla.

Así que el bloom es un paso sobre la imagen terminada: umbral, desenfoque
y suma. Funciona en cualquier Blender, no cuesta ni una muestra más, y
—al revés que un grafo de nodos— se puede probar en un Python normal.
Dos detalles que importan: el paso de brillos tiene **techo** además de
suelo, porque un filamento emisivo es muchas veces más brillante que el
blanco en luz lineal y sin tope su resplandor se come el cuadro; y el
resplandor **conserva el color** de lo que brilla, porque un halo blanco
alrededor de un láser verde es el reflejo de otra fotografía.

### Toda la suite dentro de Blender

Blender ejecuta Python, así que puede ejecutar este paquete: construir y
renderizar **en un solo proceso** en vez de pasar archivos entre dos.

```bash
blender -b -P nanocarbon_lab/blender/nanocarbon_blender.py -- --check
blender -b -P nanocarbon_lab/blender/nanocarbon_blender.py -- \
  --build "fullerene:freq=2" --style cosmos --out portada.png
```

**Renderizar un bundle ya construido no necesita nada más que Blender**
— ese camino es sólo numpy a propósito, y numpy es lo que Blender trae.
*Construir* necesita `ase`, `scipy`, `networkx` y `scikit-image` dentro
del Python de Blender, y `--check` dice cuáles faltan e imprime la orden
`pip install` exacta con la ruta de **ese** intérprete ya puesta. Existe
porque si no, el fallo es un ImportError desde dentro de una aplicación
3D, sin ninguna pista de que la solución es un install en un intérprete
que no sabías que estabas usando.

### Y el bug de los enlaces en la costura, arreglado de verdad

Antes se tiraban y ya. Ahora hay tres respuestas honestas y la de por
defecto es la útil:

- **`stub`** (por defecto) dibuja un **medio enlace** apuntando hacia
  donde la estructura continúa, que es verdad y es lo que hace un visor
  cristalográfico en una cara de celda: el tubo termina en enlaces que
  van a alguna parte, no en un anillo de átomos pelados.
- **`drop`** los quita, para una pieza que de verdad es finita y cuya
  periodicidad sólo era una comodidad para la celda DFT.
- **`Item.repeat`** convierte la costura en **interior** repitiendo la
  pieza: un tubo (5,5) con `repeat=(1,1,3)` pasa de 150 enlaces con 14
  en la costura a **436 enlaces con 14 en la costura**, todos de
  1,410–1,419 Å.

El informe dice cuántos en cualquier caso.

---

## nanocarbon_lab · composición de varias estructuras, y un Blender de verdad

### Juntar 2, 3 o más estructuras en una figura

Panel nuevo en la columna derecha de la ventana —**«Composition (several
structures)»**— y el subcomando `nanocarbon scene`. Metes lo que ya tengas
construido (lo que esté en la vista previa, o archivos del disco: carbono,
nanopartículas, TMD, lo que sea), lo colocas en el espacio, y sale **un
solo archivo** que abres en VESTA o en Blender.

```bash
nanocarbon scene c60.xyz tubo.xyz nanoparticula.xyz \
    --label cage --label tube --label NP \
    --layout row --gap 8 --rotate tube:0,90,0 --out figura
```

Cuatro disposiciones (`row`, `stack`, `grid`, `circle`) y después ajustas
cada pieza a mano: **mover** en Å y **girar** en grados, por su propio
centro. En la ventana es lo mismo con dos filas de casillas y el botón
«Apply to selected item».

**Lo que NO es.** Esto es una **ILUSTRACIÓN, no una estructura.** Las
piezas están donde se ven bien, no donde las dejó una relajación; no hay
intercara relajada entre ninguna de ellas y **las distancias no son
física**. El informe lo dice en su primer aviso, siempre, y el archivo
lleva la marca `is_illustration`. No la cuantifiques, no la relajes y no
la metas en un código DFT.

**Lo que sí hace bien, y es el punto delicado.** Los enlaces y los anillos
**se arrastran y se renumeran, no se vuelven a adivinar.** Si se
recalcularan por distancia, dos estructuras que simplemente están a 4 Å
una de otra saldrían **enlazadas entre sí**, que es lo único que una
composición no puede afirmar. Hay un test que comprueba que ningún enlace
y ningún anillo cruza de una pieza a otra.

Tres trampas que aparecieron midiendo, no pensando:

- **Un hueco de 8 Å son 8 Å.** La primera versión espaciaba por el radio
  de la esfera que envuelve cada pieza y no miraba su rotación: un hueco
  pedido de 8 salía a **17,9 Å** en fila y **23,3 Å** en rejilla. Ahora se
  mide con la caja envolvente **ya girada**, y `row`, `stack` y `grid`
  aciertan a una décima de Å. `circle` reparte en un anillo, así que ahí
  el hueco es un mínimo y no un objetivo.
- **Los enlaces que salían por la cara de una celda periódica se tiran.**
  Una composición es finita: la imagen a la que llegaba ese enlace no
  está en la escena. Dibujado tal cual, un tubo (5,5) daba una raya de
  **12,4 Å** teniendo todos sus enlaces reales a 1,42. Se detectan con la
  traslación exacta, no por longitud (un (6,0) repite cada 4,26 Å), y el
  informe dice cuántos y qué hacer: repetir la pieza antes de componerla.
- **Escalar una pieza cambia todas sus distancias de enlace.** Es una
  decisión legítima para una figura y deja de ser una estructura; queda
  registrado y avisado.

Componer sustituye la vista previa, así que los botones de exportar y de
renderizar valen ya para la composición sin tocar nada más.

### Los anillos ya son opcionales: colorea por elemento, por pieza o plano

El render tenía los anillos metidos en el estilo, así que **no se podía
apagar**. Ahora son **tres ejes independientes**:

| eje | qué decide | opciones |
|---|---|---|
| `--style` | el *carácter* de la superficie: metálico, rugosidad, barniz, shader, luces, objetivo | 12 estilos |
| `--colour-by` | **qué significa el color** | `ring`, `element`, `structure`, `uniform` |
| `--background` | el fondo | `gradient`, `solid`, `transparent`, `studio` |

Si no dices nada, manda el estilo: uno de portada sigue coloreando por
anillo y uno de figura colorea por **elemento**. Y un estilo no se pierde
al recolorear — por elemento dentro de `gold_nanotech` sale un oxígeno con
acabado de oro, no un primario plano.

Dos cosas salen de ahí y ninguna es estética:

- **El enlace se parte en dos mitades cuando el color es del átomo.** Un
  enlace C–O pintado del gris del carbono dice que el oxígeno es carbono.
  Por elemento y por pieza se dibujan dos medios cilindros; por anillo se
  mantiene uno, porque un anillo no es de ninguno de los dos extremos.
- **El radio sigue al elemento cuando el color lo hace.** Esferas iguales
  están bien en una red de carbono y mal en cuanto hay un hidrógeno y un
  átomo de oro en la misma imagen.

Y la razón de todo esto: **colorear por anillo dice poco en una jaula
cerrada y nada en una dopada.** Los 60 átomos del C60 están en un
pentágono, así que la imagen decía «todo pentágonos» de algo que son doce
pentágonos y veinte hexágonos.

### Doce estilos, fondos y shaders especializados

Los cinco de revista siguen ahí. Siete nuevos, y cuatro de ellos usan un
**shader que un Principled no puede expresar con parámetros**:

- **`paper_figure`** — blanco plano, colores de elemento, luz suave, sin
  desenfoque. Una figura de manuscrito, no una portada.
- **`electron_micrograph`** — gris, luz rasante. Por defecto colorea
  **uniforme**: una micrografía que codifica sus defectos en color vuelve
  a ser un dibujo.
- **`glass_cryo`** — vidrio **refractivo**, se ve el otro lado de la jaula
  a través del cercano. 512 muestras, que es lo que piden las cáusticas.
- **`clay_studio`** — barro mate sobre un **suelo** de estudio. Puramente
  difuso: la forma sin que el material distraiga.
- **`neon_cyber`** — el cuerpo y los enlaces **son** la luz.
- **`toon_schematic`** — sombreado en bandas, para una diapositiva.
- **`transparent_overlay`** — **sin fondo**, alfa directo, con un suelo
  que **recoge la sombra sin verse**, para montarlo en una maqueta. Sin
  eso el objeto queda pegado y no apoyado.

El **suelo** es tri-estado y no un simple sí/no: algunos estilos lo traen
y la mayoría no, así que la casilla de la ventana **se rellena desde el
estilo** cada vez que cambias de estilo o de fondo. Dejada en «no» le
quitaría a `clay_studio` el suelo que es su razón de existir.

### Abrir Blender y terminar la escena a mano

Botón nuevo: **«Open scene in Blender…»**, y `--open` en la línea de
órdenes. No renderiza: monta las luces, la cámara, los materiales y el
fondo, guarda el `.blend` al lado del archivo y **te entrega la ventana de
Blender viva**. Ningún preajuste sustituye mover una cámara a mano, y aquí
es donde el programa deja de adivinar.

Necesita la **aplicación** Blender: el paquete `bpy` de PyPI es un Blender
entero dentro de Python pero **no tiene ventana**, así que pedirle una
dice por qué no puede en vez de renderizar en silencio y decir que ha
ido bien. También hay `--save-blend` y `--no-render` sueltos, por si lo
quieres montar sin abrir nada.

### Y renderizar ya no es solo un botón de la ventana

`nanocarbon render` existe. Antes el pipeline estaba encerrado detrás de
un módulo que necesita tkinter, lo cual es el sitio equivocado: una
figura normalmente se quiere desde el mismo script que construyó la
estructura, y un estilo se compara en un bucle sobre varias estructuras.

```bash
nanocarbon render figura.xyz --style paper_figure \
    --colour-by structure --background studio --out figura.png
nanocarbon render --list-styles          # los doce, con su fondo y su color
```

De paso: **el menú de estilos de la ventana era una lista escrita a mano
de cinco nombres**, y llevaba equivocada desde que se escribió el sexto.
Ofrecía la mitad del catálogo sin nada que indicara que existía el resto.
Ahora lee el catálogo, y hay un test que falla si alguien vuelve a copiar
un nombre de estilo dentro de la ventana.

---

## nanocarbon_lab · importa un .cif y corta nanopartículas

Pestaña nueva en la ventana —**familia «nanoparticle»**— y dos subcomandos
nuevos. Abres un `.cif` (el tuyo, o uno de una base de datos), eliges una
forma, y sale la nanopartícula cortada de ese cristal. Después, si quieres,
la recubres de carbono grafítico.

**Las formas: 23, en cuatro grupos que significan cosas distintas.**

- **Redondas** — esfera, elipsoide, barra (de puntas redondas o cortadas),
  disco, cono o tronco de cono, toro. Son cortes *en el marco del
  laboratorio*: el corte expone las facetas que le toquen, que es lo que
  tiene una partícula templada o molida y no una facetada.
- **Bloques y prismas** — cubo, caja, prisma de n lados, plaquetas,
  bipirámides.
- **Hábitos facetados** — octaedro, cuboctaedro, octaedro truncado
  (el sólido de Kelvin), tetraedro, dodecaedro rómbico. Aquí las caras
  **son planos de red**, y para construirlas hace falta el cristal.
- **Envolventes de cinco pliegues** — decaedro e icosaedro. Aviso honesto
  y en el docstring: una partícula decaédrica real son cinco cristales
  *maclados*, y lo que corta esto es la silueta, no las fronteras de macla.
- **Composiciones** — estrella, tetrápodo, pesa, cubo cóncavo.

Y cualquiera de las 23 puede ser **hueca** (`--hollow 5`): una nanocaja,
una jaula, un cascarón. No son 23 entradas más del catálogo: una forma es
una función de distancia con signo, así que hueco es una diferencia y una
estrella es una unión suave. Está también la **construcción de Wulff** con
*tus* energías superficiales — no trae tabla propia y no la va a traer,
porque dependen de la terminación, de los adsorbatos y del método, y
ponerla sería tomar la decisión más importante de la construcción por ti.

**El recubrimiento grafítico.** Una o varias cortezas (una nano-cebolla,
que es lo que da el arco eléctrico), a 3.35 Å del átomo más cercano del
núcleo. Pasa por la misma tubería que las uniones y las schwarzitas, así
que el censo de anillos no se coloca: lo fija la topología del campo y el
chequeo de Euler lo verifica. Medido `suma(6-n) = +12` exacto en las seis
formas probadas. Y luego se dopa (N, B, P, S… donde quieras, incluso sobre
los pentágonos) y se funcionaliza (–OH, –COOH, =O, –NH2, –F…).

**El dopado va en la CORTEZA, nunca en el núcleo.** Poner un nitrógeno
donde había un hierro no es dopar: es otro compuesto.

**Tres cosas que conviene saber antes de usarlo.**

1. **Dónde cae el centro decide la composición de un compuesto, y por
   tanto su carga.** En sal de roca, centrado en el catión el cubo sale
   rico en catión y centrado en el anión, rico en anión. No hay opción
   neutra; `--centre cell` es solo la que no pide información extra. Con
   `--stoichiometric` se recorta, y el recorte es la receta habitual y es
   una heurística, y lo dice.

2. **Si una faceta aterriza sobre un plano de red, el conteo depende del
   último dígito del tamaño.** Un cubo de arista 4a de oro fcc centrado en
   un punto de red lee **171, 256 o 365 átomos** según un cambio de una
   parte en un millón — factor 2.1. Son tres terminaciones distintas y las
   tres son reales; lo que estaría mal es no decirlo, así que se avisa. Y
   desplazando el corte para que las caras caigan *entre* los planos, el
   número es exactamente el de la red (4n³ para todo n probado). El
   programa **no mueve el corte por su cuenta**: qué plano termina una
   faceta es una afirmación sobre la partícula.

3. **Una pared recubierta nunca sale tan buena como una jaula libre, y no
   debería.** Una corteza que se ajusta a un núcleo facetado está tensa, y
   «tensa» es el veredicto honesto. Lo que sí sería un fallo es carbono
   dentro del metal, y eso es lo que evita el anclaje. Se informa del hueco
   en tres números (media, percentil 5, mínimo) y de los enlaces en dos
   (p95 y máximo), porque el extremo sobre mil enlaces es cola: una pared
   con p95 1.48 Å y máximo 1.58 Å es una pared buena con una esquina tensa.

**Y 3.35 Å es un contacto de van der Waals, no un enlace.** Fe, Co, Ni, Ti
y los demás formadores de carburo quimisorben hacia 2.1 Å, que es otra
interfaz con otra respuesta electrónica. Se ofrecen las dos, el aviso sale
en los dos casos, y nada aquí calcula cuál prefiere tu sistema.

**Comprobado contra algo de fuera del framework**: el octaedro cortado
reproduce exactamente el conteo de `ase.cluster.Octahedron` en cinco
tamaños (44, 85, 146, 231 y 489 átomos), lo que fija toda la cadena —
familia de Miller, poliedro de semiespacios, supercelda, corte — sin que
el paquete se corrija a sí mismo.

**Pendiente de que lo pruebes tú**: el panel de la ventana está escrito y
pasa las comprobaciones estáticas, pero este contenedor no tiene tkinter,
así que **no he podido abrirlo**. La biblioteca y la línea de comandos sí
están corridas de punta a punta.

---

## ramancarbon · el VAMAS sale con la deconvolución dentro

El archivo ya abría en CasaXPS, pero abría **limpio**: la medida y nada
más. No era un fallo del archivo. Un VAMAS no tiene ningún campo para
decir «esta región se ajustó con cinco componentes», así que el ajuste se
quedaba donde estaba, en el programa. CasaXPS resuelve eso metiendo su
propia gramática en el comentario del bloque, y todo lo que lee archivos
de CasaXPS lee **eso**.

Ahora se escribe. Con **Fondo: incluido**, cada bloque con ajuste sale con
su región (la ventana y el tipo de fondo) y una componente por pico, con
posición, anchura **medida**, área y los límites que usó el ajuste. La
forma de línea no hay que traducirla: `gl` y `sgl` se escribieron desde el
principio como el `GL(p)` y el `SGL(p)` de CasaXPS, con `mixing` igual a
`p/100`.

**Un doblete sale como dos picos.** Aquí es una componente —la separación
espín-órbita y la razón de áreas las fija la degeneración, no el ajuste—
pero CasaXPS no tiene esa idea y dibuja lo que le den: con el área de los
dos puesta en el 2p3/2 pintaría un pico donde el espectro tiene dos. El
par se parte al escribirlo, con el mismo cociente que lo ataba, y el
factor de sensibilidad va entero en las dos mitades, que es lo que hace
que la cuantificación de CasaXPS salga bien.

**Lo que NO hace.** No escribe las restricciones como restricciones de
CasaXPS. Salen los números del ajuste; las ligaduras entre componentes,
no. Es deliberado: una restricción mal codificada no parece mal, parece
un ajuste bien dibujado que se va a otro sitio en cuanto alguien pulse
*Fit*. Unos números sueltos se vuelven a atar a mano; unos atados al
hermano equivocado no se notan.

**La trampa.** La deconvolución solo cabe en la exportación con **fondo
incluido**. Las áreas se apoyan en un fondo que CasaXPS calcula por su
cuenta, así que sobre un espectro que este programa ya restó lo restaría
una segunda vez y encogería todos los picos. Con «fondo restado» el
archivo sale sin anotación y el registro lo dice.

Y una sola cosa conviene mirar a ojo al abrirlo: las componentes
**metálicas**. El Doniach-Šunjić de CasaXPS lleva un segundo parámetro
propio que no es la anchura gaussiana de aquí, y la asimetría es lo único
que pasa tal cual.

---

## ramancarbon · el eje de textura lo busca el programa

Antes había que escribirlo, y escribirlo es ya saber la respuesta. Peor:
la pregunta se contesta mal por una razón que no es de la muestra. El eje
es una **dirección**, no una reflexión observada: un carbono tumbado está
texturado según (0 0 1) mientras su difractograma enseña solo la 002 —la
(0 0 1) está sistemáticamente ausente— así que mirar el propio patrón
lleva a descartar (001), que es justo el eje.

Ahora hay un botón **«Buscar eje de textura»** en la pestaña Rietveld. El
programa saca los candidatos de las propias reflexiones de cada fase,
reducidas a su dirección primitiva (la 002 se convierte en (0 0 1) sola),
prueba cada uno y dice cuál encontró, cuánto baja Rwp y por cuánto gana
al siguiente. Los ejes que acepta quedan puestos en el modelo para el
siguiente refinamiento.

Es un botón y no el comportamiento por defecto porque cuesta un ajuste
por eje candidato y por fase: unos diez minutos con cinco fases. Quien
lo quiera en línea escribe «auto» en el campo «Eje de textura».

**Y la textura no se había refinado nunca.** Con el eje verdadero
declarado a mano, sobre un patrón sintético construido con r = 0.45, el
refinamiento devolvía r = 1.000000 y un Rwp idéntico al del ajuste sin
textura. Dos causas, las dos arregladas: el caché de reflexiones
redondeaba a ocho decimales los parámetros que se refinan —y el paso de
diferencias finitas es 1.5·10⁻⁸, así que esa columna del jacobiano salía
exactamente cero— y, aun con eso arreglado, para cuando corre la etapa de
textura las anchuras ya se la han comido. Por eso cada candidato se prueba
desde un arranque fresco.

**La trampa.** El veredicto tiene tres valores, no dos. Sobre un carbono
turbostrático —cuatro reflexiones— el mejor eje era el verdadero siempre
y ganaba al siguiente por un 0.1 %: eso no es «el eje», es «el más
probable». Cuando pasa, lo dice con esas palabras y enseña el ranking.
Zanjarlo pide una figura de polos o una segunda medida con la muestra
girada.

---

## ramancarbon · cuántas capas tiene el carbono

d₀₀₂, L_c, L_a, el **número de capas** y el grado de grafitización estaban
calculados desde hacía versiones y solo se podían ver con `ramancarbon
micro` en la línea de comandos. Ahora salen en la pestaña Fases.

**La trampa.** La línea que los elegía era «el pico más cercano a 26.5°»,
y eso siempre encuentra uno: sobre un patrón sin carbono devolvía una
altura de apilamiento calculada con la reflexión del hierro. Ahora se
busca en una ventana de espaciado y se dice que no cuando está vacía. Y
la ventana de la 100 del carbono cae encima de las líneas fuertes del
hierro y de la cementita: en el patrón real, sin filtrar, la línea de
cementita de 43.89° daba L_a = 36.8 nm para un carbono cuyo apilamiento
mide 6.7 nm. Los picos que otra fase ya explica no se usan para L_a.

---

## ramancarbon · una fracción en peso es una afirmación sobre el FONDO

El Rietveld de un CVD real de carbono sobre FeSe daba **51 % de carbono en
peso** con un 002 que apenas se ve, y el ajuste no tenía nada malo a la
vista: Rwp 4.39 %, GOF 1.05, convergido, curva diferencia limpia. El número
era el que había que desconfiar, y ahora el programa lo dice.

Tres medidas nuevas, todas informadas y ninguna de las cuales cambia el
ajuste:

- **Cuánto del patrón es fondo** (`background_share`). En ese refinamiento,
  el 96.8 %: las cinco fases explican el 3.2 % restante y todas las
  fracciones salen de ahí.
- **Qué parámetros no separa la medida.** La covarianza se calculaba entera
  y se tiraba menos la diagonal. `cero` contra `desplazamiento` salía a
  −1.0000 —un grado de libertad con dos nombres, con sus dos
  incertidumbres calculadas como si fueran independientes.
- **Peso frente a señal** (`phase_contributions`). El carbono era 52.4 %
  del peso y 17.9 % de la intensidad difractada. No es incoherente (la
  dispersión va como Z² y el carbono tiene Z = 6 contra 26 del hierro),
  pero es el número que faltaba para poder juzgar el otro.

Y un botón, **«Probar el fondo»**, que refina otra vez con varios órdenes
de polinomio y compara. En ese patrón:

| orden | Rwp % | C | Fe₃C |
|---|---|---|---|
| 2 | 5.34 | 25.9 % | 52.7 % |
| 4 | 4.90 | 36.6 % | 32.3 % |
| **6 (el defecto)** | 4.39 | **52.4 %** | 16.3 % |
| 8 | 4.05 | 33.1 % | 37.7 % |
| 10 | 4.00 | 33.1 % | 36.5 % |

**La trampa.** La covarianza NO ve este efecto y suponer que sí fue el
primer diseño de la comprobación: la escala del carbono correlaciona con
los coeficientes del fondo a 0.33 como mucho. Dado un polinomio de orden
seis, la escala está bien determinada; lo que no está determinado es el
orden, y un orden es una elección discreta, no un parámetro con derivada.
Por eso hace falta el barrido, y cuesta un refinamiento por orden.

Además, la etapa de orientación preferente se saltaba **en silencio**
cuando ninguna fase declara eje de textura, que es lo que pasa por
defecto. Es la única etapa que toca las intensidades relativas, así que
si las alturas no casan y las posiciones sí, ése es el sitio por donde
empezar. Ahora lo dice.

---

## ramancarbon · un ancho cerca de su techo tiene dos lecturas

Una banda genuinamente ancha y una que está tapando el hueco de una
componente que falta se ven **exactamente igual** en el resultado: un
número cerca de un límite. Significan cosas opuestas.

El botón **«Probar el techo de anchura»** corre el único experimento que
las separa —subir el techo y volver a ajustar— solo para las componentes
que acabaron cerca del suyo. Sobre un espectro real, con los dos casos en
el mismo ajuste de cinco bandas: la D volvía en 191.0 con el techo en 200
y en 185.3 con el techo en 400 (nunca empujaba), mientras la D4 volvía en
250.0, 300.0 y 400.0 (siempre en el borde).

Y en la tabla de componentes, **tu número gana**: un centro o una anchura
escritos fuera de la ventana publicada de la banda se admiten ensanchando
el límite, en vez de recortarlos en silencio. Eso último era lo que hacía
que «editar y picar Ajustar» pareciera no hacer nada.

**La trampa.** Una componente que sale de la ventana de su banda deja de
ser prueba de esa banda, y el aviso lo dice. Es tu muestra y tu número,
pero ya no es una medida de la D3.

---

## tbkit · la ventana (`tbkit-gui`)

`uv run tbkit-gui molecula.xyz` abre todo tbkit en una ventana: estructura y
modelo a la izquierda, vista 3D en el centro y una pestaña por capacidad
(electrónica, orbitales, magnetismo, geometría y modos, espectros Raman/IR,
grafeno). Desde carbonforge, «Abrir en tbkit» manda la estructura actual.

**La trampa.** Necesita el extra `gui` (PySide6 + pyvista) y, en Linux, las
bibliotecas de Qt del sistema. Cancelar descarta el resultado, no para el
cálculo. Las frecuencias con O de `xu_chno` están en revisión (enlaces O–H y
C=O demasiado rígidos en algunas moléculas): no te fíes todavía de las
posiciones IR/Raman de grupos oxigenados.

---

## tbkit · oxígeno (`--model chno`)

`tbkit raman molecula.xyz --model chno` y `tbkit ir molecula.xyz --model chno`
ya aceptan oxígeno: hidroxilos, epóxidos, carbonilos, carboxilos, éteres,
furanos, nitro, y los motivos del óxido de grafeno. Es `xu_chn` más O,
ajustado a GPAW con la misma receta.

**Cuánto vale.** El epóxido basal sobre coroneno queda cerrado (C–C a 0,03 Å
de GPAW, C–O a 0,10 Å); carboxilo, carbonilo, éter y nitro a 0,01–0,03 Å; la
polarizabilidad a ±10 % de GPAW (±3 % en las moléculas de prueba).

**La trampa.** Los anillos de tres miembros necesitan una corrección propia
(cero para ángulos ≥ 80°, así que el carbono sp²/sp³ normal no cambia). El
ciclopropano sigue con el C–C 0,19 Å largo y el biciclobutano se abre: no los
uses con este modelo. Las energías solo son comparables dentro de una misma
composición.

---

## tbkit · infrarrojo

`tbkit ir molecula.xyz --model chn` calcula las intensidades IR (km/mol) con el
dipolo del propio modelo, el mismo que da la polarizabilidad: IR y Raman salen
coherentes. Acepta modos de QE (`--modes`).

**Cuánto vale.** Frente a GPAW en ocho moléculas, los modos intensos aparecen
donde deben y con el orden de magnitud correcto; el error típico es un factor
~2 por modo, con fallos claros en algún caso (las tensiones C–H de la
metilamina). Es semicuantitativo: sirve para asignar y comparar tendencias,
no para intensidades absolutas.

**Lo que NO hace.** Cristales (no hay polarización de Berry).

---

## tbkit · nanotubos, grafeno de segundo orden y análisis de modos

**Nanotubos.** La RBM de once tubos (zigzag, armchair, quirales) sale a entre
−1,3 y −4,0 % de la relación 227/d sin ajustar nada; la G, desplazada hacia
arriba y con un desdoblamiento G⁺/G⁻ demasiado pequeño (a Xu le falta la
anomalía de Kohn de los metálicos). `python -m tbkit.recipes.cnt_validation`.

**Grafeno.** `tbkit graphene-raman --laser 1.96 2.41 2.80` calcula G, 2D y 2D′
por doble resonancia. La 2D se desplaza 112 cm⁻¹ por eV de láser (medido: ~100);
sus posiciones absolutas salen ~190 cm⁻¹ altas, que es lo que dan los fonones
PBE. No hay banda D (no hay defectos en el cálculo).

**Qué es cada modo.** `tbkit.modes` dice qué átomos y qué especie llevan cada
modo, si es radial, tangencial o axial, su simetría, la DOS vibracional
proyectada, y cuánto de un desplazamiento se debe a la masa del dopante y
cuánto al cambio de enlace. Una frecuencia de la DOS no es un pico Raman: eso
lo decide la actividad.

---

## tbkit · Raman resonante

`tbkit raman butadieno.xyz --model chn --resonant 2.33 3.5 4.0` da la actividad
de cada modo a cada energía de láser y el perfil de excitación de cada modo.
Se calcula con la polarizabilidad compleja a la frecuencia del láser, con su
apantallamiento completo. Cerca de una transición los modos que la acoplan se
disparan: el C=C del butadieno crece unas 3000 veces y domina el espectro.

**Lo que NO hace.** No incluye la estructura vibrónica (sobretonos, forma de
Franck–Condon). Las energías de resonancia son las del modelo, más bajas que
las ópticas reales (la del butadieno sale a 4,25 eV y se mide a ~5,9): compara
perfiles, no energías. En el grafeno este camino converge mal; tendrá su
propio cálculo.

---

## tbkit · polarizabilidades cuantitativas

La α del modelo ya no se queda corta: cada átomo lleva una polarizabilidad
extra (H 0,43, C 0,95, N 0,74 Å³) por lo que su base mínima no puede
polarizarse, apantallada junto con el resto. Los tres números salen de ajustar
los tensores α de 12 moléculas calculados con GPAW en rejilla real.

**Cuánto vale.** En cinco moléculas que el ajuste no vio, la α media queda a
±2 % de GPAW y la anisotropía se reproduce (el benceno: 6,3 fuera del plano y
13,4 en él, frente a 6,9 y 13,0). C₆₀ da 84,5 Å³ (medido: 76,5 ± 8) y el
diamante ε∞ = 5,07 (medido: 5,7), sin haberlos usado.

**Lo que NO hace.** En cristales la parte extra se suma sin campos locales (no
hay apantallamiento periódico todavía). Las referencias son PBE, que ya
sobreestima las α un 4–7 %. Energías, fuerzas y geometrías no cambian: esto
solo toca la respuesta óptica.

---

## tbkit · la polarizabilidad ya no es cero fuera del plano

Hasta ahora, la α del modelo era exactamente cero en la dirección
perpendicular a una cadena y fuera del plano de una molécula plana, y eso
fijaba razones de despolarización artificiales (1/3 en HCN, 0,125 en la
respiración del benceno). Ahora el operador de posición incluye el dipolo entre
los orbitales s y p de cada átomo, calculado con el átomo libre de GPAW, y el
apantallamiento incluye dipolos atómicos sin ningún parámetro nuevo. Se activa
solo en `xu_carbon` y `xu_chn`; `onsite_dipoles=False` recupera lo anterior.

**Qué mejora.** Desaparecen los ceros y las ρ fijas; la α de C₆₀ pasa de 61,7
a 69,0 Å³ (medida: 76,5 ± 8).

**Lo que NO arregla.** A lo largo de los enlaces σ la α baja (es correcto: los
dipolos de transición unen centroides de híbridos, no núcleos), así que la del
benceno queda en 8,5 Å³ en el plano y 1,5 fuera (medidas: ~12,3 y ~6,5) y la
ε∞ del diamante en 2,96 (medida: 5,7). Lo que falta es polarización que una
base mínima no tiene: las α absolutas siguen siendo orientativas.

---

## tbkit · fonones de Quantum ESPRESSO con intensidades de tight binding

`tbkit raman estructura.xyz --model chn --modes dynmat.out` toma frecuencias y
modos de QE y calcula las intensidades con la polarizabilidad del modelo. Vale
cualquier archivo de modos de `dynmat.x` o `matdyn.x` (desplazamientos o
autovectores; se detecta). La estructura tiene que ser la del cálculo de QE, con
los átomos en el mismo orden.

**Lo que NO hace.** No lee la geometría del archivo de modos (no la trae); no
comprueba el orden de los átomos más allá de su número; y no se ha probado
todavía contra la salida de una instalación real de QE.

**Una limitación que salió al probarlo.** La α del modelo es cero en la
dirección perpendicular a una cadena y fuera del plano de una molécula plana
(solo se mueven cargas atómicas). Las intensidades de los modos en el plano son
las fiables; el arreglo está planificado.

---

## tbkit · hidrógeno y nitrógeno: `--model chn`

`tbkit raman piridina.xyz --model chn` ya funciona: el nuevo conjunto `xu_chn`
añade H y N al carbono de Xu sin tocar el C–C, con cargas autoconsistentes.
Sirve para cintas pasivadas, copos y moléculas con N piridínico, pirrólico o
en aminas y nitrilos.

**De dónde sale.** De GPAW (PBE), con dos recetas que se pueden repetir:
`tbkit.recipes.chn_references` calcula 118 geometrías de 22 moléculas
(relajadas, desplazadas y escaladas), y `tbkit.recipes.xu_chn` ajusta primero a
los niveles y después a niveles, fuerzas y energías a la vez. Las referencias
van dentro del paquete y el archivo de parámetros guarda su huella SHA-256.
Las U de Hubbard se calculan con el átomo de GPAW y salen iguales a las de
DFTB mio.

**Cuánto vale.** Enlaces C–H, N–H y aromáticos a ≤ 0,02 Å de GPAW (también en
un coroneno con N que el ajuste no vio); frecuencias a 50–70 cm⁻¹ en CH₄, NH₃,
benceno y piridina; el Raman del benceno tiene exactamente 2A₁g + 4E₂g + E₁g y
el de la piridina pone sus dos modos de anillo polarizados a 992 y 1028 cm⁻¹
(medidos: 991 y 1030).

**Lo que NO hace.** Solo sistemas finitos de capa cerrada (sin Ewald no hay
SCC periódico). La α media sale entre un 35 y un 70 % de la experimental,
sobre todo porque la componente fuera del plano es cero (ver el apartado de
QE): las intensidades absolutas no valen. Las energías solo se comparan entre geometrías de la misma
composición. Los C–C y C–N simples junto a un heteroátomo se desvían ~0,08 Å,
y los anillos tensos (aziridina) fallan.

**`.skf` de DFTB.** La convención del integral sp heteronuclear está
comprobada contra el lector de DFTB+, y un conjunto `.skf` se usa con SCC por
defecto (`--scc`/`--no-scc` lo cambia).

---

## tbkit · Raman con tight binding (primer orden, no resonante)

`tbkit raman estructura --model sp3` calcula el espectro Raman con el propio
modelo: sus fonones en Γ, la polarizabilidad de cada geometría desplazada y, por
modo, la actividad y la razón de despolarización. `-o raman.csv` guarda el
espectro ensanchado con los factores de láser (ν_L − ν)⁴ y de Bose.

**Cómo se ha comprobado:** por simetría, que no depende de ningún parámetro. En
el diamante sale un solo triplete activo con el tensor y la despolarización de un
modo T₂g; en C₆₀, exactamente las 10 frecuencias activas que exige su simetría
(2 A_g polarizadas y 8 H_g despolarizadas), con las A_g en 508 y 1607 cm⁻¹
(exp. 496 y 1469) y las H_g entre 237 y 1651 cm⁻¹ (exp. 273 a 1575). La polarizabilidad de C₆₀ sale 62 Å³
con apantallamiento (exp. 76,5 ± 8) y la constante dieléctrica del diamante 4,75
(exp. 5,7).

**Qué NO hace, y dónde está la trampa:**

- Solo Raman **no resonante**: se rechazan metales, semimetales (el grafeno,
  cuyo Raman es siempre resonante), moléculas de capa abierta y láseres cerca del
  gap.
- Por ahora **solo carbono puro** (modelo de Xu): cintas con H o N necesitan
  parámetros `.skf` o un ajuste propio.
- En cristales la polarizabilidad es de partículas independientes (sin el
  apantallamiento que sí tienen las moléculas).
- La banda 2D del grafeno (segundo orden, doble resonancia) no está.

## tbkit · energías, fuerzas, relajación y fonones

**Parte repulsiva.** Con el modelo sp³ de carbono de Xu (y con archivos `.skf`
que traen su spline), tbkit calcula ahora energías totales y fuerzas, así que
puede **relajar** estructuras y calcular **fonones en Γ**:

```bash
tbkit relax cluster.xyz --model sp3 -o relajado.extxyz
tbkit phonons diamante.extxyz --model sp3 --kmesh 8
```

Sin reajustar nada, el modelo da el diamante con 3,555 Å (exp. 3,567), el enlace
del grafeno 1,42 Å, una cohesión de 7,24 eV/átomo (exp. 7,37), la línea Raman del
diamante en 1224 cm⁻¹ (exp. 1332: −8 %) y la banda G del grafeno en ~1670 cm⁻¹
(exp. 1582: +6 %). Son los errores típicos de un modelo empírico; se informan, no
se esconden.

**Reproducibilidad.** `tbkit run simulacion.json` ejecuta una tarea (niveles,
DOS, bandas, Hubbard, relajación, fonones) y guarda un registro con la
estructura, **todos** los parámetros, la versión, el commit, la fecha y los
ajustes; con ese registro solo, el cálculo se repite.

**Parámetros con fuente.** Los conjuntos incluidos están ahora en archivos JSON
(`tbkit/parameters/`), cada número con unidad, descripción y fuente, y cada
conjunto con su referencia, el sistema para el que se obtuvo y su validez. Un
ajuste informa de cada parámetro que cambia (antes → después) y de si el error
mejora o empeora.

**Qué NO hace, y dónde está la trampa:**

- El modelo π sigue sin parte repulsiva: no relaja ni da fonones, y lo dice.
- El modelo de Xu es carbono puro: no describe H, N ni O.
- La banda G tiene una anomalía de Kohn: su frecuencia depende de la malla de
  puntos k y converge despacio (usa ≥ 36×36 y compruébalo).
- Las fuerzas con Hubbard de campo medio no están implementadas.

## tbkit · tight binding, el cuarto paquete

Nuevo paquete independiente, `packages/tbkit`, con su propio comando `tbkit`.
Lee cualquier estructura que lea ASE (el extxyz de carbonforge o nanocarbon_lab
conserva la celda y la periodicidad).

**Qué calcula:** niveles y gap, bandas, DOS y PDOS (por elemento, átomo u
orbital), cargas Mulliken y Löwdin, órdenes de enlace, la matriz densidad, y
orbitales moleculares en archivos `.cube` para ver isosuperficies en VESTA o
Avogadro. Con **Hubbard de campo medio**, el magnetismo de bordes zigzag y la
magnetización de tres maneras: frente a la energía (m(E) y dm/dE = ρ↑ − ρ↓),
frente al campo (M(h) y la susceptibilidad χ) y frente al dopaje o el nivel de
Fermi. Con **cargas autoconsistentes**, la transferencia de carga de dopantes
apantallada.

**Modelos:** π (t = −2,7 eV, con N, B y O de Hückel), sp³ de carbono de
Xu–Wang–Chan–Ho, o tus propios parámetros: archivos `.skf` de DFTB (no se
incluyen: son de sus autores) o un ajuste a tus niveles de GPAW
(`tbkit gpaw-levels calc/gpaw.txt` los lee).

```bash
tbkit levels piridina.xyz --scc --bonds
tbkit hubbard zgnr.extxyz --U 2.7 --kmesh 48 --m-energy m.csv
tbkit hubbard copo.xyz --field 0 0.5 11 -o campo.csv
tbkit orbital coroneno.xyz --band homo -o homo.cube
```

**Qué NO hace, y dónde está la trampa:**

- No hay energías totales ni fuerzas (falta la parte repulsiva): estructura
  electrónica sí, relajar no.
- El campo medio rompe la simetría de espín para imitar la correlación: los
  momentos locales de un copo con M = 0 no son el estado fundamental real (un
  singlete). El teorema de Lieb (M = |N_A − N_B|) sí se cumple y está comprobado.
- Las cargas autoconsistentes solo funcionan en sistemas finitos (un periódico
  necesitaría Ewald; se rechaza).
- Los parámetros π de los heteroátomos son de libro (Hückel): tendencias, no
  niveles cuantitativos. Ajústalos a tu DFT.
- Sin interfaz gráfica todavía.

## vibspec · Raman

**Raman con GPAW, sobre el mismo cálculo del IR.** En Calcular → IR y Raman
(GPAW), el campo **Raman** ofrece:

- `field`: polarizabilidad DFT de cada geometría desplazada, por diferencias de
  dipolo bajo un campo eléctrico ±E. Es el método con fundamento físico, y el
  caro: 36 SCF por átomo además del IR. Solo en LCAO o FD (un campo uniforme no
  cabe en una celda periódica, así que en PW se rechaza).
- `bond`: modelo de polarizabilidad de enlaces de Lippincott–Stuttman (ASE).
  Segundos. Dice qué modos son activos en Raman, pero sus intensidades son solo
  orientativas: no ve conjugación, transferencia de carga ni el efecto
  electrónico de un dopante. Se avisa en la validación y en los resultados.

Cada polarizabilidad se guarda al calcularse: un cálculo interrumpido retoma.
Resultado por modo: actividad (Å⁴/amu) y razón de despolarización.

**Comparar con el experimento.** "Frente al experimento" (antes "IR frente a
FTIR") tiene un selector IR/Raman. El Raman experimental se usa como intensidad
(sin convertir a absorbancia); el calculado puede llevar los factores de láser y
Bose (ν_láser − ν)⁴ para parecerse a lo medido, o mostrarse como actividad. El
emparejado de bandas y el ajuste del factor de escala funcionan igual que en IR.
En terminal: `carbonforge vibspec plot calc/agua --kind raman --raman-exp mi_raman.txt`.

**QE conectado al modelo finito.** Un `dynmat.out` de QE (ph.x con `lraman`) se
compara igual que un cálculo de vibspec: botón "dynmat.out (QE)…" o `vibspec plot
dynmat.out`. Al exportar una estructura finita a QE, `dynmat.in` lleva ahora
`asr='zero-dim'` (seis modos rígidos, no tres).

**Qué NO hace, y dónde está la trampa:**

- `field` no se ha podido probar contra GPAW real aquí (no hay GPAW en este
  entorno): la aritmética, las unidades y la caché están comprobadas con un
  calculador de prueba que reproduce exactamente el modelo de enlaces. Antes de
  usarlo en una cinta, compáralo con una referencia publicada (benceno).
- No hay Raman resonante: si tu láser cae cerca de una transición del modelo,
  las intensidades medidas no serán las no resonantes.
- Los modos de un `dynmat.out` de QE se animan en Resultados → Bandas y
  espectros (con `dynmat.axsf`), no en esta página.

## carbonforge · Resultados completos y recetas en la cola

**Densidad de estados en la ventana.** Resultados → Bandas y espectros abre la
DOS total (`dos.dat`) y la proyectada (la carpeta de `projwfc.x`), con el nivel
de Fermi del archivo o el que escribas, el gap estimado y la completitud de la
proyección. "Abrir resultados" en Trabajos la abre directamente.

**Modos de QE.** Si junto a `dynmat.out` está `dynmat.axsf` (carbonforge ya lo
pide en `dynmat.in`), un clic en una banda del espectro abre una ventana con el
modo animado y el reparto del movimiento por elemento y por átomo, igual que en
vibspec (comparten el código).

**Recetas en la cola.** Los proyectos de receta llevan `job.json` con la cadena
completa: relajación → `carbonforge update-geometry` (la propiedad se calcula
sobre la geometría relajada) → propiedad, con los pools de puntos k (`-nk`) que
decidió la receta. Los pools solo se aplican si dividen al número de procesos;
si no, QE no arrancaría.

**Trampas:** el lector de `.axsf` está probado con archivos sintéticos del
formato documentado, no con una salida real de QE; si los modos no cuadran con
`dynmat.out` (otro cálculo), no se animan y se dice.

## carbonforge · una cola para todos los cálculos

**Calcular → Trabajos.** Todos los cálculos que se lanzan desde la ventana
(GPAW de vibspec, QE, SIESTA, LAMMPS) van a la misma cola: motor, estado,
progreso ("paso 2/3: ph.x"), log en vivo, **Cancelar** y **Abrir resultados**,
que abre las bandas o el espectro terminados en Resultados, o el IR de vibspec
en su página.

**Encolar al exportar.** En Preparar → Cálculo, marca "Encolar al exportar": las
carpetas exportadas entran en la cola si el programa está instalado en esta
máquina. Si falta (pw.x en Windows, por ejemplo) no se encola y se dice cómo
correrlo en otra.

**`job.json` y el runner.** Cada carpeta exportada (qe/, siesta/, lammps/) lleva
un `job.json` con sus pasos en orden (pw.x → ph.x → dynmat.x; pw.x scf → bands
→ bands.x; siesta < input.fdf; lmp -in in.lammps). En cualquier máquina:

```bash
python -m carbonforge.jobs.run salida/qe --nprocs 8     # --restart para empezar de cero
```

Cada paso se registra en `job.json`; si uno falla o se cancela, la siguiente
ejecución **retoma tras el último paso terminado** (no repite el scf). Otro
binario: `CARBONFORGE_PW_X=/opt/qe/bin/pw.x`; otro lanzador MPI:
`CARBONFORGE_MPI="srun -n {n}"`.

**Qué NO hace, y dónde está la trampa:**

- Un paso que se corta a medias (pw.x cancelado) se repite entero la próxima
  vez; lo que se retoma es la cadena, no el paso.
- Cancelar detiene también el programa (pw.x, siesta) en Linux y macOS. En
  Windows solo se detiene el runner; allí QE y SIESTA no suelen correr.
- ~~La DOS terminada aún no se grafica en la ventana~~ Ya se grafica (ver abajo).
- ~~Las recetas no llevan `job.json`~~ Ya lo llevan (ver abajo).
- Una cola, un trabajo a la vez: dos cálculos DFT en los mismos núcleos van
  más lentos que uno tras otro.

## carbonforge · una sola ventana, con vibspec dentro

**Secciones por tarea.** La ventana se organiza como se trabaja:
**Estructura** (Construir, Importar, Modelo finito (IR)), **Preparar** (Celda
EDLC), **Calcular** (IR con GPAW) y **Resultados** (Bandas y espectros, IR
frente a FTIR). vibspec ya no abre una ventana aparte: `carbonforge vibspec gui`
abre la principal en su página.

**Estructura actual.** La barra inferior dice con qué estructura se está
trabajando y de dónde salió. Lo que construyes, importas o modelas pasa a ser
la estructura actual; las otras páginas la toman **con un botón**, nunca solas:

- En Modelo finito (IR), la opción "Estructura actual" arranca el modelo desde
  ella (una cinta hecha o importada en Construir), con las mismas
  comprobaciones que un archivo.
- En Construir, "Traer la estructura actual" recupera un modelo de vibspec para
  exportarlo a QE o SIESTA, o seguir funcionalizándolo.

**Qué NO hace, y dónde está la trampa:**

- Una estructura periódica (un nanotubo, una cinta infinita) no entra en
  vibspec: el IR por diferencias finitas necesita un modelo finito. Se rechaza
  y se dice por qué; no se corta nada.
- ~~La exportación a QE/SIESTA sigue en la página Construir~~ Ya está en
  **Preparar → Cálculo (QE, SIESTA, LAMMPS)**, junto con la receta, los ajustes
  del cálculo, los parámetros avanzados y las correcciones; exporta la
  estructura actual, venga de donde venga. Construir se queda con la geometría,
  el dopaje y los grupos. La celda EDLC también parte de la estructura actual, y
  se descarta si esta cambia.
- ~~La cola de trabajos solo lanza cálculos de GPAW~~ Ahora hay una sola cola
  para todo (ver abajo).
- Cerrar la ventana con cálculos de vibspec en marcha pide confirmación.

## carbonforge · la ventana, partida por pestañas

Por dentro: `gui/app.py` pasaba de 1400 líneas; ahora cada pestaña vive en su
módulo (`gui/tabs/`) y la ventana solo guarda lo compartido. Por fuera no
cambia nada, salvo un arreglo: el resumen tras construir ya tiene en cuenta los
parámetros avanzados (antes solo lo hacían "Comprobar" y la exportación). Es el
paso previo a una ventana única con vibspec.

## carbonforge · todos los parámetros de QE, SIESTA y GPAW, comprobados

**Parámetros avanzados.** Un botón "Parámetros avanzados (QE, SIESTA)…" en la
pestaña de construcción, y "Parámetros avanzados de GPAW…" en la pestaña
Cálculo de vibspec, abren el catálogo del código: buscas una palabra clave, lees
qué hace, su tipo, su valor por defecto y sus opciones, y le das un valor. Se
comprueba al escribirlo (tipo, opciones válidas) y otra vez antes de exportar;
un valor inválido bloquea la exportación. Llega al archivo: al namelist
correcto de `pw.x`, como línea del `.fdf` de SIESTA (sustituyendo la que
carbonforge habría escrito) o como argumento de `GPAW(...)`.

**Importar los manuales.** De serie hay un núcleo curado (~50 palabras de QE,
~25 de SIESTA, ~17 de GPAW). Con "Importar manual…" se lee la documentación de
tu propia instalación: `INPUT_PW.def` de QE (en `PW/Doc/` de las fuentes, ~260
parámetros con sus opciones explicadas), `siesta.tex` de SIESTA (en `Docs/`,
~540 entradas) y, para GPAW, el propio paquete instalado. Se guarda en
`~/.carbonforge/catalogos/` (o `$CARBONFORGE_HOME/catalogos/`); se importa una
vez. Los manuales no se incluyen con carbonforge: son de sus proyectos.

**Qué NO hace, y dónde está la trampa:**

- Un nombre desconocido es **aviso** con el núcleo curado (no es exhaustivo) y
  **error** con el manual importado. Importa el manual para validar de verdad.
- Si sustituyes algo que carbonforge decide (`ecutwfc`, `nspin`,
  `MeshCutoff`...) se avisa: tu valor manda, y lo que se valida es tu valor
  (un `occupations='fixed'` en un metal sigue siendo un error).
- `&IONS` y `&CELL` solo existen en relax/vc-relax: un parámetro de esos
  namelists en un scf es un error, no se descarta en silencio.
- En vibspec, `symmetry`, `spinpol` y `txt` no se pueden sustituir: la simetría
  apagada es necesaria para las diferencias finitas.
- Las tarjetas de QE (`ATOMIC_POSITIONS`, `K_POINTS`...) y los bloques de
  SIESTA los sigue escribiendo carbonforge; aquí solo van palabras clave.
- El lector de `siesta.tex` interpreta LaTeX de forma aproximada: alguna
  descripción o valor por defecto puede salir incompleto.

## carbonforge · dónde va cada cosa, y avisos que se corrigen con un botón

**Colocación con control.** Dopantes y grupos funcionales se colocan por
**región**: borde, borde armchair, borde zigzag, plano basal, pentágono,
heptágono, defecto 5-7, vecino de un defecto o borde de vacante; o en átomos
concretos (`12, 30-32`). Se mantiene una distancia mínima entre ellos **y**
respecto a lo que ya hay: un grupo ya no cae junto a un dopante. En un borde
terminado en H, el grupo sustituye al H; un N de borde queda piridínico (sin
N-H). Los grupos basales van a la cara del plano que elijas. En la GUI y en la
terminal (`--dopant-region`, `--group-site`, `--group-indices`, `--group-face`,
`--group-avoid`).

**Arreglado por el camino:**

- El H de un -OH de borde se doblaba sobre la red (a 0,5 Å del carbono
  vecino). Ahora cada grupo gira sobre su enlace hasta quedar libre.
- Un C-H de borde contaba como "basal": el N grafítico aleatorio y los grupos
  basales podían caer en el borde. Tampoco se detectaban los bordes de una cinta
  pasivada al activar el espín antiferromagnético.
- `stone_wales_defect` dejaba los dos átomos del enlace rotado uno encima del
  otro.
- Una vacante podía quitar un hidrógeno.

**Correcciones con un botón.** El panel "Correcciones" de la pestaña de
construcción muestra cada problema con remedio conocido y lo aplica: espín
antiferromagnético en una cinta zigzag, solo frecuencias en un metal,
pseudopotenciales norm-conserving para Raman, `cell_dofree` en vc-relax, vdW en
espumas, ecutrho... El panel de cálculo tiene ahora espín, funcional, vdW,
ocupaciones, smearing, ecutrho, `cell_dofree` y familia de pseudopotenciales, y
**lo que se valida es lo que se escribe**: validación y exportación usan la
misma configuración.

**Dónde está la trampa:**

- Con dopantes y la separación mínima puede que no quepan todos los grupos
  pedidos: el programa lo dice y cuántos caben, en vez de amontonarlos.
- Espín "auto" activa el estado antiferromagnético solo en cintas zigzag
  periódicas. En otras estructuras con bordes zigzag, elígelo a mano.

---

## carbonforge · vibspec — tus propias geometrías y una biblioteca

**Cargar una cinta ya hecha**, con sus átomos y grupos: en la pestaña Modelo,
"Desde archivo", o `carbonforge vibspec import mi_cinta.xyz`. Vale cualquier
formato que lea ASE (XYZ de Avogadro o GaussView, CIF, PDB, MOL, POSCAR, salida
de QE). Si no trae celda se le pone caja con vacío; lo que ya tiene se respeta,
y se le puede añadir un preset encima.

**La biblioteca** es una carpeta (`estructuras/`) cuyos archivos aparecen en una
lista de la ventana. "A la biblioteca" guarda el modelo actual con su
procedencia, para precargarlo la próxima vez.

**Dónde está la trampa:**

- Una estructura periódica de verdad (con enlaces que cruzan la celda) se
  rechaza: el IR necesita un modelo finito, y cortarla sin terminar los bordes
  dejaría carbonos colgantes. Recórtala y termínala con H antes.
- Átomos superpuestos también se rechazan; no se mueven para "arreglarlos".
- El tipo de borde se deduce de la geometría solo si uno domina claramente; en
  una cinta casi cuadrada no pasa, y hay que elegir el borde del sitio.

---

## carbonforge · vibspec (fase 4)

### La ventana: modelo, cola de cálculos y modos animados

```bash
carbonforge vibspec gui
```

Tres pestañas sobre el mismo núcleo que la línea de comandos:

- **Modelo:** eliges cinta, funcionalización y sitio, ves la estructura en 3D
  (de frente, mirando por la normal del plano) y todos los avisos, incluido si
  hace falta espín.
- **Cálculo:** los parámetros de GPAW se validan antes de escribir nada.
  *Preparar* deja el directorio listo; *Preparar y correr* además lo pone en la
  cola. Cada trabajo es un proceso aparte, así que *Cancelar* lo para de
  verdad, y `run.py` lo retoma después. Ves el estado, el progreso (paso de
  relajación, desplazamientos hechos de cuántos) y el log en vivo.
- **Resultados:** el espectro contra tu FTIR, la tabla de bandas, el ajuste del
  factor de escala y, **al hacer clic en una banda, la animación de su modo
  normal** con el reparto del movimiento por elemento y por átomo.

**Dónde está la trampa:**

- En Windows, sin GPAW, los botones de correr aparecen desactivados y la ventana
  dice por qué: se prepara ahí, se corre en Ubuntu y el resultado se abre en
  Resultados.
- Un trabajo que termina con código 0 pero sin `record.json` en estado `done`
  cuenta como error: la ventana se fía del registro, no solo del proceso.
- Una sola tarea a la vez por defecto: dos GPAW en los mismos núcleos tardan más
  que uno detrás de otro.

---

## carbonforge · vibspec (fase 3)

### Tu FTIR contra el cálculo, con tabla de asignación

```bash
carbonforge vibspec plot calculos/amina --ftir mi_ftir.csv --fit-scale -o amina.png
```

Lee el FTIR tal como lo exporta el equipo: separado por tabuladores, punto y
coma, comas o espacios, con coma decimal o punto, en absorbancia o
transmitancia. Lo pasa a absorbancia, le quita la línea base (envolvente
convexa inferior), ensancha el calculado (Lorentziano o Gaussiano, con la FWHM
que digas) y los dibuja normalizados sobre un único eje, con el número de onda
decreciente como en el espectrómetro. Además imprime una tabla de qué modo cae
en qué banda y exporta las curvas a CSV para darles estilo en ramancarbon.

**Dónde está la trampa:**

- El factor de escala no se ajusta emparejando primero: sin escalar, las
  frecuencias DFT suelen caer más lejos que cualquier tolerancia razonable (el
  agua en PBE: 3698 frente a 3756 cm⁻¹), y no habría nada que ajustar.
  `--fit-scale` busca el factor en [0,90, 1,05] que lleva más intensidad a
  bandas y luego lo refina por mínimos cuadrados. Con menos de tres parejas se
  niega.
- Si el archivo no dice si es absorbancia o transmitancia, se deduce de los
  valores y **se avisa de que es una deducción**. `--quantity` lo zanja.
- La tabla es una propuesta. Dos modos pueden caer en la misma banda, y una
  banda puede ser un sobretono que el cálculo armónico no tiene. Mira el modo
  antes de dar la banda por asignada.

---

## carbonforge · vibspec (fase 2)

### El cálculo IR con GPAW, de principio a fin

`carbonforge vibspec prepare` valida la estructura y los parámetros y escribe
un directorio autocontenido; `run.py` dentro de él relaja, vuelve a comprobar
la estructura relajada y calcula el IR por diferencias finitas del dipolo
(`ase.vibrations.Infrared`). Se prepara en Windows y se corre en Ubuntu:

```bash
carbonforge vibspec prepare amina.xyz -d calculos/amina
cd calculos/amina && mpiexec -n 4 gpaw python run.py
carbonforge vibspec show calculos/
```

Cada cálculo guarda en `record.json` los parámetros, los chequeos, las
versiones (con el commit de git), la relajación, los resultados y un historial
de estados con hora: `prepared → relaxing → relaxed → vibrations → done`, o
`error` con el motivo. `vibspec index` lo vuelca todo a una base de datos ASE.

**Qué NO hace todavía:** Raman, el ensanchado del espectro y la comparación con
tu FTIR (fase 3).

**Dónde está la trampa:**

- LCAO (por defecto) y FD usan condiciones de contorno cero. El modo PW también
  corre, pero resuelve el potencial de Hartree como periódico aunque la celda no
  lo sea: un grupo polar nota a sus imágenes. Avisa y pide 8 Å de vacío por lado.
- Los seis modos de sólido rígido se quitan del espectro y se informan. Unas
  decenas de cm⁻¹ son normales (fuerza residual y paso finito); por encima de
  100 cm⁻¹ avisa: relajación insuficiente o efecto huevera. Contra la huevera
  se aplica por defecto la corrección de Frederiksen (las fuerzas de cada
  desplazamiento suman cero): en N₂ con LCAO lleva las traslaciones de
  150–270 cm⁻¹ a cero y mueve la tensión menos de un 1 %. Las rotaciones siguen
  notando la rejilla (~80 cm⁻¹ en N₂); en una cinta, con su momento de inercia
  mucho mayor, pesan menos.
- Deja algo de margen de vacío: la relajación mueve los átomos del borde. Al
  preparar se exige 6 Å por lado y se avisa por debajo de 6,5.
- Coste: el agua, en serie, ~10 min. Una cinta de 80 átomos son ~480 SCF.

---

## carbonforge · vibspec (fase 1)

### Cintas finitas con una funcionalidad, para asignar bandas de FTIR

`carbonforge vibspec build` construye una nanocinta **finita**, terminada en H
por todos los bordes y con 7 Å de vacío por lado, y le pone una funcionalidad
en un sitio reproducible: N grafítico, piridínico de borde, piridínico en
vacante (N3V), precursor pirrólico, amina, nitrilo, N-óxido piridínico,
hidroxilo, carboxilo, carbonilo o epóxido. Los grupos de borde sustituyen un H;
el sitio por defecto es el centro de un borde largo, lejos de las esquinas.

Finita y no periódica por una razón: el IR se calcula derivando el momento
dipolar, y una cinta periódica no tiene dipolo definido a lo largo de su eje.

**Qué comprueba antes de dejarte seguir** (`carbonforge vibspec check`):
vacío por lado, que no sea periódica, número de electrones y si hace falta
espín, con momentos iniciales antiferromagnéticos para los bordes zigzag. Para
la fase de vibraciones deja preparados los rechazos: estructura sin relajar,
fmax por encima de 0,05 eV/Å, y cálculo sin espín de un sistema de capa abierta.

**Dónde está la trampa:**

- Tres presets dejan un número impar de electrones (grafítico, N3V y carbonilo
  aislado). Son de capa abierta de verdad, no es un fallo del builder.
- Una cinta armchair termina en bordes zigzag de 4 sitios. Se pide cálculo con
  espín; si los momentos se van a cero, el sistema era de capa cerrada.
- El pirrólico sigue siendo un precursor: el pentágono solo aparece al relajar,
  y el chequeo previo a las vibraciones lo exige.

Todavía no calcula nada: el flujo IR con GPAW es la fase 2.

---

## Raman · biblioteca de carbono

### Fases de precursor: N, P, B, Cl, S

El catálogo sabía en qué se convierte una muestra y no sabía nada de con qué
se hizo. Ahora trae 36 fases, e incluyen lo que entra en el horno: **melamina,
urea, g-C₃N₄, fósforo rojo, ácido bórico, B₂O₃ vítreo, BN hexagonal, azufre
S₈, sulfato sódico y cloruro de amonio**, además de los óxidos de manganeso
(Mn₃O₄, Mn₂O₃, β-MnO₂, birnesita) que faltaban junto a los de hierro, cobalto
y níquel.

Están ahí por una razón concreta: **tres de ellos caen encima de la región del
carbono y se disfrazan de lo que buscas.**

| Fase | Dónde muerde | Cómo se separa |
|---|---|---|
| h-BN | UNA banda a 1366 cm⁻¹, dentro del rango de la D | Por ANCHURA: 10 cm⁻¹ el cristal, 50–150 un carbono desordenado. La posición no los separará nunca |
| g-C₃N₄ | Bandas a 1233, 1310 y 1570: encima de la D y la G | Por las de 707 y 750 cm⁻¹, donde el carbono no tiene nada |
| NH₄Cl | La ν₄ a 1400 cm⁻¹ | Por la tensión N–H a 3050. Hay que medir hasta 3200 |

El caso del g-C₃N₄ es el que más te puede costar: un «carbono dopado con
nitrógeno» hecho desde melamina que en realidad es nitruro de carbono da un
espectro con pinta de carbono desordenado y una I_D/I_G **que no significa
nada**, porque esas bandas no son de un carbono. Y si tu medida empieza en
1000 cm⁻¹ no puedes ni confirmarlo ni descartarlo: el programa ahora lo dice
con esas palabras en vez de darlo por bueno con las bandas compartidas.

Sobre el cloro, un aviso que conviene leer: **casi todos los cloruros son mudos
en Raman.** NaCl, KCl, FeCl₂, FeCl₃ no dan primer orden útil. Que el Raman no
encuentre cloro no prueba que no lo haya — para eso, EDS o XPS Cl 2p.

### Dopaje

Ya estaba bien cubierto (N grafítico/piridínico/pirrólico, B, S, O, P, Se, con
la corrección de tamaño frente a carga). Se han añadido cuatro entradas:

- **Cl covalente**, con el aviso de arriba.
- **Codopado N+S** y **codopado N+P**. Estos dos son un aviso, no una medida:
  el nitrógeno grafítico ENDURECE la G por transferencia de carga y el azufre
  o el fósforo la ABLANDAN por tamaño de enlace, con el mismo orden de
  magnitud. Un ΔG pequeño en una muestra codopada es compatible con **mucho**
  dopante, no con poco. El programa no descompone eso, porque con un solo
  espectro no se puede.
- **Contacto con nanopartícula metálica** (Fe, Co, Ni, FeSe). Firma: dopado
  tipo p **sin** aumento de I_D/I_G, porque el carbono no gana defectos, solo
  carga. Eso lo separa de un dopante sustitucional. Ojo también con el SERS: si
  las intensidades se disparan y las posiciones no se mueven, es la partícula.

### Tres falsos positivos corregidos

Escribir las fases nuevas destapó tres, dos recién metidos y uno antiguo:

1. **Azufre en un nanotubo limpio.** Las líneas del S₈ a 153 y 219 cm⁻¹ están
   dentro de la ventana RBM. Listarlas como fuertes junto a la de 473 hacía que
   dos de tres bastaran, y un espectro de nanotubo perfectamente normal
   reportaba azufre. Ahora **solo la de 473 identifica azufre**.
2. **Magnetita reportada como MnO₂.** La Fe₃O₄ está en 668/540/310 y el
   β-MnO₂ en 665/535: cada línea del óxido de manganeso cae dentro de
   tolerancia de una del hierro. Las dos coinciden y las dos se corroboran.
   Ahora la resolución de conflictos también descarta un candidato
   **corroborado**, pero solo si otra fase de familia distinta explica los
   mismos picos con estrictamente más líneas Y menor desviación media. Entre
   polimorfos de una misma familia sigue apagada: ahí informar los dos y dejar
   que hable el veredicto de familia **es** la respuesta.
3. **Cuarzo en cualquier muestra con MoO₃.** El α-MoO₃ tenía catalogadas solo
   sus cuatro bandas principales, así que el programa no podía saber que ya
   explicaba los picos de 471, 217 y 158. Ahora tiene sus catorce. Y el cuarzo,
   que se corroboraba con la de 464 sola más una coincidencia, exige 464 **y**
   206.

Y una regla que estaba mal en general: cuando **ninguna** línea fuerte de una
fase entra en el rango medido, el programa la juzgaba por las menores. Así es
como un carbono medido desde 1000 cm⁻¹ corroboraba g-C₃N₄. Ahora esa fase se
informa como pista, con un aviso que nombra la región que habría que medir: ni
confirmada ni descartada, que es la verdad.

**Resultado:** cada espectro de demostración informa exactamente las fases que
contiene y ninguna más.

---

## Raman · dicalcogenuros (TMD)

### La biblioteca: de 5 materiales a 15

Cubre ahora los calcogenuros de **S, Se y Te de Mo, W, Ti, Nb, Ta y Fe**, con
sus óxidos.

| Nuevos | Politipo | Confianza |
|---|---|---|
| WTe₂ | Td (no tiene 2H) | media |
| TiS₂, TiSe₂ | 1T octaédrico | alta |
| NbSe₂ | 2H, metal con CDW | media |
| TaS₂ (2H **y** 1T) | prismático / con CDW | media |
| TaSe₂ | 2H, metal con CDW | media |
| Pirita FeS₂ | cúbico, **sin capas** | alta |
| β-FeSe tetragonal | laminar, no 2H | media |
| Marcasita FeSe₂ | ortorrómbico, **sin capas** | baja |

Óxidos nuevos: **TiO₂ anatasa y rutilo, Ta₂O₅ (cristalino y amorfo), Nb₂O₅,
Fe₂O₃, Fe₃O₄, FeOOH y SeO₂**, con las rutas de oxidación de cada calcogenuro,
que es lo que evita encontrar óxidos de wolframio en una muestra de molibdeno.
Tu ejemplo, MoS₂@MoO₂@MoO₃ y su análogo de selenio, está cubierto, y ahora
también TiO₂@TiS₂, Fe₂O₃@FeSe y Nb₂O₅@NbSe₂.

**No todos son 2H ni todos son laminares**, y eso importa: el conteo de capas
por separación E₂g–A₁g solo vale para los 2H de Mo y W, y de verdad solo para
el MoS₂. En un 1T, en el WTe₂ o en la pirita el programa **dice que la pregunta
no tiene respuesta** en vez de dar un número.

### Lo que NO está, y por qué

`ramancarbon tmd --listar` termina con esa lista: NbS₂, NbTe₂, TaTe₂, TiTe₂,
los calcogenuros de manganeso y el FeTe. El criterio para entrar es **tener una
fuente citable**. Una posición inventada no avisa: identifica mal, y con
seguridad aparente. (El manganeso está cubierto por sus óxidos, que es donde de
verdad acaba un catalizador de manganeso.)

### Identificación: dos reglas nuevas

Con quince materiales las ventanas se solapan. El A₁g del MoSe₂ está en 240 y
el del TaSe₂ en 234; el A₁g del TiSe₂ (198) cae encima del B₁g del β-FeSe
(196); la banda ancha de dos fonones del NbSe₂ se traga media ventana RBM.

1. **Hace falta un modo discriminante.** Una suma de coincidencias en bandas
   compartidas no identifica. Es la misma regla que ya usaban las líneas
   exclusivas de los óxidos y las discriminantes de las fases.
2. **Un pico observado lo reclama un solo modo.** En el NbSe₂ el A₁g y el E₂g
   están a 6 cm⁻¹, más cerca de lo que mide cualquiera de las dos ventanas, y
   una sola banda estaba puntuando dos veces. Eso solo bastaba para que un
   espectro de MoTe₂ puntuara más como NbSe₂ que como MoTe₂.

Con una excepción honesta: **una banda que no has medido no es evidencia en
contra.** Los modos discriminantes del 1T-TaS₂ están en 63 y 75 cm⁻¹ y casi
ninguna medida llega ahí; en ese caso la regla se levanta y el informe dice en
voz alta en qué se está apoyando.

### La pestaña de TMD

Tenía una columna y diez botones, frente a los cuarenta y cinco de la de
carbono. Ahora tiene seis pestañas:

- **Espectro** — el ajuste y el informe del calcogenuro.
- **Modos** — tabla con posición de catálogo, posición ajustada, Δ, FWHM y
  altura; y una gráfica de piruleta, porque lo que importa es el **tamaño y el
  signo** de cada desplazamiento y comparar dos curvas superpuestas a ojo es
  malo justo para eso.
- **Óxidos** — el espectro con las líneas de catálogo de los óxidos que ese
  calcogenuro **puede** dar (trazo continuo = exclusivas), y el informe de
  intercara.
- **Otras fases** — el catálogo general de fases sobre el mismo espectro, sin
  restringir al metal del calcogenuro. Aquí es donde aparece el **selenio sin
  reaccionar a 237 cm⁻¹ y el azufre a 473**: la búsqueda de óxidos no puede
  verlos por construcción, y un análisis elemental que dé la estequiometría
  correcta no distingue el selenio de la red del selenio segregado. Esto sí.
- **Lote** — la tabla, ahora exportable a CSV.
- **Biblioteca** — el catálogo entero: modos, fuentes, confianza, modos
  discriminantes marcados, y la lista de ausentes con su motivo.

Además tiene **sus propios controles de preprocesado**, que editan los mismos
ajustes que la sección de carbono — no una copia. Con un aviso propio de esta
física: **no suavices para contar capas.** Las bandas miden 2–6 cm⁻¹ y las
fronteras entre números de capa están a 2–3 cm⁻¹.

---

## Lo que sigue igual, a propósito

- El índice de oxidación es un **cociente de intensidades**, no una fracción en
  masa. La banda de 819 cm⁻¹ del α-MoO₃ es de las más intensas de la química
  inorgánica y los modos de un dicalcogenuro fuera de resonancia no lo son.
- **Raman no ve topología.** Un MoO₃@MoSe₂ y una mezcla de polvos dan el mismo
  espectro puntual. Lo que se mide es la deformación del calcogenuro, que una
  intercara íntima produce y una mezcla no. Es evidencia, no prueba.
- El escaneo de fases sigue **encendido por defecto** en la sección de carbono,
  y el de interferencias **apagado**.
