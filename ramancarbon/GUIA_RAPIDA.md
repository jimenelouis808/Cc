# Guía rápida

Para usar el programa sin saber Python. Si algo no sale, en cada paso hay
una nota de qué mirar.

---

## 1. Instalar

Necesitas **Python 3.10 o más nuevo**. Compruébalo abriendo una terminal
(en Windows: *Símbolo del sistema*) y escribiendo:

```
python --version
```

Si dice 3.10, 3.11, 3.12… vas bien. Si dice 2.7 o «no se reconoce el
comando», instala Python desde [python.org](https://www.python.org/downloads/)
y **marca la casilla «Add Python to PATH»** durante la instalación. En
Windows marca también **«tcl/tk and IDLE»**, que es lo que dibuja la ventana.

Luego, dentro de la carpeta `ramancarbon`:

* **Linux / macOS**: `./install.sh`
* **Windows**: doble clic en `install.bat`

El instalador crea un entorno aislado, instala todo, comprueba que Tkinter
está y ejecuta las pruebas. Si termina diciendo `OK`, ya está.

> **Linux y falta Tkinter**: `sudo apt install python3-tk` (Ubuntu/Debian),
> `sudo dnf install python3-tkinter` (Fedora) o `sudo pacman -S tk` (Arch).
> Sin Tkinter todo lo demás funciona; solo te falta la ventana.

Cada vez que abras una terminal nueva, activa el entorno antes de usar el
programa:

```
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

---

## 2. Probarlo sin tener datos

```
ramancarbon demo datos_prueba/
```

Eso escribe seis espectros **sintéticos** (SWCNT, SWCNT metálico, DWCNT,
MWCNT, grafeno y óxido de grafeno) en la carpeta `datos_prueba/`. No son
medidas reales: sirven para ver cómo funciona todo antes de meter tus datos.

---

## 3. La interfaz gráfica

```
ramancarbon-gui
```

La ventana tiene la lista de espectros a la izquierda y seis pestañas:

| Pestaña | Para qué |
|---------|----------|
| **Espectro** | Cargar, preprocesar y ver qué ha hecho el preprocesado |
| **Deconvolución** | Montar el modelo de bandas, ajustarlo, mirar el residuo y exportarlo |
| **Informe** | El análisis completo por escrito |
| **Índices** | Γ_G, R1, R2, fracción amorfa, etapa de amorfización |
| **Diámetros** | La región RBM y los diámetros que implica |
| **Multiláser** | Combinar el mismo material medido a 532 y 633 nm |
| **Comparación** | La tabla de todo el lote, su estadística, y exportarla a CSV |
| **Base de datos** | Qué cree el programa y de dónde lo ha sacado |

### El flujo normal

1. **Abrir…** (o **Carpeta…** para cargar muchos de golpe). Si no tienes
   datos todavía, pulsa **Ejemplo**.
2. **Comprueba el láser** arriba a la derecha. Si el archivo no lo dice, el
   programa lo avisa. Escríbelo y pulsa **Aplicar** (o **Aplicar a todos**).
   Esto no es un detalle: sin el láser no se corrigen las posiciones por
   dispersión, no se puede calcular el tamaño de cristalito, y los
   desplazamientos de D y 2D no significan nada.
3. En **Espectro**, pulsa **Elegir parámetros automáticamente**. Rellena los
   campos con valores deducidos del espectro y explica cada decisión en el
   recuadro de abajo. Puedes cambiar cualquiera a mano después: lo que
   toques manda sobre lo automático. Luego **Aplicar preprocesado**.

   Verás el espectro original en gris detrás y la línea base propuesta
   encima, para comprobar que no se está comiendo ninguna banda.
4. En **Deconvolución**, elige el **perfil**. Para espectros reales,
   ruidosos, pseudo-Voigt es la opción sensata: contiene la gaussiana y la
   lorentziana como casos particulares, así que el ajuste mide la forma en
   vez de que tú la impongas, y te devuelve η.

5. Pulsa **Analizar** (o **Analizar todos** para el lote). El programa salta
   solo a la pestaña **Informe**.

6. Para sacar los datos: **Exportar todo…** en la pestaña *Informe* escribe
   el informe, la tabla de componentes, las curvas de la deconvolución
   (una columna por banda, para redibujar en Origin), el espectro procesado
   y un JSON completo. En *Deconvolución* hay un botón que exporta solo el
   ajuste.

### Si mides a dos láseres

Nombra los archivos con la longitud de onda: `muestra_532nm.txt` y
`muestra_633nm.txt`. Analiza los dos, ve a **Multiláser** y pulsa
**Combinar excitaciones**. Eso te da tres cosas que con un solo láser no
puedes tener:

* si una banda **no se desplaza**, no es de doble resonancia — una línea
  fija cerca de 1332 cm⁻¹ es diamante, no la banda D;
* la **dispersión de la banda G**, que es cero en material grafítico y
  6–10 cm⁻¹/eV cuando hay carbono amorfo;
* la comprobación de que las dos medidas dan el **mismo tamaño de
  cristalito**, ya que I_D/I_G escala como λ⁴ (a 633 nm sale ~2 veces el de
  532 nm en la misma muestra).

### Si tienes una muestra de control

Marca la casilla **«Usar como control (referencia)»** en el espectro sin
tratar. Los demás se compararán contra él en vez de contra valores de la
literatura. Esto es **mucho** mejor: elimina la deriva del equipo, que es
del mismo tamaño que los desplazamientos que quieres medir.

---

## 4. La línea de comandos

Más cómoda para lotes grandes y para meterla en un script.

```bash
# Un espectro, informe por pantalla
ramancarbon analizar muestra.txt --laser 532

# Guardar el informe y la figura resumen
ramancarbon analizar muestra.txt --laser 532 \
    --salida informe.txt --figura figura.png

# Comparando contra un control
ramancarbon analizar dopado.txt --laser 532 --control pristino.txt

# Una carpeta entera a una tabla CSV
ramancarbon lote datos/ --laser 633 --csv resultados.csv

# Solo la deconvolución, comparando 2, 3, 4 y 5 bandas
ramancarbon deconvolucionar muestra.txt --comparar

# Consultar la base de datos
ramancarbon bd                          # todas las bandas
ramancarbon bd --banda 2D --laser 785   # una banda, corregida a tu láser
ramancarbon bd --rbm                    # relaciones RBM ↔ diámetro
ramancarbon bd --materiales             # huellas de referencia
```

Con `--base height` los cocientes se calculan con alturas de pico en vez de
áreas integradas. **No mezcles las dos entre muestras**: para el mismo
espectro, un I_D/I_G de áreas es 2–3 veces el de alturas.

---

## 5. Qué formatos lee

Cualquier archivo de texto con dos columnas numéricas: `.txt`, `.csv`,
`.dat`, `.asc`, `.tsv`, `.prn`, `.xy`. El programa detecta solo:

* el separador (tabulador, coma, punto y coma, espacios);
* la **coma decimal** (`1580,25` funciona);
* las líneas de cabecera, y busca en ellas la longitud de onda del láser en
  varios idiomas;
* el sentido del eje (descendente se ordena solo).

**No lee formatos binarios de fabricante** (Renishaw `.wxd`, Thermo `.spa`,
Bruker `.opus`). Expórtalos como ASCII desde el programa del equipo.

---

## 6. Cómo leer el informe

El informe tiene siete secciones. Las que más se malinterpretan:

**Identificación.** Fíjate en la *confianza* y en las **«Reglas que no se han
podido aplicar»**. Si dice que no se ha podido comprobar el RBM porque tu
espectro empieza en 400 cm⁻¹, la identificación se está apoyando solo en la
forma de la banda G y es mucho más débil. Vuelve a medir desde ~100 cm⁻¹.

**Deconvolución.** Mira los avisos. «Parámetros casi degenerados» significa
que dos componentes se solapan tanto que sus áreas individuales no están
determinadas por separado: el ajuste no está mal, pero no cites esas áreas
con tres cifras. «Pegado a su límite» significa que el modelo no encaja y
hay que revisarlo.

**Cocientes.** Cada uno sale en las dos bases. `L_a` y `n_D` dependen de
λ⁴, así que I_D/I_G de dos láseres distintos **no** son comparables sin
corregir. Y ojo con la **rama**: si la banda G es ancha, estás en la zona de
amorfización, donde un I_D/I_G *menor* significa *más* desorden, no menos.

**Índices.** Si tus muestras están muy desordenadas — MWCNT, nanofibras,
dopado fuerte — mira **Γ_G** antes que I_D/I_G. Es la anchura de la banda G,
y a diferencia del cociente crece de forma monótona con el desorden en todo
el rango, así que no tiene la ambigüedad de las dos ramas. La sección
también te dice en qué **etapa de amorfización** estás, que es lo que
resuelve esa ambigüedad.

**Diámetros.** Si el RBM y el desdoblamiento G no coinciden, el informe lo
dice con un ✗ y explica qué revisar. No promedies: uno de los dos está mal.

**Desplazamientos.** Antes de interpretar un desplazamiento de pocos cm⁻¹:

1. ¿Calibraste el equipo ese día con la línea de 520.7 cm⁻¹ del silicio?
2. ¿Se mueve la banda al bajar la potencia del láser? Si sí, era
   calentamiento, no química.
3. ¿Tienes la banda 2D? Sin ella no puedes saber el signo del dopado: la G
   sube tanto con electrones como con huecos.

---

## 7. Antes de creerte un número

El programa comprueba solo la calidad de la medida y te avisa. Dos fallos
que **no se ven** en el informe:

* **Detector saturado.** Si el 3 % de los puntos está pegado al máximo,
  I_D/I_G puede salir el doble de lo que es. Baja el tiempo de integración.
* **Muestreo grueso.** Con menos de 6 puntos por anchura de banda el ajuste
  no puede fijar la anchura. A 16 cm⁻¹/punto, un I_D/I_G real de 1.0 salió 3.9.

Si tu muestra está sobre silicio, usa el botón **Calibrar con la línea de
Si**: el fonón del silicio está en 520.7 cm⁻¹ exactos y su desviación es el
error de tu equipo ese día.

Para un número que vas a publicar, pulsa **Incertidumbres por remuestreo**
en la pestaña *Deconvolución*. Las barras de error del ajuste salen cortas
—en la posición de G, cinco veces cortas—.

## 8. Problemas frecuentes

**«no se ha encontrado la longitud de onda del láser»** — el archivo no la
lleva en la cabecera. Escríbela en la casilla de arriba y pulsa *Aplicar*.

**«el espectro no cubre la región RBM»** — tu medida empieza demasiado
arriba. Sin la zona de 100–350 cm⁻¹ no se puede distinguir pared simple o
doble de multipared, ni sacar diámetros por RBM.

**La línea base se come el valle entre D y G** — sube la *Rigidez (λ)* en la
pestaña *Espectro*, o pulsa *Elegir parámetros automáticamente*, que la
calcula a partir de la anchura de tus bandas. Ojo: λ depende del **paso de
muestreo**, así que un valor copiado de un artículo casi nunca sirve — entre
1 y 4 cm⁻¹ por punto hay un factor 256.

**Mucha fluorescencia** — el modo automático avisa si el fondo varía tan
rápido como las bandas. En ese caso ninguna línea base los separa bien:
recorta el extremo de bajo desplazamiento, cambia de láser (a 633 nm suele
haber mucha menos fluorescencia que a 532), o fotoblanquea la muestra antes
de medir.

**Normalicé a 0–100 y los cocientes cambiaron** — es esperable y el programa
lo avisa. A diferencia de las demás, esa normalización resta un
desplazamiento además de escalar, y un desplazamiento no se cancela en un
cociente. Resta la línea base primero.

**El ajuste no converge o sale raro** — normalmente sobran componentes.
Pulsa *Comparar modelos* y usa el que gane. Si el material es un nanotubo,
comprueba que el modelo elegido incluye G⁻: sin ella, la banda D se estira
para absorber esa intensidad y I_D/I_G sale varias veces demasiado grande.

**Creo que hay óxido de catalizador o azufre sin reaccionar** — marca
*Buscar bandas no carbonosas* en la barra lateral. Está apagado por defecto
porque decirle al buscador lo que podría encontrar sesga lo que informa. Solo
acepta una especie si aparecen varias de sus líneas fuertes.

**Sale un pico donde no hay nada** — el detector exige una significancia
alta precisamente para no inventar picos en el ruido, pero si tu espectro
tiene mucha ondulación de fondo, prueba a restar la línea base antes.

---

## 8. Recuerda

Todo lo que hace este programa está validado contra **espectros
sintéticos**, generados por él mismo. Sirve para asegurar que las fórmulas
están bien implementadas y que la lógica no se contradice. No sustituye a
contrastar con tus propios patrones antes de publicar un número.
