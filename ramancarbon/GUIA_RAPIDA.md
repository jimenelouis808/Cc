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
| **Deconvolución** | Montar el modelo de bandas, ajustarlo y mirar el residuo |
| **Informe** | El análisis completo por escrito |
| **Diámetros** | La región RBM y los diámetros que implica |
| **Comparación** | La tabla de todo el lote, y exportarla a CSV |
| **Base de datos** | Qué cree el programa y de dónde lo ha sacado |

### El flujo normal

1. **Abrir…** (o **Carpeta…** para cargar muchos de golpe). Si no tienes
   datos todavía, pulsa **Ejemplo**.
2. **Comprueba el láser** arriba a la derecha. Si el archivo no lo dice, el
   programa lo avisa. Escríbelo y pulsa **Aplicar** (o **Aplicar a todos**).
   Esto no es un detalle: sin el láser no se corrigen las posiciones por
   dispersión, no se puede calcular el tamaño de cristalito, y los
   desplazamientos de D y 2D no significan nada.
3. En **Espectro**, ajusta el preprocesado si hace falta y pulsa **Aplicar
   preprocesado**. Verás el espectro original en gris detrás y la línea base
   propuesta encima, para que puedas comprobar que no se está comiendo
   ninguna banda.
4. Pulsa **Analizar** (o **Analizar todos** para el lote). El programa salta
   solo a la pestaña **Informe**.

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

**Diámetros.** Si el RBM y el desdoblamiento G no coinciden, el informe lo
dice con un ✗ y explica qué revisar. No promedies: uno de los dos está mal.

**Desplazamientos.** Antes de interpretar un desplazamiento de pocos cm⁻¹:

1. ¿Calibraste el equipo ese día con la línea de 520.7 cm⁻¹ del silicio?
2. ¿Se mueve la banda al bajar la potencia del láser? Si sí, era
   calentamiento, no química.
3. ¿Tienes la banda 2D? Sin ella no puedes saber el signo del dopado: la G
   sube tanto con electrones como con huecos.

---

## 7. Problemas frecuentes

**«no se ha encontrado la longitud de onda del láser»** — el archivo no la
lleva en la cabecera. Escríbela en la casilla de arriba y pulsa *Aplicar*.

**«el espectro no cubre la región RBM»** — tu medida empieza demasiado
arriba. Sin la zona de 100–350 cm⁻¹ no se puede distinguir pared simple o
doble de multipared, ni sacar diámetros por RBM.

**La línea base se come el valle entre D y G** — sube la *Rigidez (λ)* en la
pestaña *Espectro*. De `1e7` a `1e8`, por ejemplo.

**El ajuste no converge o sale raro** — normalmente sobran componentes.
Pulsa *Comparar modelos* y usa el que gane. Si el material es un nanotubo,
comprueba que el modelo elegido incluye G⁻: sin ella, la banda D se estira
para absorber esa intensidad y I_D/I_G sale varias veces demasiado grande.

**Sale un pico donde no hay nada** — el detector exige una significancia
alta precisamente para no inventar picos en el ruido, pero si tu espectro
tiene mucha ondulación de fondo, prueba a restar la línea base antes.

---

## 8. Recuerda

Todo lo que hace este programa está validado contra **espectros
sintéticos**, generados por él mismo. Sirve para asegurar que las fórmulas
están bien implementadas y que la lógica no se contradice. No sustituye a
contrastar con tus propios patrones antes de publicar un número.
