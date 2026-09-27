# Novedades

Lo que ha cambiado en esta ronda, escrito para quien va a usarlo y no para
quien lo ha escrito. Cada apartado dice qué hace ahora, qué NO hace, y dónde
está la trampa.

---

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
