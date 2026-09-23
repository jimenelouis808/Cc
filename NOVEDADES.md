# Novedades

Lo que ha cambiado en esta ronda, escrito para quien va a usarlo y no para
quien lo ha escrito. Cada apartado dice qué hace ahora, qué NO hace, y dónde
está la trampa.

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
