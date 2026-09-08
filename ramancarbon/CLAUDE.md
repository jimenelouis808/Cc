# CLAUDE.md

Instrucciones para Claude Code (o cualquier asistente) que trabaje en este
proyecto.

## Alcance

`ramancarbon` **analiza espectros Raman experimentales** de nanomateriales de
carbono: identifica SWCNT/DWCNT/MWCNT y material tipo grafeno, deconvoluciona
las bandas D y G, calcula cocientes de intensidad, deduce diámetros del RBM y
compara posiciones con la literatura.

No confundir con `carbonforge`, el proyecto hermano en este mismo repositorio:
aquél **prepara cálculos** de primeros principios (Quantum ESPRESSO, SIESTA,
LAMMPS). Éste analiza medidas. Mantén la distinción en nombres y documentación
y no muevas funcionalidad de uno a otro.

La interfaz de usuario (GUI, CLI, informes, mensajes de error) está **en
español**. El código, los nombres de identificadores y los docstrings están en
inglés. No mezcles.

## Estructura

```
ramancarbon/
├── core/        # Spectrum, lectura de archivos, línea base, despiking, picos
├── models/      # perfiles (incl. Breit-Wigner-Fano) y motor de ajuste
├── database/    # JSON de literatura + API tipada; NO importa del resto
├── analysis/    # asignación, cocientes, índices, diámetros, desplazamientos,
│                #   clasificador, multiláser, exportación
├── xrd/         # difracción: CIF, simetría, patrón calculado, Rietveld,
│                #   microestructura (Williamson-Hall…), Le Bail y Pawley
├── echem/       # CV, GCD, EIS, DRT, mecanismo de almacenamiento, HER/OER,
│                #   cinética (Randles-Ševčík, GITT, dQ/dV, Koutecky-Levich)
├── plotting/    # motor de figuras: estilo, series, ejes secundarios, paneles
├── dataio/      # detección de formato, lectura universal, exportación, .rcproj
├── mapping/     # mapas Raman: cubo, imágenes por píxel, quimiometría
├── gui/         # app Tkinter; la lógica vive en state.py y plots.py, sin Tk
├── cli/         # ramancarbon analizar / lote / deconvolucionar / bd / demo
├── examples/    # scripts ejecutables + demo_data.py (espectros sintéticos)
└── tests/       # pytest
```

Dependencias en un solo sentido: `core → models → analysis → gui/cli`, y
`database` no importa de ninguno.

## Reglas

1. **Python ≥ 3.10**, anotaciones de tipo en todo lo público.
2. **Docstrings obligatorios**, con la suposición física cuando la haya.
3. **Ningún valor de literatura codificado en Python.** Va en
   `database/data/*.json` con su `source` y su `confidence`. Esto es
   deliberado: el usuario es especialista y tiene que poder corregir la base
   de datos sin tocar código.
4. Todo número que dependa de la excitación **debe** corregirse por
   dispersión antes de compararse con nada.
5. Degradar, no reventar: si faltan datos, devuelve `None` con una razón o un
   objeto con `available=False` y `reason`, y dilo en el informe.
6. Los avisos son parte del resultado. Si un número puede estar sesgado, el
   objeto que lo lleva tiene que decirlo.
7. Nada estocástico sin `seed`.
8. Cada función nueva con su prueba en `tests/`.
9. Sin dependencias fuera de `numpy`, `scipy`, `matplotlib` sin justificarlo.

## Barreras físicas — no las bajes

Cada una está ahí porque su ausencia produjo un resultado incorrecto, y cada
una tiene una prueba que la protege.

- **Despiking por mediana móvil, no por primera diferencia.** El criterio de
  primera diferencia escala con la pendiente y marcaba los flancos de una
  banda G intensa como rayos cósmicos, dejando muescas en la banda.
- **Detección de picos por significancia de filtro adaptado con corrección
  por número de pruebas.** Un umbral de «3σ de altura» encuentra picos en
  ruido puro casi siempre; con él aparecían cuatro RBM fantasma en todo
  espectro de grafeno. El valor por defecto está calibrado sobre 400
  ventanas de ruido puro.
- **La ausencia de RBM solo es prueba si el espectro cubrió esa región.** Sin
  esa comprobación (`Spectrum.covers`), todo espectro de 400–3000 cm⁻¹ se
  clasifica como multipared.
- **Los cocientes se comparan en la base en que se publicó el rango.** Áreas
  y alturas difieren en un factor 2–3. Usa `Ratio.on_basis()`.
- **La metalicidad se decide ajustando BWF contra lorentziana y comparando
  por BIC**, no adivinando por la altura del hombro. Las dos constantes del
  desdoblamiento G difieren un 40 %.
- **`bwf_peak_position` es `ω₀ + Γ/q`.** La derivada factoriza como
  `(1 + b u)(b − u)`; el máximo está en `u = b`. Una versión anterior
  devolvía la rama del mínimo.
- **Un modelo D–G de nanotubo tiene que incluir G⁻** (preajuste
  `swcnt_full`). Sin ella la D se estira absorbiendo esa intensidad y
  I_D/I_G sale varias veces demasiado grande.
- **La 2D tiene que estar detectada antes de ajustarla.** Si no, en un óxido
  el ajustador clava una componente en el borde de la ventana y llama banda
  a la cola de D+G.
- **El desdoblamiento G no se aplica a multipared**: allí lo que acompaña a
  la G es D', una banda de defectos, y la fórmula convertiría un defecto en
  un diámetro.
- **Los rangos de FWHM de `bands.json` son también los límites del ajuste.**
  Estrecharlos hace que los ajustes se peguen a ellos y sesga las anchuras.
- **La asignación se compone de todas las regiones ajustadas más el detector
  de picos.** Con solo el ajuste D–G, la 2D queda sin asignar.
- **I_D/I_G no es monótona con el desorden.** Informa siempre las dos ramas
  de Tuinstra–Koenig; citar solo la de bajo desorden puede errar un orden de
  magnitud.
- **La G sube con electrones y con huecos.** El signo del dopado solo lo da
  la 2D. No escribas nada que sugiera lo contrario.
- **La separación deformación/dopado está calibrada para grafeno monocapa.**
  Aplicarla a nanotubos exige el aviso correspondiente.
- **Raman no separa las configuraciones del nitrógeno.** Para eso hace falta
  XPS, y la base de datos lo dice explícitamente. No lo suavices.
- **asLS es la línea base por defecto, medido, no por reputación.** arPLS se
  recomienda mucho para fluorescencia; contrastado contra fondos sintéticos
  conocidos a la rigidez que cada uno usaría de verdad, asLS ganó en todos
  los casos y arPLS arrastraba un desplazamiento sistemático. Si alguien lo
  revierte, que sea con medidas nuevas, no con la cita.
- **La rigidez de la línea base sale de la función de transferencia del
  suavizador de Whittaker**, fijando el corte en cinco anchuras de banda. El
  primer intento puntuaba rigideces por cómo seguían al fondo: eso NO
  funciona, y el módulo explica por qué (el residuo lo domina el ruido, y su
  parte sistemática se minimiza con la línea base más blanda, la que se come
  los picos). No vuelvas a la búsqueda en rejilla.
- **λ depende del paso de muestreo** porque el corte es un periodo en
  puntos. Copiar un λ de un artículo sin reescalarlo es un error de 256×
  entre 1 y 4 cm⁻¹/punto.
- **SNIP se pega con relleno reflejado IMPAR.** Con recorte en el borde,
  la media de un punto consigo mismo y su vecino cae por debajo de la
  curva, y el clipeo se come el extremo: un tercio del fondo en el primer
  canal, con el interior exacto. La reflexión par lo reduce a la mitad; la
  impar, que continúa la pendiente en vez de darle la vuelta, lo elimina.
- **La normalización 0-100 resta un desplazamiento y eso cambia los
  cocientes.** Es la única de las normalizaciones que lo hace. Avisa.
- **Un modelo D–G de nanotubo con G⁻ hace falta también con perfil
  forzado**: la BWF de un G⁻ metálico sobrevive al override, porque es
  mecanismo, no preferencia de forma.
- **Γ_G es monótona con el desorden e I_D/I_G no.** En material muy
  desordenado, Γ_G manda. No lo entierres entre los demás índices.
- **Raman no distingue MWCNT de CNF con fiabilidad.** Peso bajo y aviso
  explícito cuando cae en el solape. No subas ese peso.
- **Con S, P y Se la G puede BAJAR aunque el dopado sea de tipo n**, porque
  el enlace C–X más largo ablanda la red más de lo que la carga la endurece.
  Los rangos de esos dopantes cruzan el cero a propósito.
- **Los materiales con `role: "reference"` no compiten en la clasificación.**
  Un MWCNT dopado con N sigue siendo un MWCNT.
- **Un ajuste por mínimos cuadrados SIEMPRE devuelve componentes.** Sobre un
  espectro de ceros el modelo de cinco bandas devolvía D4, D, D3, G y D′ y el
  informe anunciaba «MWCNT, confianza media, I_D/I_G = 1.43». Las componentes
  ajustadas pasan por un umbral de significancia antes de asignarse
  (`MIN_COMPONENT_SIGNIFICANCE`). No lo quites.
- **Sin banda G no se clasifica nada.** Es el único modo permitido en primer
  orden del carbono sp² y la referencia de todos los cocientes. Un espectro
  que solo cubría la región 2D se clasificaba como grafito.
- **`Spectrum` rechaza intensidades no finitas.** Un NaN atravesaba la línea
  base y el ajuste y salía como una clasificación con aspecto de válida.
- **Usa `core.compat.trapezoid`, nunca `np.trapezoid`.** El paquete declara
  `numpy>=1.24` y `np.trapezoid` es de NumPy 2.0. Hay una prueba que audita
  esto.
- **La detección de fases no carbonosas exige corroboración**: varias líneas
  fuertes de la especie, no una. Con una sola, en la ventana RBM la
  coincidencia es más probable que la especie, y la primera versión tiró
  cuatro de cinco RBM auténticos.
- **El escaneo de interferencias va APAGADO por defecto**, en la API, en la
  CLI y en la GUI. Es una decisión, no un descuido.
- **`analyse_many` fija el BLAS a un hilo por proceso y usa `spawn`.** Sin lo
  primero el paralelismo es 8× más lento que el serie; sin lo segundo la
  variable de entorno no llega al hijo. Y si el script del usuario no tiene
  guarda `__main__`, degrada a serie con un aviso: no lo conviertas en error.
- **Las incertidumbres analíticas del ajuste son cortas.** Para números que
  se publican, usa `models.bootstrap`. Documenta siempre que el remuestreo
  mide reproducibilidad frente al ruido, no acierto.

## Fases de la muestra (FeSe, Se, carburos)

- **`analysis.phases` SÍ va encendido por defecto**, al contrario que
  `analysis.interference`. No es una incoherencia: son dos cosas distintas.
  Allí una banda ajena es suciedad; aquí es la muestra. Y el motivo concreto
  es que el β-FeSe pone sus dos modos en 181 y 196 cm⁻¹, el selenio en 237 y
  254, y la cementita en 212 y 280 — los seis dentro de la ventana RBM. Sin
  este paso, un espectro de nanotubos decorados con FeSe devuelve cinco
  diámetros de nanotubo creíbles y falsos. Hay una prueba que lo demuestra en
  las dos direcciones.
- **La corroboración también manda aquí**: una fase es un espectro, no una
  línea. Las coincidencias sueltas se listan aparte y ocultas por defecto.
- **Un límite SUPERIOR de anchura no corrobora nada.** «Más estrecha de
  15 cm⁻¹» lo cumple cualquier RBM. Solo un límite INFERIOR es prueba
  positiva, y por eso el selenio amorfo se identifica con una sola banda y el
  monoclínico no. Convertir el 254 cm⁻¹ del SWCNT limpio en selenio
  monoclínico fue exactamente ese error.
- **Raman propone la fase, DRX la demuestra.** Cada fase lleva su `xrd_hint`
  con la reflexión y el 2θ que zanja la duda, y el informe lo dice cuando la
  familia queda sin resolver. No conviertas eso en una afirmación.
- **La fase hexagonal del FeSe lleva confianza `low` a propósito**: su
  literatura Raman no es consistente. No la subas sin medidas nuevas.
- **Las intensidades relativas entre fases NO son fracciones másicas.** Las
  secciones eficaces difieren en órdenes de magnitud y dependen de la
  resonancia. Para cuantificar, Rietveld.

## Dicalcogenuros (TMD)

- **La medida es una SEPARACIÓN, no una posición.** E₂g se ablanda y A₁g se
  endurece al apilar; la diferencia cancela los errores comunes de
  calibración. No lo sustituyas por posiciones absolutas.
- **En WSe₂ el método no vale** (modos casi degenerados) y en MoSe₂/MoTe₂
  apenas discrimina. Allí se cuenta por el modo B¹₂g, prohibido en monocapa.
  No inventes un número de capas donde el material no lo permite.
- **Los modos J prueban fase 1T′; su ausencia no prueba nada.** Mantén esa
  asimetría en el texto.
- **La resolución manda**: bandas de 2–6 cm⁻¹ y fronteras a 2–3 cm⁻¹. Avisa
  siempre que el paso de muestreo no llegue.
- **MoS₂ es el único con confianza alta.** MoSe₂, WSe₂ y MoTe₂ llevan
  confianza baja a propósito.
- **El análisis TMD es un camino aparte del de carbono, no una rama del
  clasificador.** Nadie necesita que el programa decida entre «MWCNT» y
  «MoS₂ bicapa»: el usuario sabe qué puso bajo el objetivo.

## Heteroestructuras óxido/calcogenuro

- **Raman NO ve topología.** Un MoO₃@MoSe₂ y una mezcla física de polvos dan
  el mismo espectro puntual. Lo que sí se mide es el desplazamiento de los
  modos del calcogenuro, que una intercara íntima produce y una mezcla no.
  Eso es indicio, no prueba, y el texto lo dice con esas palabras. La
  notación `A@B` solo se emite si el llamante afirma la topología.
- **Un óxido solo cuenta si aparece alguna de sus líneas exclusivas.** El
  MoO₂ tiene bandas en 203 y 228 cm⁻¹ y los modos del MoSe₂ están en 240 y
  287: sin esa puerta, todo espectro de MoSe₂ «contiene» MoO₂. Las
  exclusivas del MoO₂ son 495 y 744, donde el calcogenuro no tiene nada.
- **El denominador del índice de oxidación se comprueba.** El MoO₃ tiene
  bandas en 246 y 291 justo donde están los modos del MoSe₂; medir ahí la
  altura del «huésped» es medir el óxido, y el cociente sale 1 haya el óxido
  que haya. Si ningún modo queda limpio, el índice se rechaza con su motivo.
- **El índice de oxidación no es una fracción másica** y va acompañado del
  modo con el que se calculó. Dos índices medidos contra modos distintos no
  son comparables.
- **A₁g fuera del plano = dopado; A₁g y E₂g juntos = deformación.** Con óxido
  metálico encima (MoO₂, WO₂) lo esperable es lo primero; con aislante
  (MoO₃, WO₃), lo segundo. El veredicto se juzga SOLO sobre esos dos modos:
  el B¹₂g cae donde el óxido tiene bandas y un ajuste arrastrado no es física
  de intercara.
- **Un espectro que no llega a 1000 cm⁻¹ no puede descartar un óxido.** Las
  líneas que lo identifican (819 y 995 del MoO₃, 744 del MoO₂, 807 del WO₃)
  están ahí arriba. «No hay óxido» sin esa región no significa nada, y el
  aviso lo dice.
- **Un óxido tiene tres orígenes posibles y una sola medida no los separa**:
  precursor, aire, o el propio láser. El aviso propone la comprobación
  concreta (punto virgen, media potencia, ver si crece). No lo quites.

## Difracción de rayos X

- **Nada de tablas de posiciones de pico.** Una fase es una ESTRUCTURA
  cristalina y cada posición e intensidad se calcula de ella. Por eso un
  parámetro de red refinado mueve los picos como los movería la física, y por
  eso la biblioteca de referencia son CIF: añadir una fase es soltar su
  archivo en un directorio, no editar código.
- **Nada de búsqueda en línea en la COD.** Es una limitación deliberada: un
  resultado que depende de la red no se reproduce, y los ordenadores de los
  equipos suelen estar sin conexión. Se descarga el CIF una vez y se guarda.
- **Las operaciones de simetría se CIERRAN antes de usarse.** Para un CIF
  completo eso no cambia nada; para uno truncado lo repara, y permite que las
  estructuras incluidas listen solo generadores. El orden del grupo cerrado es
  además una comprobación exacta: una operación mal escrita casi siempre lo
  cambia. Hay una prueba por cada grupo espacial usado.
- **`ClosedGroup` existe por rendimiento, no por elegancia.** Rietveld
  construye un `Crystal` nuevo por cada celda de prueba; volver a cerrar las
  192 operaciones de la magnetita cada vez se llevaba dos tercios del tiempo
  total. Igual con la caché de `expanded()`. No las quites.
- **Las imágenes de simetría que coinciden se FUNDEN.** Un átomo en el origen
  de un grupo cúbico dispersaría 24 veces de más.
- **La celda se refina con las ligaduras de la simetría**, leídas de las
  operaciones y no del nombre del grupo ni de que los números salgan iguales.
  Sin eso una celda hexagonal devuelve a=2.46359 y b=2.46451: dos números
  separados por cuatro veces su propio error que la simetría dice que son uno.
- **La longitud de onda de `Pattern` es la Kα₁, no la media Kα₁₂.** El doblete
  se modela aparte. Una longitud de onda promediada no acierta en ningún
  ángulo: reparte el error abajo y deja un pico visiblemente asimétrico arriba,
  que el ajuste de perfil se traga como anchura y devuelve como un tamaño de
  cristalito equivocado.
- **El Kα₂ se pela (Rachinger) antes de buscar picos, y NO antes de refinar.**
  Rietveld modela las dos líneas. Y el ruido se mide ANTES de pelar: la
  recursión correlaciona puntos vecinos y encoge la estimación de σ.
- **El umbral de detección está calibrado sobre ruido Poisson puro**, veinte
  patrones de 3500 puntos: 44 picos falsos por patrón con umbral 8, 3.2 con 12,
  0.2 con 16, 0 con 20. Está en 18. Bajarlo no encuentra más fases, encuentra
  más rizos en el flanco del pico más intenso y los llama fases desconocidas.
- **La incertidumbre es PUNTO A PUNTO**, no un número por patrón. Con
  estadística de conteo, el ruido sobre un pico de 10 000 cuentas es 100 y
  sobre un fondo de 100 es 10.
- **Posiciones e intensidades se juzgan por separado.** Las posiciones solo
  dependen de la red y son prueba fuerte; las intensidades las estropea la
  orientación preferente en cualquier material laminar, que es justo el que
  aquí interesa. Fundirlas en un solo número dejaría que un artefacto de
  textura vetase una identificación correcta.
- **Los picos sin explicar SON el resultado**, no un resto. Son la fase que no
  esperabas. Y las colas de Kα₂ sobre picos ya explicados se apartan aparte:
  llamar fase desconocida a la cola de una línea del cobre es peor que callar.
- **Rietveld por etapas, siempre.** Soltar todos los parámetros a la vez
  converge, da factores R plausibles y devuelve una estructura equivocada.
- **El perfil es POR FASE.** Un carbono nanocristalino y un seleniuro bien
  cristalizado en la misma muestra tienen anchuras distintas de verdad.
- **Cero y desplazamiento de muestra son funciones distintas del ángulo**
  (constante frente a cos θ). Refinar solo el cero contra una muestra mal
  montada deja un residuo sistemático que ningún refinamiento posterior quita.
- **Las intensidades para refinar NO se normalizan.** El factor de escala solo
  es proporcional a la cantidad de fase si multiplica intensidades absolutas.
- **Las fracciones en peso son de la parte CRISTALINA E IDENTIFICADA.** Una
  fase sin modelar no baja el total de 100 %: su intensidad se reparte entre
  las demás. El amorfo no aparece. Ese aviso va siempre.
- **La curva diferencia antes que los factores R.** Un Rwp bonito con una
  ondulación sistemática en el residuo es peor ajuste que un Rwp feo con
  residuo sin estructura. El informe da además el estadístico de rachas.
- **Scherrer se niega por encima de ~120 nm.** Ahí el ensanchamiento por
  tamaño es menor que la resolución del equipo y el «resultado» es una
  reformulación de lo que hayas supuesto para el instrumento.
- **U_iso es el parámetro que se traga los errores de los demás.** Si sale
  cero o enorme, el problema está en el fondo, la absorción o una fase que
  falta. El aviso lo dice.
- **Sin dispersión anómala (f′, f″), a propósito.** Lo que de verdad arruina
  una muestra de hierro con tubo de cobre es la fluorescencia, que sube el
  FONDO y no toca los picos; de eso sí avisa el programa.
- **Las estructuras incluidas se validaron con dos comprobaciones
  independientes**: densidad cristalográfica frente a la literatura (valida
  celda y contenido a la vez) y posición de la reflexión más intensa frente a
  su ficha. Las dos están en las pruebas. Si añades una fase, añade las dos.

## Modelos a medida

- **Las componentes llevan nombre solo mientras la física se lo dé**: D, D3,
  D4 en la región D; G, D′, G⁻ en la G. Las extra salen sin `band`, y por eso
  no entran en cocientes ni clasificación. No les asignes una banda para que
  «cuadre».

## Electroquímica

- **Un material con picos redox tiene CAPACIDAD, no capacitancia.** Dividir
  la carga por la ventana entera y dar F/g es la práctica más criticada de
  toda la literatura de almacenamiento de energía y puede multiplicar la
  cifra por varias veces. `echem.evaluate.classify_storage` decide el
  mecanismo y, si es tipo batería, el informe se niega a presentarlo en
  faradios. No lo suavices.
- **La b≈1 NO distingue doble capa de pseudocapacitivo**, porque los dos la
  dan. Solo separa capacitivo de batería. La primera versión trataba b≥0.9
  como prueba de doble capa y clasificaba TODO pseudocondensador como EDLC.
  Lo que separa esos dos es la presencia de picos redox en el CV.
- **Un pico de batería es ESTRECHO y ALTO sobre su fondo**, no simplemente
  muy separado de su pareja. Hay materiales de batería con ΔEp por debajo de
  100 mV. El criterio son anchura frente a ventana y prominencia frente a
  fondo.
- **La b que importa es la del pico, no la mediana de la ventana.** Un
  electrodo de batería está limitado por difusión solo donde ocurre su
  proceso redox; en el resto la doble capa mantiene b en 1 y una mediana se
  come el único valor informativo.
- **Hay DOS convenios de capacitancia desde un CV y difieren en un factor
  exacto de 2.** Se informan los dos, con su nombre. Publicar sin decir cuál
  deja esa ambigüedad.
- **Recortar los vértices y luego dividir por la ventana ENTERA** dejaba
  toda capacitancia un 4 % baja. Se divide por el intervalo realmente
  integrado.
- **La caída IR se quita de la ventana de descarga**, y su tamaño se
  informa: es el diagnóstico más útil que da una curva galvanostática.
- **La energía se INTEGRA (∫V dq), no se supone (½CV²).** Ojo con la prueba:
  ½CV² acierta también en una meseta CENTRADA, por casualidad. Por eso la
  meseta del demo está descentrada, como las de verdad.
- **El potencial de referencia va con su relleno.** Ag/AgCl 3 M y SCE están
  a 31 mV. Sin pH, la conversión a RHE se RECHAZA en vez de suponerlo: son
  59 mV por unidad de pH en todo lo que venga después.
- **Sin corrección óhmica no hay pendiente de Tafel ni ΔEp cinético**, y el
  sesgo crece con la corriente. Se avisa siempre que falte.
- **Una pendiente de Tafel sobre menos de una década no es una pendiente de
  Tafel.** La región lineal se BUSCA (residuo del ajuste frente al ruido de
  los datos, con sumas acumuladas para que sea O(n)). El primer intento
  exigía pendiente local constante y troceaba una década recta en tramos de
  0.1: la pendiente salía bien y se rechazaba por corta.
- **El ECSA arrastra un factor de tres** por la Cs que se adopte. Va con su
  rango y con el aviso de que solo sirve para comparar TUS muestras.
- **Kramers-Kronig ANTES del circuito.** Un circuito ajustado a datos que
  han derivado da parámetros sin significado y el χ² no lo delata. La base
  del test lleva capacidad en serie e inductancia: sin ellas, un
  supercondensador perfectamente bueno falla el test por su cola capacitiva.
  Y se juzga por la ESTRUCTURA del residuo (rachas de signo), no por un
  umbral fijo: 0.5 % de ruido pone el residuo máximo en 1.5 % por azar.
- **Los circuitos se ajustan en espacio logarítmico**, con varios reinicios.
  Los parámetros abarcan diez órdenes de magnitud y un paso razonable para
  40 Ω es absurdo para 2e-4 S·sⁿ. Ajustado en lineal, un Randles con cola
  capacitiva llevaba su resistencia al límite y decía haber convergido.
- **Las incertidumbres se convierten de vuelta del logaritmo** (σ_p/p =
  ln10·σ_log). Informar la σ logarítmica como si fuera lineal convirtió un
  3 % en 3 000 000 %.
- **La Q de un CPE NO son faradios.** La conversión (Brug) necesita la
  resistencia en paralelo y es una llamada aparte, a propósito.
- **El apex del semicírculo es un máximo LOCAL de −Z″.** Tomar el global
  falla en todo lo que tenga cola capacitiva y sembraba la resistencia de
  transferencia con el valor de la cola.
- **Z″ se guarda con su signo físico (negativo si es capacitivo).** El −Z″
  del diagrama de Nyquist es convenio de dibujo; guardarlo negado es una
  fuente permanente de errores de signo en los ajustes.

## Microestructura y ajuste de patrón completo

- **Un solo pico NO separa tamaño de deformación.** Scherrer sobre una
  reflexión asigna todo el ensanchamiento al tamaño en silencio, y en una
  muestra tensionada eso está mal en lo que aporte la deformación. Hacen
  falta varias reflexiones, y repartidas en ángulo: tres picos entre 20 y
  30° determinan una pendiente tan mal como uno.
- **Los tres métodos se informan siempre, porque discrepan.** Williamson-
  Hall supone dos Lorentzianas que se suman, el gráfico tamaño-deformación
  supone tamaño Lorentziano y deformación Gaussiana, y Halder-Wagner
  supone Voigt en espacio recíproco. Tres tamaños que coinciden significan
  que el modelo vale; tres que difieren en un factor de dos significan que
  no, y entonces NO se cita un número.
- **Un corte negativo no es un «tamaño de cristalito negativo».** Es el
  modelo aditivo fallando, y se dice así en vez de devolver el número.
- **Halder-Wagner no lleva constante de forma**, así que su tamaño difiere
  de los otros dos exactamente en K aunque el ajuste sea perfecto. Está
  documentado porque si no parece un error.
- **Las fórmulas de deformación se derivan, no se copian.** Con ε definido
  por β = 4ε tan θ, el corte del gráfico tamaño-deformación es 4ε²λ² y el
  de Halder-Wagner 4ε². Copiar el «(ε/2)²» del libro sin comprobar qué ε
  es y en qué unidades están d y λ es un factor de ocho.
- **Con y a la vez plano, R² es 0/0.** Una muestra sin deformación da
  exactamente eso en un Williamson-Hall, y la fórmula ingenua devolvía
  −8.5, que se lee como un ajuste catastrófico en vez de uno perfecto.
- **El carbono lleva K = 1.84 para L_a**, no 0.89. Es la constante
  bidimensional, y usar 0.89 ahí da la mitad de lo que publica todo el
  mundo.
- **El grado de grafitización solo vale entre 3.354 y 3.44 Å.** Fuera de
  ahí no es un porcentaje de nada: por encima, el apilamiento no tiene
  registro; por debajo, hay un error de cero o de desplazamiento que va
  entero a L_c también.
- **El amorfo NO se mide con Rietveld.** Sus fracciones son de la parte
  cristalina modelada y suman 100 % siempre. Hace falta patrón interno:
  se añade una cantidad conocida y el exceso que el refinamiento le
  atribuye es lo que falta.

- **Le Bail y Pawley refinan la celda sin estructura**, y fallan de forma
  distinta: Le Bail reparte a partes iguales lo que se solapa exactamente
  (el reparto es la suposición inicial sobreviviendo), Pawley pone las
  intensidades como parámetros y ahí la correlación se ve.
- **El Kα₂ va en la MISMA columna que su reflexión madre**, con la razón
  fija. Dejarlo fuera no es neutro: el ajuste tiene que tapar el satélite
  con algo, y usa la anchura, que salía cuatro veces la verdadera.
- **La partición de Le Bail se corre MUCHAS veces por ciclo.** Desde
  intensidades uniformes mejora el Rwp una décima parte por pasada —24 %
  a las diez, 7 % a las cuarenta— y un ciclo que reparte una vez y luego
  refina está refinando celda y anchura contra intensidades equivocadas
  por un factor de dos. Cada pasada son dos productos de matrices.
- **El perfil de partida se mide del patrón, no se hereda.** Los
  coeficientes de Caglioti por defecto describen el difractómetro de otro,
  y con picos modelados a la mitad de anchos la partición no baja del 35 %.
- **Las reflexiones se enumeran UNA vez.** Volver a enumerarlas cada ciclo
  parece equivalente y no lo es: una reflexión que cruza el borde de la
  ventana cambia la longitud del vector de intensidades a mitad del
  refinamiento, y las intensidades pasan a ser de otras reflexiones.
- **El solapamiento se calcula por reflexión y se informa.** Para una
  reflexión aislada la intensidad extraída es una medida; para un par a
  0.01° es un convenio, y esa diferencia tiene que verse antes de pasarle
  los datos a un programa de resolución estructural.
- **Un ajuste de patrón completo SIEMPRE gana a un Rietveld del mismo
  patrón**, porque tiene un número libre por reflexión. Comparar sus
  factores R no dice nada. Lo que da es un suelo.

## Mapas Raman y quimiometría

- **Un mapa es un CUBO, no una lista de espectros con coordenadas.**
  Extraer la intensidad de una banda en 40 000 píxeles es una operación
  de array sobre el cubo y 40 000 llamadas sobre la lista, y esa
  diferencia es la que hay entre un mapa que se redibuja al mover un
  deslizador y uno que tarda un minuto.
- **El paso espacial frente al tamaño del punto láser decide si los
  píxeles vecinos son medidas independientes.** Con paso menor que medio
  punto, cualquier estadística píxel a píxel está midiendo la óptica. El
  mapa lo dice, y no adivina el tamaño del punto cuando no está declarado.
- **Un píxel perdido es NaN y se propaga.** Colarlo como cero baja todas
  las medias sin que nada lo delate.
- **Un área por píxel necesita SU PROPIA línea base local.** La
  fluorescencia varía por la muestra mucho más que la señal Raman, así que
  un «mapa de la banda D» hecho con áreas crudas suele ser un mapa de la
  pendiente del fondo. En el mapa de demostración el gradiente mete un
  68 % de izquierda a derecha.
- **Una posición se INTERPOLA, no se elige.** `argmax` solo devuelve
  valores de la rejilla espectral, así que con paso de 2 cm⁻¹ el mapa sale
  en terrazas, y las terrazas se leen como dominios. Parábola por los tres
  puntos del máximo.
- **Un cociente se enmascara donde el denominador es ruido.** Dividir por
  una banda que no está produce los píxeles más brillantes de la imagen, y
  están todos donde no hay muestra. Se informa cuántos se han enmascarado:
  medio mapa enmascarado significa que la banda no está.
- **Los rayos cósmicos, antes que todo lo demás.** Un pico de un canal en
  un píxel es la mayor excursión del cubo entero, así que sale como
  primera componente principal y como grupo propio de k-medias.
- **Y el despicado se juzga con el ruido DEL PÍXEL, no con el local.** La
  desviación local es diminuta donde el espectro es suave, así que el
  máximo de cada banda bien muestreada supera el umbral y se aplana: esa
  versión tocaba medio por ciento de todos los canales de un mapa
  sintético limpio, casi todos máximos de banda. El criterio son dos
  cosas: excursión grande en unidades del ruido del píxel Y estrecha.
- **PCA sobre datos SIN centrar da la media como primera componente.** Se
  centra. Y una componente principal es una dirección de varianza, no una
  sustancia: son ortogonales por construcción y nada obliga a que dos
  componentes químicas reales lo sean.
- **k-medias sin normalizar ordena por BRILLO**, que lo fijan el enfoque y
  la topografía. Por eso la normalización por área va puesta por defecto.
  Y k-medias devuelve exactamente k grupos haya o no k fases: la silueta
  es lo que dice si están separados.
- **MCR-ALS da espectros porque impone no negatividad**, y su solución NO
  es única: cualquier rotación del par que mantenga las dos partes no
  negativas ajusta igual de bien. Eso es del problema, no del programa, y
  el aviso va siempre. Las concentraciones son relativas, nunca fracciones
  másicas.
- **El cierre solo si las componentes son todo lo que hay.** En un mapa con
  sustrato desnudo, obligar a que sumen uno obliga a que el vacío esté
  hecho de algo.

## Temperatura y polarización

- **El exponente del prefactor anti-Stokes/Stokes no está zanjado**: 4 para
  potencia dispersada, 3 para un detector que cuenta fotones, que es lo
  que es un CCD. Son decenas de kelvin. El parámetro existe para que el
  convenio se declare, no se suponga.
- **Sin calibración de la respuesta del equipo, la temperatura es
  APARENTE.** La red, los filtros y el detector no transmiten igual las
  dos ramas, y un filtro de borde corta la anti-Stokes a machete. El sesgo
  es de cientos de kelvin y siempre hacia arriba.
- **Las áreas van con su fondo local restado.** Un desplazamiento plano de
  20 cuentas bajo una banda Stokes de 1000 y una anti-Stokes de 400 sube
  el cociente un 8 %, y un 8 % son 25 K a temperatura ambiente, siempre
  hacia arriba.
- **Los modos duros no sirven a temperatura ambiente.** La rama
  anti-Stokes de la G a 300 K es 8·10⁻⁴ de la Stokes. Se rechaza con su
  motivo en vez de devolver la temperatura del fondo.
- **Un cociente por encima del límite clásico se rechaza**: en dispersión
  no resonante de orientaciones al azar, ρ > 0.75 no puede pasar, y una
  temperatura por encima de 1200 K en un experimento corriente casi
  siempre es fondo dentro de la ventana anti-Stokes.
- **Una red de difracción es ella misma un polarizador.** Su eficiencia
  difiere entre s y p en decenas por ciento, así que un ρ medido lleva
  dentro el del espectrómetro, y el del espectrómetro suele ser el mayor.
- **La dependencia angular de un dispersor unidimensional es cos⁴, no
  cos².** El atajo de cos² da un parámetro de orden como el doble del que
  toca. Y con tres ángulos cualquier modulación ajusta perfecto: hacen
  falta cuatro, y 180° de recorrido.

## Entrada y salida

- **El tipo de archivo se decide por los NÚMEROS, no por la extensión.**
  Los cuatro instrumentos escriben `.txt`, `.csv`, `.dat` y `.asc`, y el
  usuario tiene los cuatro en la misma carpeta. Lo que separa las medidas
  es su forma: un voltamperograma vuelve sobre sí mismo y nada más lo
  hace; una impedancia trae tres columnas y siete décadas de frecuencia;
  un difractograma cabe en 0–165° con cuentas enteras.
- **Un reverso se cuenta con umbral, no por cambio de signo de la primera
  diferencia.** Sin umbral, un difractograma limpio tiene «varios cientos
  de reversos» y se confunde con una curva de carga-descarga. El umbral es
  un 5 % del recorrido de la propia columna.
- **Se lee el archivo entero, no las primeras líneas.** La decisión depende
  de dónde TERMINA el eje x, y los primeros 400 puntos de un espectro
  Raman van de 90 a 490 cm⁻¹, que está dentro del intervalo de 2θ.
- **Una detección lleva sus motivos.** Un tipo adivinado mal tiene que
  poder discutirse; por eso `Detection` guarda las razones y las lecturas
  alternativas, y el informe las enseña.
- **Un archivo desconocido es un resultado, no una excepción.** Una
  importación de carpeta que aborta con el primer LÉEME pierde los otros
  noventa y nueve archivos, y el LÉEME es el caso normal.
- **Los formatos binarios se rechazan diciendo qué exportar.** Adivinar la
  disposición de un `.wdf` o un `.raw` es cómo se leen mal unos datos sin
  que nadie se entere. Y el binario se detecta por el contenido: los
  fabricantes escriben binario dentro de `.txt`.
- **JCAMP-DX se lee con sus tres codificaciones** (AFFN, SQZ, DIF/DUP). Un
  lector que ignora DIF convierte una rampa en un espectro y no parece
  roto. Un DUP después de un DIF repite la DIFERENCIA, no el valor.
- **Las comprobaciones internas del JCAMP se verifican, no se creen.** Cada
  línea repite su propia X y la última Y de la anterior; un archivo
  truncado o concatenado solo se detecta ahí, y `##NPOINTS` lo confirma.
- **Nada de adivinar un número que hace falta.** Una velocidad de barrido
  supuesta da una capacitancia equivocada exactamente en esa proporción y
  nada aguas abajo lo delata. Se pide el argumento.
- **Una curva de carga-descarga se exporta CON su columna de corriente.**
  El signo es lo que separa las ramas: sin él la curva se puede mirar,
  pero ya no se puede analizar.
- **Las unidades van en su propia fila.** `C (F/g)` en la cabecera no se
  puede volver a leer; una fila de unidades sí, y la hoja de cálculo la
  enseña igual.
- **Se redondea a cifras SIGNIFICATIVAS, y a la incertidumbre cuando la
  hay.** 50.1234 ± 0.4 afirma una precisión que nadie tiene.
- **Un proyecto `.rcproj` lleva los datos DENTRO.** Un proyecto que apunta
  a `C:\Users\...` no se abre en el ordenador de nadie más, y ese es
  justo el momento en que se quiere un archivo de proyecto: mandárselo a
  un coautor. La ruta original se guarda como procedencia.
- **Y lleva los ajustes con los números**: láser, longitud de onda,
  velocidad de barrido, electrodo. Sin ellos los datos se reabren y ya no
  se pueden analizar.
- **Un proyecto es ZIP + JSON + CSV, legible a mano.** Un formato que solo
  este programa sabe leer es una forma de perder datos dentro de cinco
  años. Y un proyecto escrito por una versión más nueva se abre con aviso,
  no con error: las medidas son lo único irreemplazable del archivo.

## El motor de figuras

- **Una figura es un VALOR, no una llamada.** `Plot` lleva sus series, su
  estilo, sus marcas, sus bandas, sus notas y su recuadro, y se convierte a
  diccionario y vuelve. Por eso una figura se guarda en un proyecto, se
  reabre y se redibuja idéntica, y por eso el mismo objeto se exporta con
  el ancho de columna de una revista y con los tamaños de una presentación
  sin tocar los datos.
- **JSON no tiene tuplas.** `restore_tuples` las reconstruye al leer. Sin
  eso, un estilo guardado y releído no es igual al que se escribió —
  `(10.0, 80.0)` vuelve como `[10.0, 80.0]` — y «¿ha cambiado algo desde que
  guardé?» deja de tener respuesta.
- **Los preajustes de revista llevan el ancho de columna REAL y el tamaño
  de letra a la vez.** Una figura dibujada a 6 pulgadas y encogida a 3.25
  llega con la letra a la mitad. Ese emparejamiento es el preajuste.
- **La paleta por defecto es segura para daltónicos** (Okabe-Ito) y sigue
  separándose en escala de grises. No la cambies por la de matplotlib.
- **Lo que se exporta son los números DIBUJADOS**, con desplazamiento y
  normalización aplicados. Exportar el origen obliga a quien lea el archivo
  a adivinar qué se le hizo a la curva para que se pareciera a la figura.
- **Una curva magnificada lo dice en la leyenda** (`(×5)`). Es la forma más
  común de que una figura mienta sin que nadie lo pretenda.
- **El paso de una cascada es una fracción del recorrido DESPUÉS de
  normalizar.** Con espectros crudos, el más intenso fija el paso y los
  demás se aplastan en una línea.
- **Los ticks de un eje secundario son redondos en las unidades
  SECUNDARIAS.** Un eje de d con marcas en 3.113, 2.668 y 2.403 Å porque
  ahí cayeron los 2θ redondos no es un eje que nadie pueda leer. Y las
  marcas que se pisarían se descartan: una transformación no lineal
  amontona sus valores redondos en un extremo.
- **La escala de raíz necesita sus propios ticks.** El localizador
  automático elige valores redondos (0, 2000, 4000) que se apelotonan
  arriba y deshacen la razón de usar la escala; los redondos son los de √y.
  Y se aplican DESPUÉS de fijar los límites.
- **Un recuadro no puede mover los ejes principales.** Los conectores de
  `indicate_inset_zoom` cuentan como datos y estiraban los límites; se
  congelan alrededor del dibujo del recuadro.
- **Un estilo compartido en una figura de paneles no toca los ejes de cada
  panel.** Etiquetas, límites y escalas son del panel; compartir estilo no
  puede rotular un difractograma con los ejes de un voltamperograma.

## Cinética electroquímica

- **Todo esto acaba en un coeficiente de difusión, y un coeficiente de
  difusión depende de un ÁREA que casi nadie mide.** La superficie
  electroquímicamente activa de un electrodo poroso es de diez a mil veces
  la geométrica, y Randles-Ševčík divide por el área al cuadrado: seis
  órdenes de magnitud. Ninguna función calcula un D sin que se le diga qué
  área usa, y todas lo dicen en el resultado.
- **La linealidad de i_p frente a √v ES la comprobación del método**, no
  un detalle. Si no se cumple, el proceso no está limitado por difusión y
  lo que devuelve la fórmula es la pendiente de una recta por puntos que
  no están en una recta.
- **Si la recta no pasa por el origen hay corriente capacitiva
  superpuesta**, y hay que restarla antes de que la pendiente signifique
  algo.
- **En la región de Warburg, las pendientes de Z′ y de −Z″ frente a
  ω^(−1/2) tienen que COINCIDIR.** Ajustarlas por separado y compararlas
  es lo que distingue la región de Warburg de la cola del semicírculo de
  transferencia de carga.
- **En GITT, la caída óhmica se excluye ajustando E frente a √t**, no
  restando el salto inicial a ojo; y si se incluye, ΔE_τ sube y D baja con
  su cuadrado.
- **Que el reposo haya terminado se decide EXTRAPOLANDO**, no mirando
  cuánto se movieron los últimos puntos. Un reposo cortado a un tercio de
  su constante de tiempo tiene la cola casi plana y no está ni cerca del
  equilibrio: parece quieto solo porque la exponencial es lenta.
- **dQ/dV se calcula sobre UNA rama, no sobre todas las descargas del
  archivo pegadas.** Cada unión mete un escalón en el potencial, y la
  derivada ahí es enorme: salían setenta y ocho picos falsos.
- **El suavizado de una dQ/dV es parte de la medida.** Poco, y sale un
  peine de ruido; mucho, y dos transiciones próximas se funden. Va siempre
  informado: una curva dQ/dV sin su suavizado no se puede comparar con la
  de nadie.
- **Los extremos no son picos.** Una curva de descarga se aplana al
  agotarse, así que |dQ/dV| sube monótona hasta el último punto y el
  último punto es «mayor que su vecino».
- **Sin el ciclo de vuelta a la velocidad inicial, una limitación cinética
  y una degradación dan la MISMA curva.** El 95 % recuperado dice que era
  cinética; el 60 %, que la prueba de velocidad fue un experimento de
  degradación.
- **La forma del desvanecimiento es el mecanismo**: lineal es pérdida de
  material activo, √ciclos es una capa que se frena a sí misma al crecer,
  exponencial acaba en cero. Y si los tres ajustan igual de bien sobre los
  ciclos medidos, la extrapolación no significa nada: se compara el mejor
  con el SEGUNDO mejor, no con el peor.
- **La n de Koutecky-Levich solo vale lo que valgan la D y la ν
  tabuladas.** Un 10 % en D mueve n un 7 %, así que un 3.7 y un 4.0 no se
  distinguen.
- **Un Ragone sin decir su base no se compara con nada**: por gramo de
  material activo y por kilogramo de celda empaquetada difieren en un
  factor de tres a cinco.

- **La DRT es una inversión mal condicionada, y la regularización es una
  ELECCIÓN sobre cuánta estructura creerse.** Va informada siempre, y se
  elige por la curva L cuando no se da. Una década a cada lado del codo da
  una DRT defendible con otro número de picos.
- **γ se penaliza en su RUGOSIDAD, no en su tamaño.** Penalizar el tamaño
  encoge todos los procesos hacia cero y sesga las resistencias a la baja;
  penalizar la rugosidad funde constantes de tiempo vecinas, que es el
  fallo honrado: dice «estas dos no se resuelven» en vez de «las dos son
  más pequeñas de lo que son».
- **γ es una DENSIDAD, no un peso por casilla.** El modelo resuelve
  g_k = γ(τ_k)·Δlnτ; olvidar dividir por el ancho de casilla es un factor
  silencioso de 0.23 en la rejilla por defecto, que convirtió un proceso de
  20 Ω en uno de 4.7 Ω.
- **La resistencia de un pico se integra sobre su CUENCA entera**, de
  mínimo a mínimo, no sobre la parte por encima de una fracción de su
  altura: truncar a la décima parte tira casi toda el área.
- **Dos picos a menos de un factor de tres en τ son uno.** A esa distancia
  la separación depende de λ más que de los datos, y un circuito
  equivalente los describiría igual de bien con un solo CPE.

## La suite: cuatro secciones

- **Cuatro instrumentos, cuatro secciones**: Raman carbono, Raman TMD, DRX,
  Electroquímica. Son medidas distintas y no se mezclan. Lo único que se
  comparte es la `Session` entre las dos secciones Raman, porque un archivo
  es un archivo.
- **Las secciones se construyen al visitarlas por primera vez.** Construir
  las cuatro al arrancar importa matplotlib, monta una docena de figuras y
  lee la biblioteca de referencia antes de que aparezca la ventana.
- **La lógica sigue sin Tk**: `xrd_state.py` y `echem_state.py` son el
  equivalente de `state.py`, y `plots_xrd.py`/`plots_echem.py` el de
  `plots.py`. `base.py` tiene la fontanería común (lienzos, hilos, estado);
  una sección es su distribución y su lógica, no una copia de esa fontanería.
- **Un hilo de trabajo NUNCA toca un widget.** Devuelve por la cola y el hilo
  principal la vacía. Hacerlo mal no revienta: corrompe la pantalla de vez en
  cuando en unas plataformas y en otras no.
- **`root.update()`, no `update_idletasks()`, para procesar el cambio de
  pestaña.** Los eventos virtuales no son tareas de reposo, y con
  `update_idletasks()` la sección no se construye nunca. La prueba real lo
  documenta.
- **El diagrama de Nyquist va con caja cuadrada y límites iguales**, no con
  `aspect="equal"` y límites libres: eso último satisface el aspecto
  ensanchando el eje X hacia Z′ negativa, que ninguna impedancia alcanza.
- **El gráfico de Rietveld lleva la diferencia DEBAJO y en la misma escala.**
  Ponerla en un eje propio reescalado queda más limpio y destruye justo la
  comparación para la que sirve.

## Portabilidad

- **Un ajuste de mínimos cuadrados no da el mismo mínimo en dos versiones de
  SciPy.** Una prueba que compare dos rutas de refinamiento tiene que liberar
  EXACTAMENTE los mismos parámetros; comparar un ajuste de 7 grupos con uno de
  9 y exigir que coincidan pasó con SciPy 1.17 por suerte y falló con 1.18.
  Un ajuste con menos grados de libertad no puede bajar tanto el Rwp, y eso es
  aritmética, no un fallo.
- **`auto_refine` se salta el grupo `preferred` si no hay eje de textura.**
  Cualquier comparación contra él tiene que saltárselo también.

## Honestidad sobre la validación

Todo está validado contra espectros **sintéticos** generados por
`examples/demo_data.py`. Eso comprueba las fórmulas y la lógica, no la
exactitud sobre datos reales. Cualquier documentación que escribas debe
decirlo; no escribas nada que sugiera que el paquete está contrastado contra
medidas.

## Comandos

```bash
pip install -e ".[dev]"
pytest ramancarbon/tests -q
ruff check ramancarbon
ramancarbon demo datos/
ramancarbon analizar datos/demo_DWCNT_532nm.txt --laser 532 --auto --perfil pseudo_voigt
ramancarbon laseres datos/m_532nm.txt datos/m_633nm.txt
```

## La capa Tk no tiene pruebas de ejecución

No hay pantalla en el entorno donde se construyó esto, así que
`gui/app.py` nunca se ha ejecutado. Toda su lógica vive en `gui/state.py` y
`gui/plots.py`, que sí se prueban. `tests/test_gui_wiring.py` comprueba por
análisis del código que las piezas encajan — cada canvas tiene su pestaña y
su función de dibujo, cada `command=self._x` existe, cada `self.attr` que se
lee se asigna en algún sitio. Eso ya ha cazado dos errores reales que ni el
linter ni las pruebas veían. No cubre si la ventana se ve bien.
