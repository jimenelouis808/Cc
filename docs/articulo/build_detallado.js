const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, PageBreak, PageOrientation,
        LevelFormat, AlignmentType, BorderStyle } = require("docx");
const K = require("./comun.js");
const { p, rich, h1, h2, h3, bullet, code, cita, figura, table, spacer, ACCENT } = K;

const FIG = path.join(__dirname, "figuras");
const F = n => path.join(FIG, n);
const hay = n => fs.existsSync(F(n));

const c = [];

/* ============ PORTADA ============ */
c.push(new Paragraph({ spacing: { before: 1500, after: 120 },
  children: [new TextRun({ text: "Generación computacional de estructuras de nanocarbono",
    size: 44, bold: true, font: "Calibri", color: ACCENT })] }));
c.push(new Paragraph({ spacing: { after: 360 },
  children: [new TextRun({ text: "Metodología detallada, catálogo de construcciones, validación y agradecimientos",
    size: 24, font: "Calibri", color: "595959" })] }));
c.push(p("Material de apoyo para la redacción de un artículo científico.", { italics: true, color: "595959" }));
c.push(p("Paquete: nanocarbon_lab v0.2.0 (workspace nanocarbon).", { color: "595959" }));
c.push(p("Documento generado el 25 de septiembre de 2026.", { color: "595959" }));
c.push(spacer(240));
c.push(new Paragraph({ border: { top: { style: BorderStyle.SINGLE, size: 6, color: ACCENT } },
                       spacing: { after: 180 }, children: [] }));
c.push(p("Sobre las figuras: todas las imágenes de este documento son estructuras reales, "
  + "renderizadas directamente del paquete por el guion docs/articulo/figuras.py y los "
  + "guiones fig_*.py que lo acompañan. Ninguna es un esquema dibujado a mano, y los censos "
  + "de anillos impresos en cada panel salen del mismo cálculo que produjo la geometría, de "
  + "modo que la figura y el texto no pueden discrepar.", { italics: true, color: "1F4E3D" }));
c.push(p("Sobre las referencias: las citas proceden de los comentarios y docstrings del "
  + "código, donde se registraron al implementar cada método. Verifique volumen, página y año "
  + "contra el original antes de enviar: este documento acredita la procedencia, no sustituye "
  + "la comprobación bibliográfica.", { italics: true, color: "7F2704" }));
c.push(new Paragraph({ children: [new PageBreak()] }));

/* ============ 1 ============ */
c.push(h1("1. El principio que organiza todo el generador"));
c.push(p("nanocarbon_lab genera estructuras atomísticas de carbono junto con dopado, "
  + "funcionalización y análisis estructural. Pero lo que conviene defender en un artículo no "
  + "es el catálogo, sino el principio que lo organiza:"));
c.push(cita("La estadística de anillos no se impone: se deriva. En la mayoría de los "
  + "constructores no existe ninguna instrucción que diga «coloca un pentágono aquí». Se "
  + "define una superficie, se malla, y los pentágonos, heptágonos y octágonos aparecen donde "
  + "la curvatura gaussiana los exige. Euler tiene la última palabra."));
c.push(p("De ahí se sigue el resto. Si el censo se deriva, entonces la comprobación de Euler "
  + "es una verificación independiente y no una tautología; si un nodo de tres brazos lleva "
  + "heptágonos no es porque alguien lo decidiera, sino porque un nodo es una silla y una "
  + "silla tiene curvatura negativa."));
c.push(p("Las familias se reparten en dos rutas, y la distinción importa porque los controles "
  + "disponibles son distintos en cada una:"));
c.push(spacer(80));
c.push(table(["Ruta", "Cómo nace la estructura", "Familias"],
  [["Red directa", "La red de carbono se teje explícitamente: se enrolla una lámina, se "
    + "subdivide un poliedro, se corta un sector, se rota un enlace. La topología se conoce "
    + "de antemano y el censo la confirma.",
    "Nanotubos, fulerenos, cebollas, nanoconos, haeckelitas, toroides polihex, grafeno, "
    + "nanocintas, bobinas geométricas"],
   ["Superficie implícita", "Se define un campo escalar f(x); su nivel cero se malla, se "
    + "remalla y se dualiza. El censo NO se conoce de antemano: sale de la curvatura.",
    "Uniones L/T/Y/X, schwarzitas, toroides mallados, bobinas periódicas, redes de nanotubos, "
    + "superredes, jaulas"]],
  [1500, 4200, 3660]));
c.push(spacer(120));
c.push(p("El resto de este documento describe primero la ruta implícita, que es la que "
  + "requiere explicación, y después cada familia por separado."));

/* ============ 2 ============ */
c.push(h1("2. La ruta implícita, etapa por etapa"));
c.push(p("Cuatro etapas, y la figura siguiente es el mismo objeto en los cuatro momentos del "
  + "mismo cálculo."));
if (hay("fig1_metodo.png")) c.push(...figura(F("fig1_metodo.png"),
  "Figura 1. Las cuatro etapas, sobre una unión Y (radio de tubo 6 Å, brazo 22 Å, mezcla 4 Å). "
  + "(a) El campo implícito. (b) Marching cubes: los grados de vértice se dispersan de 4 a 9. "
  + "(c) Tras el remallado isotrópico quedan 5, 6 y 7. (d) El dual: cada vértice de grado d es "
  + "un anillo de d miembros."));

c.push(h2("2.1 El campo"));
c.push(p("Cada constructor de esta ruta devuelve una función escalar f(x) cuyo conjunto de "
  + "nivel cero es la pared de carbono. Dos construcciones cubren todas las familias:"));
c.push(bullet("Unión suave de cápsulas. Una cápsula es el conjunto de puntos a distancia "
  + "menor que r de un segmento. La unión suave (smooth union) de varias cápsulas mezcla sus "
  + "campos en una anchura fijada por el parámetro de mezcla. Esa mezcla es lo que crea el "
  + "cuello acampanado y de curvatura negativa en la ramificación: una unión dura dejaría una "
  + "arista viva que ningún panal puede teselar limpiamente."));
c.push(bullet("Superficies mínimas triplemente periódicas. Schwarz P, Schwarz D y el giroide, "
  + "en sus aproximaciones trigonométricas estándar. Son los templados clásicos del carbono de "
  + "curvatura negativa: donde un fulereno cierra con doce pentágonos, una schwarzita se abre "
  + "en una esponja periódica cuyos puntos de silla se teselan con heptágonos y octágonos."));
c.push(p("El campo es la única interfaz: el mallador no sabe de dónde vino, de modo que una "
  + "función suministrada por el usuario funciona igual que las predefinidas.", { italics: true }));

c.push(h2("2.2 Marching cubes, y por qué su malla no sirve todavía"));
c.push(p("El algoritmo de Lorensen y Cline da una triangulación estanca del nivel cero, pero "
  + "inservible para este fin: sus triángulos siguen la rejilla de muestreo. Medido sobre la "
  + "unión Y de la Figura 1, los grados de vértice se reparten así:"));
c.push(spacer(80));
c.push(table(["Grado del vértice", "4", "5", "6", "7", "8", "9"],
  [["Marching cubes crudo", "137", "913", "2 387", "707", "210", "16"],
   ["Tras el remallado", "—", "27", "489", "15", "—", "—"]],
  [2760, 1100, 1100, 1200, 1100, 1050, 1050]));
c.push(spacer(120));
c.push(p("Esto importa más aquí que en gráficos por computadora, porque en el dual un vértice "
  + "de malla de grado d se convierte en un anillo de carbono de d miembros. Un vértice de "
  + "grado 3 sería un anillo de tres miembros; uno de grado 9, un hueco de nueve. Ninguno "
  + "existe en carbono sp2 real. La malla cruda no es una aproximación mala de la red de "
  + "carbono: no es una red de carbono en absoluto."));

c.push(h2("2.3 Remallado isotrópico"));
c.push(p("Se aplica el esquema de Botsch y Kobbelt, repetido hasta convergencia: dividir las "
  + "aristas largas, colapsar las cortas, voltear aristas hacia el grado 6, suavizar "
  + "tangencialmente y reproyectar sobre la superficie. La longitud de arista objetivo es "
  + "1.42·√3 Å, que es la que hace del dual una red con enlaces de 1.42 Å."));
c.push(p("El resultado concentra los grados en 6, con 5 donde la superficie es convexa y 7 "
  + "donde ensilla, que es exactamente la distribución que adopta el carbono curvo real, "
  + "obtenida de la geometría y no impuesta a mano."));
c.push(h3("Dos detalles que decidieron el resultado"));
c.push(rich([{ t: "Condición de enlace en el colapso. ", b: true },
  { t: "Colapsar una arista cuyos extremos comparten más que los dos vértices opuestos rompe "
     + "la superficie. Esos colapsos se rechazan, y toda operación preserva la manifoldidad, "
     + "lo que permite que mesh_statistics vuelva a derivar la característica de Euler y que "
     + "el llamador la afirme en lugar de confiar en ella." }]));
c.push(rich([{ t: "Límite de longitud en el volteo. ", b: true },
  { t: "La aceptación del volteo de aristas era puramente topológica, sin comprobación de "
     + "longitud, de modo que una arista podía crecer sin límite y la pared de los tubos "
     + "colapsaba. Con el límite (1.8 Å, con convención de imagen mínima en celdas periódicas) "
     + "la desviación fuera de superficie de tres bobinas cayó de 0.801, 1.010 y 2.378 Å a "
     + "0.387, 0.327 y 0.361 Å. Conviene mencionarlo en la sección de métodos porque es la "
     + "diferencia entre un tubo y un tubo aplastado." }]));

c.push(h2("2.4 El dual"));
c.push(p("Cada cara de la malla se convierte en un átomo de carbono, situado en el centroide "
  + "del triángulo; cada vértice de la malla se convierte en un anillo, cuyos miembros son los "
  + "centroides de las caras incidentes en orden cíclico. Un vértice de grado 5, 6, 7 u 8 "
  + "produce por tanto un pentágono, hexágono, heptágono u octágono, y todo átomo resulta "
  + "exactamente tricoordinado porque bordea exactamente tres aristas de cara."));
c.push(p("Esa correspondencia es exacta, no aproximada, y se ve en los números: para la unión "
  + "Y de la Figura 1 el histograma de grados de la malla remallada es {5: 27, 6: 489, 7: 15} "
  + "y el censo de anillos del dual es {5: 27, 6: 489, 7: 15}, el mismo. El déficit de Euler "
  + "resulta Σ(6−n) = +12, que es 6χ con χ = 2.", { italics: true }));

/* ============ 3 ============ */
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(h1("3. La contabilidad topológica"));
c.push(h2("3.1 El presupuesto de Euler"));
c.push(p("Para una red trivalente sobre una superficie cerrada de característica de Euler χ:"));
c.push(code("   Σ (6 − n) = 6 · χ"));
c.push(p("donde n recorre los tamaños de anillo. Un fulereno (χ = 2) exige exactamente doce "
  + "pentágonos; un toroide (χ = 0) exige que cada pentágono esté emparejado con un heptágono, "
  + "sin necesidad de ningún otro tamaño. χ se lee de la malla, no se supone."));
c.push(p("Las cifras medidas lo confirman en todas las familias:"));
c.push(spacer(80));
c.push(table(["Estructura", "Censo de anillos", "Σ(6−n)", "χ esperada", "Comprobación"],
  [["Fulereno C₆₀", "12×5, 20×6", "+12", "2", "6·2 = 12 ✔"],
   ["Cebolla de 3 capas", "36×5, 390×6", "+36", "2 por capa", "3 · 12 = 36 ✔"],
   ["Nanotubo (6,6) periódico", "108×6", "0", "0", "cilindro ✔"],
   ["Nanocono de 1 pentágono", "1×5, 175×6", "+1", "—", "una disclinación ✔"],
   ["Haeckelita R5,7 plana", "16×5, 16×7", "0", "0", "plana: se cancelan ✔"],
   ["Toroide polihex (5,5)", "600×6", "0", "0", "género 1 ✔"],
   ["Toroide mallado", "38×5, 621×6, 38×7", "0", "0", "38 = 38 ✔"],
   ["Unión Y (3 brazos)", "50×5, 443×6, 38×7", "+12", "2", "6·2 = 12 ✔"],
   ["Unión X (4 brazos)", "63×5, 490×6, 49×7, 1×8", "+12", "2", "6·2 = 12 ✔"],
   ["Unión 3D (6 brazos)", "69×5, 481×6, 55×7, 1×8", "+12", "2", "6·2 = 12 ✔"]],
  [2200, 2500, 900, 1400, 2360]));
c.push(spacer(120));
c.push(rich([{ t: "Las tres uniones dan +12 sin importar cuántos brazos tengan", b: true },
  { t: ", porque una unión cerrada es topológicamente una esfera. Nadie lo dispuso: cada brazo "
     + "añade heptágonos en su cuello y pentágonos en su tapa, y la cuenta se cierra sola." }]));

c.push(h2("3.2 La segunda comprobación, verdaderamente independiente"));
c.push(p("Para las superredes existe algo mejor. La pared de una red de tubos es la frontera "
  + "de un grafo engrosado, y para esa superficie χ = 2·(V − E) —un asa por cada ciclo "
  + "independiente del grafo—, de modo que"));
c.push(code("   Σ (6 − n) = 12 · (V − E)"));
c.push(p("queda fijado por el esqueleto antes de mallar nada. Esto detecta lo que la "
  + "comprobación de Euler sobre la propia malla no puede detectar: una malla que cerró "
  + "perfectamente alrededor del grafo equivocado. Una mezcla lo bastante ancha para fundir "
  + "dos puntales en uno, o una rejilla lo bastante gruesa para estrangular un cuello, produce "
  + "exactamente eso: una superficie impecable del género equivocado."));
c.push(spacer(80));
c.push(table(["Superred", "V", "E", "12·(V−E) exigido", "Σ(6−n) medido", "Átomos"],
  [["Super-grafeno", "4", "6", "−24", "−24 ✔", "1 180"],
   ["Super-cúbica", "1", "3", "−24", "−24 ✔", "836"],
   ["Super-hipercubo (4-cubo)", "16", "32", "−192", "−192 ✔", "6 494"],
   ["Jaula icosaédrica", "12", "30", "−216", "−216 ✔", "4 292"],
   ["Super-fcc", "4", "24", "−240", "−240 ✔", "2 982"],
   ["Superfulereno C₆₀", "60", "90", "−360", "−360 ✔", "7 534"]],
  [2500, 750, 750, 1900, 1900, 1560]));
c.push(spacer(120));
c.push(rich([{ t: "Seis de seis, exactamente, sobre un rango de −24 a −360", b: true },
  { t: ". Ninguna de esas cifras se le dijo al constructor: cada una es el censo de anillos "
     + "que salió del remallado, y cada una coincide con lo que el esqueleto exigía antes de "
     + "que hubiera geometría. Reproduce además las dos constantes que el módulo de redes "
     + "traía escritas a mano: cúbica 12·(1−3) = −24 y diamante 12·(8−16) = −96." }]));
c.push(spacer(100));
c.push(p("Las schwarzitas admiten la misma lectura por el género de su superficie: la Schwarz "
  + "P primitiva da Σ(6−n) = −24, que es 6χ con χ = −4 (género 3), y el giroide da −48, es "
  + "decir χ = −8 (género 5). La red cúbica de nanotubos da −24, como su superred homóloga.",
  { italics: true }));

c.push(h2("3.3 Dónde va cada disclinación: Gauss-Bonnet discreto"));
c.push(p("Que el censo global sea correcto no garantiza que cada disclinación esté donde la "
  + "curvatura la pide. El déficit angular"));
c.push(code("   K(v) = 2π − Σ θ"));
c.push(p("es la curvatura gaussiana discreta en un vértice, y es geométrica y no "
  + "combinatoria: un vértice de grado 5 sobre una lámina plana tiene cinco ángulos de 72° que "
  + "suman exactamente 2π, de modo que su déficit es cero. Eso es lo que lo hace utilizable "
  + "como objetivo: dice lo que hace la superficie, no lo que hace la malla. Sobre una región,"));
c.push(code("   Σ (6 − grado) = (3/π) · ∫ K dA"));
c.push(p("de modo que la parte que toca a un vértice es 3·K(v)/π. Sumado sobre una malla "
  + "cerrada devuelve 6χ exactamente: medido, 11.951, 11.943 y 11.865 frente al valor exacto "
  + "12. Una malla que igualara sus objetivos en todas partes satisfaría automáticamente la "
  + "comprobación de Euler."));
c.push(h3("La advertencia que conviene declarar"));
c.push(cita("La carga debe compararse sobre un entorno, nunca por vértice. La curvatura está "
  + "repartida sobre un área; una disclinación es un punto. Los objetivos por vértice valen "
  + "todos menos de 0.2 frente a un exceso de grado de ±1, de modo que un heptágono «cuesta» "
  + "1.045 en cualquier posición —una razón de 2067 a 1— y el criterio se vuelve ciego a la "
  + "posición. La comparación se hace por tanto sobre vecindarios de dos anillos."));
c.push(p("El descenso es codicioso y recalcula el objetivo en cada ronda. Dos versiones "
  + "incrementales fracasaron de forma instructiva y merecen una línea en el artículo, porque "
  + "son el tipo de error que no da excepción sino un resultado plausible: los vecindarios "
  + "quedan obsoletos entre barridos (el desajuste subió de 355 a 418, peor que no hacer nada) "
  + "y también dentro de un mismo barrido (45 volteos aceptados, todos «mejorantes», y el "
  + "barrido en conjunto peor). Se añadió además una restricción que impide crear "
  + "disclinaciones nuevas: sin ella el algoritmo persigue curvatura sub-cuántica y fabrica "
  + "pares 5-7 espurios."));

c.push(h2("3.4 El origen de los doce, y por qué el radio no interviene"));
c.push(p("En un toro, K dA = cos φ dφ dθ: los radios se cancelan. La mitad exterior integra "
  + "exactamente 4π para cualquier R y cualquier r, lo que da un presupuesto de disclinaciones "
  + "de exactamente +12 fuera y −12 dentro, con independencia del tamaño."));
c.push(p("Se consigna porque la alternativa intuitiva —hacerlo escalar con 4πr/e— fue ensayada "
  + "y da 17 o 27, valores que no corresponden a nada.", { italics: true }));

/* ============ 4 ============ */
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(h1("4. Catálogo de construcciones"));

c.push(h2("4.1 Familias tejidas directamente"));
if (hay("fig2_basicas.png")) c.push(...figura(F("fig2_basicas.png"),
  "Figura 2. Familias construidas tejiendo la red explícitamente. Naranja: pentágonos. "
  + "Azul: heptágonos. Los censos impresos son los del propio cálculo."));

c.push(h3("Nanotubos"));
c.push(p("Enrollado exacto por índices quirales (n, m): armchair, zigzag y quiral general. El "
  + "tubo resultante es abierto y periódico a lo largo de su eje, con la quiralidad exacta y "
  + "sin extremos. Admite la inserción de defectos —vacantes, divacantes, Stone-Wales— como "
  + "una lista de operaciones sobre la red ya tejida."));
c.push(p("Para un tubo cerrado existe un constructor aparte, porque un tubo con tapas es un "
  + "fulereno alargado y no un cilindro: cuerpo cilíndrico recto o suavemente curvado, "
  + "terminado en ambos extremos por domos hemisféricos, con pentágonos, heptágonos y "
  + "octágonos compuestos a petición. Se imponen dos invariantes, ambos comprobados por el "
  + "conjunto de pruebas: la topología es consistente con Euler por construcción, y la "
  + "geometría relajada tiene enlaces a milésimas de 1.42 Å."));

c.push(h3("Fulerenos y cebollas"));
c.push(p("Por subdivisión geodésica de un poliedro semilla y dualización, de modo que los doce "
  + "pentágonos no se colocan: son el precio que Euler cobra por cerrar la superficie. Las "
  + "cebollas apilan capas concéntricas con separación grafítica; el censo total es doce "
  + "pentágonos por capa, y medir 36 en una cebolla de tres capas es la comprobación."));

c.push(h3("Nanoconos: una disclinación y el ángulo que fuerza"));
c.push(p("Un cono de carbono se obtiene cortando un sector de una lámina y cosiendo los "
  + "bordes. Retirar k sectores de 60° deja k pentágonos de disclinación en el ápice y fija el "
  + "semiángulo de apertura; ésos son los cinco ángulos que observaron Krishnan y colaboradores."));
c.push(rich([{ t: "El constructor rechaza lo que la naturaleza no hace. ", b: true },
  { t: "Una disclinación de 5×60° convertiría el ápice en un «monógono», que no es un anillo; "
     + "y 3×60° da un anillo de tres miembros cuyos enlaces salen a 1.230 Å —topología "
     + "correcta, química mala—. La naturaleza no hace eso: reparte la disclinación en "
     + "pentágonos separados alrededor de la punta, que es como funciona el nanocuerno de "
     + "19.2°, y eso es un diseño de tapa y no un corte de sector. El constructor lo dice en "
     + "el mensaje de error en lugar de devolver la estructura imposible." }]));

c.push(h3("Haeckelitas: la rotación de Stone-Wales, sobre el dual"));
c.push(p("Una haeckelita es el panal del grafeno con pentágonos y heptágonos teselados "
  + "periódicamente. Las tres originales de Terrones —R5,7, H5,6,7 y O5,6,7— son tres puntos "
  + "de un espacio, y el módulo permite moverse por ese espacio en lugar de elegir de una lista."));
c.push(rich([{ t: "El detalle que define el módulo: ", b: true },
  { t: "la rotación de Stone-Wales se aplica a la malla, no al grafo de enlaces. Rotar un "
     + "enlace convierte los cuatro anillos que lo rodean en dos pentágonos y dos heptágonos, "
     + "y es tentador hacerlo directamente sobre los enlaces de los átomos: se intercambia un "
     + "vecino entre los dos átomos y todos los grados siguen valiendo tres. Se intentó así "
     + "primero y está mal. Un movimiento de Stone-Wales es un movimiento sobre un grafo "
     + "encajado, y hecho como recableado desnudo produce en silencio un grafo que ya no "
     + "encaja en el toro. El síntoma no es una excepción: la topología parece perfecta "
     + "—3-regular, número de enlaces correcto— y la relajación devuelve una lámina con "
     + "enlaces de 2.3 Å que ningún censo de anillos puede describir." }]));
c.push(p("El trabajo ocurre por tanto un nivel más abajo, sobre el dual triangulado, igual que "
  + "para las cáscaras cerradas: cada vértice de malla es un anillo, cada triángulo un átomo, "
  + "cada arista un enlace. El constructor exige además que las dimensiones sean múltiplos del "
  + "bloque que tesela cada patrón (4×4 para la R5,7), porque un bloque parcial deja parte de "
  + "la lámina hexagonal, y para esta red eso no es una aproximación sino un material distinto."));

c.push(h3("Heptaneno: cuando Euler decide la existencia antes que la geometría"));
c.push(p("Este caso merece un párrafo propio en el artículo porque enseña el método en su "
  + "forma más limpia. La pregunta no es «cómo lo construyo» sino «puede existir», y la "
  + "responde el mismo presupuesto de Euler antes de calcular ninguna geometría. Una red de "
  + "puros heptágonos aporta −1 por cara, de modo que F heptágonos fuerzan χ = −F/6. Esa sola "
  + "línea resuelve las tres geometrías a la vez:"));
c.push(bullet("Elíptica. Una esfera, o cualquier poliedro trivalente cerrado, tiene χ = +2 y "
  + "necesitaría F = −12 heptágonos. Un número negativo de caras no es un caso difícil: es una "
  + "contradicción. Imposible. (Es el mismo presupuesto que da sus doce pentágonos al "
  + "fulereno: la curvatura positiva se paga en caras menores que seis, y un heptágono tiene "
  + "el signo contrario.)"));
c.push(bullet("Euclídea. Una lámina periódica es un toro, χ = 0, luego F = 0. La única red "
  + "euclídea de puros heptágonos es la que no tiene ninguno. Imposible, y ésta es exactamente "
  + "la razón por la que una haeckelita debe emparejar cada heptágono que introduce."));
c.push(bullet("Hiperbólica. χ < 0, que es el caso que sí admite solución: una superficie de "
  + "curvatura negativa, es decir una esponja periódica, no una lámina."));

c.push(h3("Toroides"));
c.push(p("Un toroide es la estructura sobre la que la regla de disclinaciones se puede ver "
  + "directamente, y por eso es la mejor figura de validación del artículo. Es de género 1, de "
  + "modo que Σ(6−n) = 0: cada pentágono emparejado con un heptágono, exactamente, y ningún "
  + "otro tamaño hace falta. Su ecuador exterior tiene curvatura gaussiana positiva y el "
  + "interior negativa, así que los pentágonos van fuera y los heptágonos dentro, y no hay "
  + "otro sitio donde puedan ir."));
if (hay("fig3_toroide.png")) c.push(...figura(F("fig3_toroide.png"),
  "Figura 3. Toroide mallado, R = 20 Å y r = 5 Å, visto desde arriba. Izquierda: sólo los "
  + "pentágonos, que bordean el perímetro exterior. Derecha: sólo los heptágonos, que rodean "
  + "el agujero. Nadie coloca ninguno."));
c.push(p("Hay dos rutas al toroide y conviene distinguirlas. El toroide polihex se teje "
  + "directamente —un tubo (n,m) curvado sobre un número entero de periodos— y sale de "
  + "hexágonos puros, con Σ(6−n) = 0 por no tener ninguna disclinación. El toroide mallado "
  + "pasa por la ruta implícita y sale con pares 5-7, también con Σ(6−n) = 0 pero por "
  + "cancelación. Ambos son correctos y describen objetos distintos."));
c.push(rich([{ t: "La relación de aspecto es toda la física aquí. ", b: true },
  { t: "Un toro de radio mayor R y menor r curva su pared en r/R en el ecuador interior, de "
     + "modo que un toro grueso alrededor de un agujero pequeño está tensionado por mucho que "
     + "se relaje; y por debajo de R = 2r el agujero se cierra del todo y la forma es una "
     + "esfera con un hoyuelo. Los carbonos toroidales publicados están cerca de R/r entre 3 y "
     + "6; el constructor se niega por debajo de 2.5 y lo explica." }]));

c.push(h2("4.2 Familias nacidas de un campo implícito"));
if (hay("fig5_implicitas.png")) c.push(...figura(F("fig5_implicitas.png"),
  "Figura 4. Familias de la ruta implícita. Ninguna se arma pegando fragmentos: se define "
  + "f(x), se malla su nivel cero y el censo de anillos sale de la curvatura."));

c.push(h3("Uniones multiterminal"));
c.push(p("L (2 brazos a 90°), T (3 brazos), Y (3 brazos a 120°), X (4 brazos) y una unión 3D "
  + "de 6 brazos a lo largo de ±x, ±y, ±z. El radio de mezcla es el parámetro con significado "
  + "físico: más ancho da un nodo más redondo y suavemente curvado, y demasiado estrecho "
  + "reintroduce la arista viva que el panal no puede teselar."));
c.push(p("Nadie dice que un nodo de tres vías necesita heptágonos. Un nodo es una silla, una "
  + "silla lleva curvatura gaussiana negativa, y la curvatura negativa sale del remallador "
  + "como vértices de grado 7 y 8, que el dual convierte en heptágonos y octágonos."));

c.push(h3("Schwarzitas"));
c.push(p("Superficies mínimas triplemente periódicas —Schwarz P, Schwarz D y el giroide— en "
  + "sus aproximaciones trigonométricas estándar. Son el templado clásico del «carbono de "
  + "curvatura negativa»: donde un fulereno cierra con doce pentágonos, una schwarzita se abre "
  + "en una esponja periódica cuyos puntos de silla se teselan con heptágonos y octágonos."));

c.push(h3("Bobinas"));
c.push(p("Una nanobobina de carbono es un nanotubo helicoidal. La literatura ofrece dos rutas "
  + "y el paquete implementa la geométrica: un tubo recto se aplica sobre una trayectoria "
  + "helicoidal, conservando la sección local, de modo que los enlaces se mantienen cerca de "
  + "su valor de equilibrio mientras el radio de la bobina sea grande frente al del tubo. La "
  + "alternativa topológica (Dunlap, Ihara) sostiene la curvatura intrínseca con un patrón "
  + "regular de pares 5-7, y está disponible como inserción posterior de defectos "
  + "Stone-Wales a densidad controlable."));
c.push(p("El constructor conoce la banda que la literatura reporta para bobinas monocapa "
  + "relajadas —Popović y colaboradores encuentran D/d ≈ 3.5 en sus dos clases; Liu y "
  + "colaboradores dan una tabla compatible— y avisa cuando los parámetros pedidos caen fuera "
  + "de ella."));

c.push(h3("Redes y superredes de nanotubos"));
c.push(p("Aquí el esqueleto es un grafo con un encaje: un conjunto de vértices, un conjunto de "
  + "aristas y, cuando el objeto es periódico, las imágenes de celda que esas aristas "
  + "alcanzan. Cada arista se convierte en una cápsula del radio de tubo elegido; las cápsulas "
  + "se unen con unión suave, de modo que cada vértice adquiere curvatura real en lugar de un "
  + "pliegue; se malla el nivel cero, se remalla y se dualiza."));
c.push(p("Ésa es la jerarquía de Romo-Herrera, Terrones, Terrones, Dag y Meunier: tomar un "
  + "bloque unidimensional, usar operaciones de grupo puntual para formar un nodo "
  + "multiterminal, y después usar el nodo como nuevo bloque constructivo dejando que las "
  + "traslaciones generen la arquitectura. Super-grafeno, super-cuadrada, super-cúbica y "
  + "super-diamante son las cuatro que construyen los autores. Los mismos dos pasos generan un "
  + "superfulereno o una jaula icosaédrica, porque nada en ellos es específico de un cristal: "
  + "cualquier estructura de carbono terminada puede convertirse en el esqueleto de una mayor."));
if (hay("fig4_superredes.png")) c.push(...figura(F("fig4_superredes.png"),
  "Figura 5. La jerarquía de superredes: cada arista del esqueleto es un nanotubo y cada "
  + "vértice una unión multiterminal. El censo de anillos de cada una queda fijado por "
  + "Σ(6−n) = 12·(V−E) antes de mallar nada."));

c.push(h3("Espumas desordenadas"));
c.push(p("Construcción deliberadamente estocástica y pre-relajación: se tesela el interior de "
  + "una caja con escamas grafíticas pequeñas en posiciones y orientaciones aleatorias, "
  + "rechazando las que acerquen átomos de escamas distintas por debajo de una distancia "
  + "mínima. El resultado es una red de carbono tridimensional de baja densidad y "
  + "topológicamente desordenada."));
c.push(rich([{ t: "Conviene declarar su límite: ", b: true },
  { t: "no es una schwarzita completamente enlazada. No se intenta saturar los enlaces "
     + "colgantes, y el objeto está pensado como punto de partida para un recocido de dinámica "
     + "molecular o una optimización guiada por aprendizaje automático, no como estructura "
     + "final." }]));

c.push(h2("4.3 Coordenadas topológicas: posiciones a partir de los enlaces"));
c.push(p("Muy a menudo lo único que se conoce de una nanoestructura es qué átomo está enlazado "
  + "con cuál. La revisión de István László recoge los métodos que convierten eso en "
  + "coordenadas cartesianas, y la idea detrás de todos es la misma: ciertos vectores propios "
  + "de la matriz de adyacencia son bilobulares —al borrar los vértices donde se anulan y las "
  + "aristas donde cambian de signo quedan exactamente dos componentes— y se comportan como "
  + "las primeras ondas estacionarias sobre la superficie, de modo que pueden leerse como "
  + "ángulos."));
c.push(bullet("Esférico (Fowler y Manolopoulos; Pisanski y Shawe-Taylor): tres vectores "
  + "propios bilobulares, escalados por 1/√(λ₁ − λₖ), son las x, y, z de un fulereno. Para la "
  + "mayoría de los fulerenos son simplemente los vectores 2, 3 y 4."));
c.push(bullet("Toroidal (László y colaboradores): tres no bastan —Graovac y colaboradores "
  + "demostraron que el toro sale plano desde alguna dirección— y cuatro sí. Un toro es el "
  + "producto de dos circunferencias, de modo que los cuatro se dividen en dos pares "
  + "degenerados: uno da el ángulo alrededor del anillo y el otro el ángulo alrededor del tubo."));
c.push(p("Cuáles cuatro se midió, no se supuso. Tomando un polihexágono toroidal que el "
  + "paquete construye exactamente —un tubo (5,5) curvado sobre 60 periodos, 1200 átomos, todo "
  + "hexágonos—, descartando sus coordenadas y conservando sólo sus enlaces, los vectores "
  + "propios bilobulares en orden decreciente de valor propio son 1, 2, 13 y 14.", { italics: true }));

/* ============ 5 ============ */
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(h1("5. La ruta de rodillas: un censo exacto por construcción"));

c.push(p("Las secciones 2 a 4 describen la ruta implícita: se muestrea un campo, se "
  + "extrae una isosuperficie, se remalla y se toma el dual. Es general —teje cualquier "
  + "superficie que se sepa escribir— y por eso mismo no controla dónde caen sus "
  + "disclinaciones: el remallado coloca pentágonos y heptágonos donde le conviene a la "
  + "malla, no donde los pide la curvatura. Medido sobre una unión Y, la ruta implícita "
  + "devuelve cincuenta pentágonos y treinta y ocho heptágonos; Gauss-Bonnet exige seis "
  + "heptágonos y ningún pentágono."));

c.push(p("La ruta de rodillas ataca el problema por el otro extremo. En lugar de tejer una "
  + "superficie y contar después lo que salió, ensambla la estructura a partir de piezas "
  + "cuyo censo se conoce antes de existir: tramos rectos de tubo, todo hexágonos, unidos "
  + "por codos a inglete. El censo no se comprueba al final —se fija al principio—, y la "
  + "comprobación posterior sirve para detectar un error de programación, no para "
  + "descubrir qué se construyó."));

c.push(h2("5.1 El codo, y por qué lleva dos pares"));
c.push(p("Un codo a inglete corta dos tramos de tubo en el plano bisector y los une. La "
  + "unión es una identidad, no un promedio: los dos tramos son imágenes especulares a "
  + "través de ese plano, así que una vez que sus bordes están en él coinciden punto por "
  + "punto. Esa es la diferencia entre aristas de malla de 2.46 Å contra un objetivo de "
  + "2.46 Å y aristas de 1.9 a 5.7 Å, que es lo que daba fundir los bordes crudos y de lo "
  + "que ninguna relajación se recupera."));
c.push(p("El codo es simétrico respecto del plano del toro, así que todo defecto que cree "
  + "tiene pareja al otro lado: un codo a inglete es un codo de Dunlap DOBLE, y lleva dos "
  + "pares pentágono-heptágono, no uno. No es una elección: buscando sobre número de "
  + "codos, circunferencia, longitud de brazo, desfase de costura y fase azimutal, la "
  + "familia de un solo par no aparece nunca."));
c.push(p("De ahí sale el número de codos de un anillo sin más aritmética. Un par gira el "
  + "eje 30°, un toro pide 360°, luego doce pares, luego seis codos — y doce pentágonos "
  + "fuera y doce heptágonos dentro, que es exactamente Σ(6−n) = 0 para χ = 0."));
c.push(cita("El desfase de la costura elige el codo, y es la única libertad real: girar un "
  + "lado una posición da dos pentágonos fuera y dos heptágonos dentro; no girarlo da un "
  + "CUADRADO fuera y un octógono dentro. Los dos cumplen Σ(6−n) = 0. El codo de "
  + "pentágonos es el predeterminado porque un anillo de cuatro miembros es carbono sp2 "
  + "pobre, no porque el otro esté mal."));

c.push(h2("5.2 El presupuesto de un nodo, y una imposibilidad"));
c.push(p("Un nodo de c brazos es topológicamente una esfera con c agujeros, así que "
  + "χ = 2 − c y Σ(6−n) = 6(2 − c). Sumado sobre una red entera, Σ_v 6(2 − grado v) = "
  + "12V − 12E = 12(V − E), que es el mismo número al que llega el otro camino, "
  + "χ = 2(V − E). Dos derivaciones independientes, un solo número; el generador calcula "
  + "las dos y compara en cada entrada del catálogo."));
c.push(p("La condición para que un nodo salga LIMPIO —sólo hexágonos más los heptágonos "
  + "que el presupuesto obliga— es que sus brazos sumen cero como vectores. Un nodo "
  + "desequilibrado paga el presupuesto igual de exacto, pero lo paga en cuadrados, "
  + "pentágonos o eneágonos."));
c.push(cita("De ahí se sigue que NO EXISTE una jaula finita de rodillas. En un poliedro "
  + "convexo todo vértice está en su envolvente, así que todas sus aristas apuntan hacia "
  + "el mismo semiespacio de apoyo y no pueden sumar cero; y todo grafo finito tiene un "
  + "vértice en su envolvente. Las jaulas de esta ruta son periódicas o tienen bordes; no "
  + "hay una tercera posibilidad, y no es una carencia del programa."));
c.push(p("Los brazos coplanares dejan además dos agujeros polares, uno arriba y otro "
  + "abajo, con una arista por brazo: tres brazos coplanares dejan triángulos —que la "
  + "malla ya cierra— y cuatro dejan cuadrados. Ésa es la razón real de que la X plana "
  + "tenga χ = −4 en lugar de −2: es una esfera con SEIS agujeros, las cuatro bocas más "
  + "los dos polos, no con cuatro."));

c.push(h2("5.3 Qué formas cierran"));
c.push(p("No toda terna (codos, circunferencia, filas de brazo) es exacta, porque la cuña "
  + "que el inglete quita tiene que ser un número entero de pasos de red. Las filas de "
  + "brazo han de ser impares, porque la reflexión del inglete manda la fila i a la fila "
  + "«filas − 1 − i». Las que cierran se encuentran construyendo y contando, y el rechazo "
  + "nombra las que sí lo hacen en vez de limitarse a negarse — que es la diferencia entre "
  + "un parámetro y una adivinanza."));
c.push(p("Cada tipo de nodo tiene su propia forma: un nodo Y cierra en circunferencia 14 y "
  + "uno tetraédrico no cierra ahí en absoluto. No es un matiz: llevar la forma de otro "
  + "tipo no es un error pequeño, o se niega de plano o construye una estructura distinta "
  + "de la que se pidió."));

c.push(h2("5.4 El catálogo medido"));
c.push(p("Todas las filas de la tabla salen de ejecutar el verificador del paquete sobre "
  + "esta misma versión, no de la documentación. «Ley» dice si el Σ(6−n) medido coincide "
  + "con el presupuesto que el esqueleto fija ANTES de mallar nada; «colocación» es la "
  + "fracción de disclinaciones que caen del lado de la curvatura que les toca, medida "
  + "ajustando la superficie sobre cada anillo y no leída del tamaño del anillo."));
c.push(table(
  ["estructura", "átomos", "censo", "Σ(6−n)", "ley", "enlaces (Å)", "colocación"],
  [
    ["toroide, 6 codos", "1032", "5:12 6:492 7:12", "0", "sí", "1.398–1.453", "100 %"],
    ["bobina, finita", "406", "5:14 6:167 7:14", "0", "—", "1.367–1.499", "100 %"],
    ["bobina, celda periódica", "672", "5:12 6:312 7:12", "0", "sí", "1.376–1.495", "100 %"],
    ["unión Y", "536", "6:240 7:6", "−6", "sí", "1.410–1.431", "100 %"],
    ["unión X (plana)", "780", "5:4 6:328 7:16", "−12", "sí", "1.405–1.450", "100 %"],
    ["unión, nodo de diamante", "756", "6:328 7:12", "−12", "sí", "1.415–1.428", "100 %"],
    ["lámina, super-cuadrada", "180", "5:4 6:68 7:16", "−12", "sí", "1.374–1.505", "100 %"],
    ["lámina, super-grafeno", "920", "6:432 7:24", "−24", "sí", "1.408–1.436", "100 %"],
    ["celda de Schwarz P", "968", "6:456 7:24", "−24", "sí", "1.417–1.540", "—"],
    ["celda de Schwarz D", "1440", "6:608 7:96", "−96", "sí", "1.353–1.452", "—"],
    ["celda del giroide", "1744", "6:816 7:48", "−48", "sí", "1.395–1.508", "—"],
  ],
  [2100, 900, 2000, 800, 700, 1500, 1000]));
c.push(spacer(120));
c.push(p("Tres lecturas de esa tabla merecen texto propio."));
c.push(bullet("Las superficies mínimas no tienen NI UN pentágono, que es lo que debe "
  + "ocurrir: toda su curvatura es negativa. La ruta implícita devuelve treinta y tres "
  + "pentágonos en una celda comparable."));
c.push(bullet("Los pentágonos de la X plana y de la lámina super-cuadrada sí están bien: "
  + "un cruce plano de cuatro brazos tiene una almohadilla arriba y otra abajo del punto "
  + "de cruce que está genuinamente curvada en positivo. Son cuatro, y son los polos."));
c.push(bullet("La celda periódica de la bobina es un TORO —el tubo se cierra sobre sí "
  + "mismo a través de la celda—, así que su Σ(6−n) es 0 como el del anillo, con D/d = "
  + "3.52, dentro de la banda 3.5–3.9 que Popović y Liu publican para bobinas monocapa."));

if (hay("fig6_rodillas.png")) c.push(...figura(F("fig6_rodillas.png"),
  "Figura 6. Nueve estructuras de la ruta de rodillas, renderizadas del paquete. Cada "
  + "panel imprime el Σ(6−n) MEDIDO sobre la estructura que muestra, no el esperado: la "
  + "figura y la tabla anterior salen del mismo cálculo y no pueden discrepar."));

c.push(h2("5.5 Percibir los anillos de una superficie abierta"));
c.push(p("Contar los anillos de una estructura terminada no es trivial y tiene un modo de "
  + "fallo que no se anuncia. Sobre una superficie teselada, los anillos se obtienen "
  + "trazando CARAS —recorriendo el borde de cada polígono— y no por caminos mínimos "
  + "entre vecinos: el camino mínimo omite sistemáticamente los anillos grandes, porque un "
  + "heptágono cuyas siete aristas lindan además con hexágonos nunca es el circuito más "
  + "corto que cierra por ninguna de ellas."));
c.push(cita("Medido: sobre la unión X, el camino mínimo devuelve {5: 4, 6: 328} y un "
  + "Σ(6−n) de +4; el trazado de caras devuelve {5: 4, 6: 328, 7: 16} y −12. Los dieciséis "
  + "heptágonos no se pierden con ruido, se pierden enteros, y el censo resultante sigue "
  + "pareciendo un censo."));
c.push(p("Qué estructuras admiten el trazado de caras se decide por una pregunta y no por "
  + "una tolerancia: ningún átomo puede tener más de tres vecinos, y al menos uno debe "
  + "tener tres. Grado uno o dos es borde, que una superficie abierta tiene con todo "
  + "derecho; grado cuatro o más no tesela superficie ninguna. Un umbral porcentual sobre "
  + "la trivalencia decide casos reales por su margen —la X plana se queda en 89.74 % y un "
  + "nanocono en 89.36 %, los dos definidos por tener borde—, y un cono y una cinta caían "
  + "al camino mínimo en silencio."));
c.push(p("Queda un caso que no se puede resolver, y se informa en lugar de taparse: el "
  + "trazado distingue una cara de un borde por TAMAÑO, lo cual es exacto sólo mientras el "
  + "borde sea más largo que un anillo. En una cinta de seis por tres, los dos bordes miden "
  + "seis átomos —el tamaño de un hexágono— y se cuentan como anillos. El informe lo dice y "
  + "marca el resultado como no fiable, en vez de restar dos caras sin saber cuáles."));

/* ============ 6 ============ */
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(h1("6. Validación"));

c.push(h2("6.1 Hibridación medida desde la geometría"));
c.push(p("La fracción sp3 se mide directamente de la suma de los tres ángulos de enlace de "
  + "cada carbono tricoordinado: 360° cuando es plano (sp2 ideal, como en grafeno) y 328.4° "
  + "cuando es tetraédrico (sp3 ideal, tres ángulos de 109.47° con el cuarto enlace fuera de "
  + "la red). La fracción"));
c.push(code("   carácter sp3 = (360 − Σθ) / (360 − 328.4)"));
c.push(p("va de 0 para plano a 1 para tetraédrico. Es la medida directa de lo que una razón "
  + "D/G de Raman o una asimetría del C1s en XPS estiman de forma indirecta."));
c.push(h3("La distinción que hay que declarar"));
c.push(cita("Ésta es la misma suma angular que aparece en el análisis de curvatura, y los dos "
  + "usos se parecen sin ser lo mismo. El déficit 2π − Σθ nunca es negativo en un vértice "
  + "trivalente —la desigualdad del triángulo esférico lo prohíbe—, de modo que no puede "
  + "distinguir un casquete de un cuello y es inútil como signo de curvatura. Lo que sí mide "
  + "fielmente es cuánto se ha piramidalizado el carbono, y eso es hibridación."));
c.push(p("De ahí se sigue algo que conviene decir explícitamente en el artículo: una pared de "
  + "curvatura negativa no aparece como sp3, y no debe aparecer. La curvatura vive en el censo "
  + "de anillos; la hibridación, en los ángulos."));
c.push(p("Controles: una haeckelita plana, llena de pentágonos y heptágonos, mide exactamente "
  + "0.00 de carácter sp3; un tubo (5,5) prístino mide 3.390 ± 0.000."));

c.push(h2("6.2 Un criterio de imposibilidad, no de calidad"));
c.push(p("Una suma angular por debajo de 328.4° está más allá de lo tetraédrico, que ningún "
  + "carbono alcanza. No es una estructura peor: es una que no puede existir. El constructor "
  + "no la devuelve, sino que la reconstruye sujetando la pared a su propia superficie o "
  + "afinando la rejilla, y escala el remedio hasta que la suma angular mínima supera el "
  + "umbral con margen."));
c.push(p("El campo implícito vale cero sobre la pared, de modo que |f(átomo)| es la desviación "
  + "exacta fuera de superficie, sin aproximación alguna, y sirve como métrica de calidad."));

c.push(h2("6.3 Dos métricas, y por qué hay que reportar las dos"));
c.push(rich([{ t: "Esta sección corrige una versión anterior de este documento", b: true },
  { t: ", y la corrección es en sí misma el punto metodológico más útil que tiene." }]));
c.push(p(
  "El paquete juzga una estructura por dos criterios independientes. El primero es "
  + "geométrico: validation.sp2_quality exige que toda longitud de enlace caiga en "
  + "1.30–1.55 Å y todo ángulo en 100–135°, y clasifica en «clean», «strained» o «broken». "
  + "El segundo es de hibridación: una suma angular por debajo de 328.4° está más allá de lo "
  + "tetraédrico y describe una pared imposible."));
c.push(p(
  "Una auditoría del catálogo reportó sólo el segundo y concluyó que ocho de ocho "
  + "estructuras estaban sanas. Medidas contra el primero, cuatro de seis estaban «broken» y "
  + "ninguna «clean». Las dos cosas eran ciertas a la vez, porque no son la misma pregunta."));
c.push(spacer(80));
c.push(table(["Superred", "Enlaces (Å)", "sp2_quality", "Suma angular mín.", "Pared"],
  [["Super-grafeno", "1.344 – 1.544", "strained", "335.1°", "sana"],
   ["Jaula icosaédrica", "1.326 – 1.525", "strained", "331.4°", "sana"],
   ["Super-cúbica", "1.295 – 1.546", "broken", "330.6°", "sana"],
   ["Super-fcc", "1.301 – 1.575", "broken", "334.0°", "sana"],
   ["Super-hipercubo", "1.316 – 1.574", "broken", "331.6°", "sana"],
   ["Superfulereno C₆₀", "1.281 – 1.644", "broken", "333.0°", "sana"]],
  [2300, 2100, 1700, 1900, 1360]));
c.push(spacer(120));
c.push(h3("La distribución dice más que el extremo"));
c.push(p(
  "sp2_quality usa el mínimo y el máximo, que sobre miles de enlaces son estadísticos de "
  + "valor extremo. La distribución completa separa dos situaciones muy distintas:"));
c.push(spacer(80));
c.push(table(["Superred", "Enlaces", "Media (Å)", "p99 (Å)", "Máx (Å)", "Fuera de ventana"],
  [["Super-grafeno", "1 770", "1.420", "1.493", "1.544", "0"],
   ["Jaula icosaédrica", "6 438", "1.420", "1.483", "1.525", "0"],
   ["Super-hipercubo", "9 741", "1.420", "1.488", "1.574", "1"],
   ["Super-cúbica", "1 254", "1.420", "1.508", "1.546", "1"],
   ["Super-fcc", "4 473", "1.420", "1.511", "1.575", "4"],
   ["Superfulereno C₆₀", "11 301", "1.432", "1.551", "1.644", "128"]],
  [2200, 1300, 1400, 1300, 1300, 1860]));
c.push(spacer(120));
c.push(p(
  "Cinco de seis tienen la media exactamente en el valor objetivo de 1.420 Å y entre cero y "
  + "cuatro enlaces fuera de la ventana: el hipercubo se clasifica «broken» por UN enlace "
  + "entre 9 741. El superfulereno es el único genuinamente dañado, con la media desplazada y "
  + "128 enlaces fuera."));
c.push(h3("La causa, y lo que enseña"));
c.push(cita("Optimizar una métrica y después auditar con esa misma métrica no es una "
  + "comprobación. El superfulereno se reconstruía sujetando la pared a su superficie para "
  + "subir la suma angular, y eso pelea con el campo de fuerzas que lleva los enlaces a "
  + "1.42 Å. Sin esa reconstrucción tenía UN átomo de 7 534 a 328.33° —siete centésimas de "
  + "grado por debajo del umbral— y CERO enlaces fuera de ventana. Con ella, el átomo quedaba "
  + "corregido y 128 enlaces rotos, el peor a 1.644 Å."));
c.push(p(
  "Es el mismo error que este paquete ya había documentado para la colocación de "
  + "disclinaciones —la carga debe compararse sobre un entorno, nunca por vértice— repetido "
  + "en un detector que mira el mínimo sobre miles de átomos. Un átomo pasado de tetraédrico "
  + "es un defecto local; una pared colapsada es una región doblada.", { italics: true }));
c.push(p(
  "Comprobación independiente: reproduciendo el colapso real de una bobina (desactivando el "
  + "límite de longitud en el volteo de aristas, que es el fallo que lo causaba) la estructura "
  + "sale con 103 enlaces de 1 122 fuera de ventana y el peor a 1.696 Å, y el detector de "
  + "suma angular reporta CERO átomos bajo el umbral. No detecta el colapso real; la longitud "
  + "de enlace sí."));
c.push(p(
  "El remedio aplicado: la reconstrucción cuenta ahora los enlaces fuera de ventana antes y "
  + "después y descarta toda variante que empeore ese número. El superfulereno pasa de "
  + "1.281–1.644 Å con 128 enlaces fuera a 1.306–1.538 Å con cero."));

c.push(h2("6.4 Un censo correcto puede esconder dislocaciones"));
c.push(p(
  "El presupuesto de Euler fija la suma Σ(6−n), no cada anillo por separado, y un par "
  + "pentágono-heptágono aporta exactamente cero a esa suma. De ahí que una estructura pueda "
  + "cumplir su presupuesto al dígito y estar llena de pares 5-7 que ninguna curvatura pide: "
  + "se cancelan entre sí y la contabilidad no los ve."));
c.push(p(
  "Son dislocaciones de Stone-Wales. Sobre un tubo recto, cuya curvatura gaussiana es cero, "
  + "no hay nada que las justifique. Contadas en el catálogo:"));
c.push(spacer(80));
c.push(table(["Superred", "5", "6", "7", "8", "Σ(6−n)", "Pares 5-7 que se cancelan", "% de los no hexagonales"],
  [["Super-grafeno", "42", "480", "62", "2", "−24", "42", "79.2 %"],
   ["Super-hipercubo", "254", "2 539", "398", "24", "−192", "254", "75.1 %"],
   ["Super-cúbica", "33", "326", "53", "2", "−24", "33", "75.0 %"],
   ["Jaula icosaédrica", "129", "1 666", "285", "30", "−216", "129", "58.1 %"],
   ["Superfulereno C₆₀", "196", "2 981", "504", "26", "−360", "196", "54.0 %"],
   ["Super-fcc", "84", "1 089", "234", "42", "−240", "84", "46.4 %"]],
  [1750, 650, 800, 700, 620, 900, 1900, 2040]));
c.push(spacer(120));
c.push(p(
  "Entre el 46 % y el 79 % de todos los pentágonos y heptágonos no aportan nada a la "
  + "topología. La colocación por curvatura los reduce a un tercio —de 42 a 14 en "
  + "super-grafeno, de 33 a 16 en super-cúbica— y de paso mejora la geometría, llevando el "
  + "super-grafeno de «strained» a «clean» y el acuerdo con el signo de la curvatura del 93 % "
  + "al 100 %."));
c.push(rich([{ t: "Lo que la colocación no puede hacer, y conviene declararlo: ", b: true },
  { t: "los pares que quedan son dislocaciones aisladas, y un volteo de arista no las "
     + "elimina, las mueve. Aniquilarlas exige juntar dos dislocaciones opuestas, que es un "
     + "problema de transporte y no una corrección local; el descenso codicioso se detiene en "
     + "ellas porque ningún volteo suelto mejora la desviación local. La configuración "
     + "7-7-5-5 —un Stone-Wales completo— sí se deshace de un volteo, y la rutina de "
     + "remallado ya lo hace: los cuatro grados pasan a 6 y la desviación cae de 4 a 0." }]));
c.push(p(
  "Para un artículo esto es una limitación que conviene declarar antes que esconder, y "
  + "sugiere la métrica que falta: junto al censo y al déficit de Euler, informar cuántos "
  + "pares se cancelan. Es la diferencia entre una red que cumple su topología y una red "
  + "limpia.", { italics: true }));

c.push(h2("6.5 Conjunto de pruebas"));
c.push(p("1476 pruebas automatizadas (1 omitida por ausencia de tkinter en un entorno sin "
  + "pantalla). No comprueban únicamente que el código corra: fijan los resultados medidos y, "
  + "en varios casos, fijan también los criterios que se ensayaron y no funcionan, para que no "
  + "vuelvan a intentarse. Es un punto metodológico defendible: los contraejemplos están "
  + "versionados junto al código."));

/* ============ 6 ============ */
c.push(h1("7. Reproducibilidad"));
c.push(spacer(60));
c.push(table(["Elemento", "Valor"],
  [["Paquete", "nanocarbon_lab v0.2.0"],
   ["Python", "≥ 3.11"],
   ["Entorno y bloqueo de versiones", "uv (lockfile único para todo el workspace)"],
   ["Semilla aleatoria", "seed = 0 en todas las cifras reportadas"],
   ["Hilos de álgebra lineal", "OMP_NUM_THREADS = 1"],
   ["Longitud de arista objetivo", "1.42·√3 Å"],
   ["Parámetros por estructura", "escala de celda, radio de tubo, anchura de mezcla, "
    + "resolución de rejilla, iteraciones de remallado y de relajación"],
   ["Figuras de este documento", "docs/articulo/figuras.py y fig_*.py"]],
  [3200, 6160]));
c.push(spacer(140));
c.push(p("Ejecución del conjunto de pruebas:"));
c.push(code("OMP_NUM_THREADS=1 uv run python -m pytest nanocarbon_lab/tests \\"));
c.push(code("    -q -n 4 --dist loadscope -m \"not slow\""));
c.push(spacer(100));
c.push(p("Declare la semilla en el artículo. Los constructores son deterministas dada la "
  + "semilla, pero no entre versiones del remallador: un cambio en el criterio de volteo "
  + "modifica la malla y, con ella, el censo de anillos.", { italics: true }));

/* ============ 7 ============ */
c.push(h1("8. Librerías utilizadas"));
c.push(p("Todas son de código abierto y todas piden ser citadas en la bibliografía, no sólo en "
  + "los agradecimientos."));
c.push(spacer(60));
c.push(table(["Librería", "Versión mínima", "Papel en este trabajo"],
  [["NumPy", "1.24", "Aritmética de arreglos en todo el paquete: campos escalares, posiciones, mallas"],
   ["SciPy", "1.10", "Álgebra lineal dispersa, vectores propios para las coordenadas topológicas, búsquedas espaciales"],
   ["ASE (Atomic Simulation Environment)", "3.22", "Objeto Atoms, celdas, condiciones periódicas, entrada y salida de formatos atomísticos"],
   ["NetworkX", "3.0", "Grafos: esqueletos de superredes, ciclos, detección de anillos"],
   ["scikit-image", "0.22", "Marching cubes para la extracción del conjunto de nivel cero"],
   ["Matplotlib", "—", "Visualización 3D en la interfaz y las figuras de este documento"],
   ["pytest, pytest-xdist, pytest-cov", "7.4 / 3.5 / 4.1", "Conjunto de pruebas y ejecución en paralelo"],
   ["Ruff", "0.4", "Análisis estático"],
   ["uv", "—", "Resolución de dependencias y lockfile reproducible"]],
  [2500, 1400, 5460]));
c.push(spacer(120));
c.push(p("Nota sobre bpy (Blender): está declarado como conflictivo en la raíz del workspace "
  + "porque fija numpy 1.26 y arrastraría esa versión a todos los paquetes. Si el artículo "
  + "incluye figuras renderizadas con Blender, decláre­lo por separado.", { italics: true }));

/* ============ 8 ============ */
c.push(h1("9. Agradecimientos"));
c.push(h2("9.1 Fuentes metodológicas"));
c.push(p("Deben acreditarse en el cuerpo del artículo, no sólo en los agradecimientos:"));
c.push(bullet("Romo-Herrera, Terrones, Terrones, Dag y Meunier — Nano Letters 7 (2007) 570. "
  + "La jerarquía de superredes de nanotubos. La comprobación del censo de anillos reproduce "
  + "la que describe su Información de Apoyo S2."));
c.push(bullet("Terrones y col. — Physical Review Letters 84 (2000) 1716. Haeckelitas: redes "
  + "planas y tubulares de pentágonos, hexágonos y heptágonos."));
c.push(bullet("Krishnan y col. — Nature 388 (1997). Ángulos de apertura observados en "
  + "nanoconos de carbono, que fijan los casos construibles."));
c.push(bullet("László, I. — Theoretical Chemistry Accounts 134 (2015) 104. Revisión de "
  + "coordenadas topológicas a partir de la matriz de adyacencia."));
c.push(bullet("Fowler y Manolopoulos; Pisanski y Shawe-Taylor — caso esférico de las "
  + "coordenadas topológicas."));
c.push(bullet("Graovac y col. — demostración de que tres vectores propios no bastan para el toro."));
c.push(bullet("Popović y col. — Contemporary Materials III-1 (2012) 51; Liu y col. — "
  + "Nanoscale Research Letters 5 (2010) 478. Relaciones D/d de bobinas monocapa relajadas."));
c.push(bullet("Botsch, M. y Kobbelt, L. — esquema de remallado isotrópico (dividir, colapsar, "
  + "voltear, suavizar, reproyectar)."));
c.push(bullet("Lorensen, W. E. y Cline, H. E. — Computer Graphics 21 (1987) 163. Marching cubes."));
c.push(bullet("Dunlap, B. I. — criterio de disclinaciones en carbonos toroidales; y, con "
  + "Ihara, la ruta topológica a las bobinas. El codo a inglete de la sección 5 es un codo "
  + "de Dunlap doble, y el giro de 30° por par pentágono-heptágono es el ángulo del codo "
  + "publicado: ésa es la comprobación de que la ley empleada es la correcta."));
c.push(bullet("Lenosky, T.; Gonze, X.; Teter, M. y Elser, V. — Nature 355 (1992) 333. "
  + "Carbono grafítico de curvatura negativa sobre la superficie D. La celda primitiva de "
  + "D es la D216 de ese trabajo, de género 3; las celdas de la sección 5 son las "
  + "convencionales cúbicas, que son cuatro primitivas de la misma superficie."));
c.push(bullet("Stone, A. J. y Wales, D. J. — la rotación de enlace que genera pares 5-7."));

c.push(h2("9.2 Software"));
c.push(p("Formulación sugerida: «Este trabajo se apoya en el ecosistema científico de Python. "
  + "Los autores agradecen a las comunidades de NumPy, SciPy, ASE, NetworkX, scikit-image y "
  + "Matplotlib, cuyo software libre hace posible este tipo de trabajo.»"));

c.push(h2("9.3 Asistencia de inteligencia artificial"));
c.push(p("El desarrollo del generador se realizó con asistencia de Claude (Anthropic), "
  + "utilizado a través de Claude Code. Conviene declararlo, y en el lugar correcto."));
c.push(rich([{ t: "Sobre la autoría. ", b: true },
  { t: "Las políticas del ICMJE, del COPE y de las principales editoriales (Elsevier, Springer "
     + "Nature, Wiley, ACS) coinciden en que un sistema de IA no puede figurar como autor, "
     + "porque no puede asumir responsabilidad por el trabajo ni declarar conflictos de "
     + "interés. Su uso se declara en los agradecimientos o en una sección específica de "
     + "métodos. Verifique la redacción exacta que exige la revista: varias piden un formato "
     + "propio." }]));
c.push(spacer(80));
c.push(p("Texto sugerido:", { bold: true }));
c.push(cita("Los autores agradecen a Anthropic. El código de generación y análisis estructural "
  + "descrito en este trabajo se desarrolló con asistencia de Claude (Anthropic) a través de "
  + "Claude Code. La asistencia comprendió la implementación de los algoritmos, el diseño del "
  + "conjunto de pruebas y la documentación del código. El diseño de la investigación, la "
  + "elección de los métodos, la interpretación de los resultados y la verificación de toda "
  + "cifra reportada corresponden íntegramente a los autores, quienes asumen la "
  + "responsabilidad completa del contenido."));
c.push(p("Ajuste el alcance a lo que realmente ocurrió. Una declaración que exagere o minimice "
  + "el papel de la herramienta es más problemática que una precisa.",
  { italics: true, color: "7F2704" }));

/* ============ 9 ============ */
c.push(h1("10. Qué conviene destacar"));
c.push(p("Cuatro puntos son metodológicamente defendibles y distinguen al generador de una "
  + "colección de constructores:"));
c.push(bullet("La estadística de anillos es derivada, no impuesta. Eso permite enunciar la "
  + "comprobación de Euler como verificación independiente y no como tautología."));
c.push(bullet("Existe una segunda comprobación verdaderamente independiente para las "
  + "superredes: Σ(6−n) = 12·(V−E) queda fijado por el esqueleto antes de mallar, y detecta el "
  + "caso que la comprobación sobre la propia malla no puede detectar: una superficie "
  + "impecable alrededor del grafo equivocado."));
c.push(bullet("La validez física se comprueba con un criterio de imposibilidad, no de calidad: "
  + "una suma angular bajo 328.4° no es una estructura peor, es una que no puede existir."));
c.push(bullet("Euler decide la existencia antes que la geometría. El caso del heptaneno lo "
  + "muestra en su forma más limpia: tres geometrías resueltas por una sola línea de "
  + "contabilidad, dos de ellas declaradas imposibles sin calcular nada."));
c.push(bullet("Dos criterios independientes, y hay que reportar los dos. La topología (censo "
  + "y déficit de Euler) y la geometría (longitudes y ángulos) responden preguntas distintas, "
  + "y una estructura puede cumplir una y fallar la otra. Optimizar una métrica y auditar con "
  + "esa misma métrica no es una comprobación: es una tautología, y en este trabajo produjo "
  + "un catálogo declarado sano que tenía enlaces de 1.644 Å."));
c.push(spacer(140));
c.push(new Paragraph({ border: { top: { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" } },
                       spacing: { before: 140, after: 120 }, children: [] }));
c.push(p("Todas las cifras y todas las figuras de este documento proceden de ejecuciones "
  + "registradas del paquete en la versión indicada. Antes de publicarlas, vuelva a "
  + "ejecutarlas en su propio entorno y repórtelas desde esa ejecución.",
  { italics: true, color: "595959" }));

/* ============ DOCUMENTO ============ */
const doc = new Document({
  creator: "nanocarbon_lab",
  title: "Generación computacional de estructuras de nanocarbono",
  description: "Metodología detallada, catálogo de construcciones, validación y agradecimientos",
  numbering: { config: [{ reference: "vinetas", levels: [
    { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 460, hanging: 240 } } } },
    { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 900, hanging: 240 } } } }] }] },
  sections: [{ properties: { page: {
      size: { width: 12240, height: 15840, orientation: PageOrientation.PORTRAIT },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    children: c }],
});
Packer.toBuffer(doc).then(b => {
  const out = path.join(__dirname, "metodologia-nanocarbon.docx");
  fs.writeFileSync(out, b);
  console.log("escrito", out, (b.length / 1024).toFixed(0) + " KB");
});
