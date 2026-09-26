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
c.push(h1("5. Validación"));

c.push(h2("5.1 Hibridación medida desde la geometría"));
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

c.push(h2("5.2 Un criterio de imposibilidad, no de calidad"));
c.push(p("Una suma angular por debajo de 328.4° está más allá de lo tetraédrico, que ningún "
  + "carbono alcanza. No es una estructura peor: es una que no puede existir. El constructor "
  + "no la devuelve, sino que la reconstruye sujetando la pared a su propia superficie o "
  + "afinando la rejilla, y escala el remedio hasta que la suma angular mínima supera el "
  + "umbral con margen."));
c.push(p("El campo implícito vale cero sobre la pared, de modo que |f(átomo)| es la desviación "
  + "exacta fuera de superficie, sin aproximación alguna, y sirve como métrica de calidad."));

c.push(h2("5.3 Estado del catálogo de superestructuras"));
c.push(p("Auditoría completa, monohilo, semilla 0:"));
c.push(spacer(80));
c.push(table(["Preset", "Átomos", "Suma angular mín.", "Veredicto", "Remedio", "Tiempo"],
  [["super-hypercube", "6 494", "331.6°", "sana", "—", "85 s"],
   ["super-icosahedron", "4 292", "331.4°", "sana", "—", "34 s"],
   ["supertube-(6,6)", "5 922", "330.7°", "sana", "—", "407 s"],
   ["superfullerene-C60", "7 534", "333.0°", "sana", "wall_anchor = 1", "123 s"],
   ["super-graphene", "1 180", "335.1°", "sana", "—", "35 s"],
   ["super-cubic", "836", "330.6°", "sana", "wall_anchor = 1", "71 s"],
   ["super-diamond", "3 752", "331.8°", "sana", "rejilla más fina", "980 s"],
   ["super-fcc", "2 982", "334.0°", "sana", "—", "413 s"]],
  [2200, 1000, 1700, 1200, 1900, 1360]));
c.push(spacer(120));
c.push(p("Ocho de ocho sanas, cero paredes colapsadas."));

c.push(h2("5.4 Conjunto de pruebas"));
c.push(p("1476 pruebas automatizadas (1 omitida por ausencia de tkinter en un entorno sin "
  + "pantalla). No comprueban únicamente que el código corra: fijan los resultados medidos y, "
  + "en varios casos, fijan también los criterios que se ensayaron y no funcionan, para que no "
  + "vuelvan a intentarse. Es un punto metodológico defendible: los contraejemplos están "
  + "versionados junto al código."));

/* ============ 6 ============ */
c.push(h1("6. Reproducibilidad"));
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
c.push(h1("7. Librerías utilizadas"));
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
c.push(h1("8. Agradecimientos"));
c.push(h2("8.1 Fuentes metodológicas"));
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
  + "Ihara, la ruta topológica a las bobinas."));
c.push(bullet("Stone, A. J. y Wales, D. J. — la rotación de enlace que genera pares 5-7."));

c.push(h2("8.2 Software"));
c.push(p("Formulación sugerida: «Este trabajo se apoya en el ecosistema científico de Python. "
  + "Los autores agradecen a las comunidades de NumPy, SciPy, ASE, NetworkX, scikit-image y "
  + "Matplotlib, cuyo software libre hace posible este tipo de trabajo.»"));

c.push(h2("8.3 Asistencia de inteligencia artificial"));
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
c.push(h1("9. Qué conviene destacar"));
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
