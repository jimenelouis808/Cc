const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  PageOrientation, LevelFormat, PageBreak, convertInchesToTwip,
} = require("docx");

const ACCENT = "1F3864";
const HEAD_BG = "E8EDF5";

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after === undefined ? 120 : opts.after,
               before: opts.before || 0, line: 276 },
    alignment: opts.align,
    indent: opts.indent,
    border: opts.border,
    children: [new TextRun({
      text, size: opts.size || 21, font: "Calibri",
      bold: opts.bold, italics: opts.italics,
      color: opts.color || "1A1A1A",
    })],
  });
}

function rich(runs, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after === undefined ? 120 : opts.after, line: 276 },
    indent: opts.indent,
    children: runs.map(r => new TextRun({
      text: r.t, size: r.size || 21, font: r.mono ? "Consolas" : "Calibri",
      bold: r.b, italics: r.i, color: r.c || "1A1A1A",
    })),
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 160 },
    children: [new TextRun({ text, size: 30, bold: true, font: "Calibri", color: ACCENT })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 260, after: 120 },
    children: [new TextRun({ text, size: 24, bold: true, font: "Calibri", color: ACCENT })],
  });
}
function bullet(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "vinetas", level },
    spacing: { after: 80, line: 276 },
    children: [new TextRun({ text, size: 21, font: "Calibri", color: "1A1A1A" })],
  });
}
function code(text) {
  return new Paragraph({
    spacing: { after: 60, before: 60 },
    indent: { left: convertInchesToTwip(0.3) },
    children: [new TextRun({ text, size: 18, font: "Consolas", color: "333333" })],
  });
}

const W = 9360; // usable width for Letter with 1" margins

function table(headers, rows, widths) {
  const cw = widths || headers.map(() => Math.floor(W / headers.length));
  const mk = (text, i, isHead) => new TableCell({
    width: { size: cw[i], type: WidthType.DXA },
    shading: isHead ? { type: ShadingType.CLEAR, fill: HEAD_BG, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: [new Paragraph({
      spacing: { after: 0, line: 240 },
      children: [new TextRun({
        text, size: 18, font: "Calibri", bold: isHead, color: isHead ? ACCENT : "1A1A1A",
      })],
    })],
  });
  return new Table({
    columnWidths: cw,
    width: { size: W, type: WidthType.DXA },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 2, color: "BFBFBF" },
      bottom: { style: BorderStyle.SINGLE, size: 2, color: "BFBFBF" },
      left: { style: BorderStyle.SINGLE, size: 2, color: "BFBFBF" },
      right: { style: BorderStyle.SINGLE, size: 2, color: "BFBFBF" },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 1, color: "D9D9D9" },
      insideVertical: { style: BorderStyle.SINGLE, size: 1, color: "D9D9D9" },
    },
    rows: [
      new TableRow({ tableHeader: true, children: headers.map((t, i) => mk(t, i, true)) }),
      ...rows.map(r => new TableRow({ children: r.map((t, i) => mk(t, i, false)) })),
    ],
  });
}
function spacer(h = 160) { return new Paragraph({ spacing: { after: h }, children: [] }); }

const children = [];

// ---------- Portada ----------
children.push(new Paragraph({
  spacing: { before: 1800, after: 120 },
  children: [new TextRun({
    text: "Generación computacional de estructuras de nanocarbono",
    size: 44, bold: true, font: "Calibri", color: ACCENT })],
}));
children.push(new Paragraph({
  spacing: { after: 400 },
  children: [new TextRun({
    text: "Metodología, validación, dependencias y agradecimientos",
    size: 26, font: "Calibri", color: "595959" })],
}));
children.push(p("Material de apoyo para la redacción de un artículo científico.", { italics: true, color: "595959" }));
children.push(p("Paquete: nanocarbon_lab v0.2.0 (workspace nanocarbon).", { color: "595959" }));
children.push(p("Documento generado el 25 de septiembre de 2026.", { color: "595959" }));
children.push(spacer(300));
children.push(new Paragraph({
  border: { top: { style: BorderStyle.SINGLE, size: 6, color: ACCENT } },
  spacing: { after: 200 }, children: [],
}));
children.push(p(
  "Aviso sobre las referencias: todas las citas de este documento fueron extraídas de los " +
  "comentarios y docstrings del propio código fuente, donde se registraron al implementar cada " +
  "método. Antes de enviar el artículo, verifique volumen, página y año contra el original: " +
  "este documento acredita la procedencia de cada método, no sustituye la comprobación " +
  "bibliográfica.", { italics: true, color: "7F2704" }));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ---------- 1 ----------
children.push(h1("1. Resumen del sistema"));
children.push(p(
  "nanocarbon_lab genera estructuras atomísticas de carbono —nanotubos, fulerenos, conos, " +
  "toroides, uniones multiterminal, bobinas, haeckelitas, schwarzitas, espumas, superredes y " +
  "dicalcogenuros— junto con dopado, funcionalización y análisis estructural. El paquete es " +
  "autónomo: no importa ningún otro paquete del repositorio y conserva su propio conjunto de " +
  "pruebas."));
children.push(p(
  "El principio metodológico que distingue al generador es que la estadística de anillos no se " +
  "impone, se deriva. En la mayoría de los constructores no existe ninguna instrucción que diga " +
  "«coloca un pentágono aquí»: se define una superficie, se malla, y los pentágonos, heptágonos " +
  "y octágonos aparecen donde la curvatura gaussiana los exige. Euler tiene la última palabra."));
children.push(spacer());
children.push(table(
  ["Módulo", "Contenido"],
  [
    ["builders/", "23 constructores: cnt, fullerene, nanocone, toroid, junction, nanocoil, periodic_coil, haeckelite, haeckelite_tube, heptanene, foam3d, network, supernetwork, swept, centerline, graphene, nanoribbon, capped_cnt, assemblies, lattice_edits, implicit, remesh, fullerene_mesh"],
    ["analyse/", "curvature, hybridisation, rings, shape, topological, report"],
    ["gui/", "Aplicación Tkinter con catálogo de presets y visualización 3D"],
  ], [2200, 7160]));

// ---------- 2 ----------
children.push(h1("2. Metodología de generación"));

children.push(h2("2.1 La ruta implícita: campo → malla → dual"));
children.push(p(
  "Las familias curvas (uniones, schwarzitas, toroides, bobinas, superredes) no se construyen " +
  "pegando fragmentos, sino a partir de una superficie implícita. Cada constructor devuelve un " +
  "campo escalar f(x) cuyo conjunto de nivel cero es la pared de carbono:"));
children.push(bullet("Uniones (L, T, Y, X): unión suave (smooth union) de cápsulas que irradian desde el origen. El radio de mezcla de la unión suave es lo que crea el cuello acampanado y de curvatura negativa en la ramificación; una unión dura dejaría una arista que ningún panal puede teselar limpiamente."));
children.push(bullet("Schwarzitas: superficies mínimas triplemente periódicas (Schwarz P, Schwarz D, giroide) en sus aproximaciones trigonométricas estándar."));
children.push(bullet("Superredes: cada arista de un grafo se convierte en una cápsula del radio de tubo elegido, unidas por unión suave, de modo que cada vértice adquiere curvatura real en lugar de un pliegue."));
children.push(p(
  "El conjunto de nivel cero se extrae por marching cubes (Lorensen y Cline). Para celdas " +
  "periódicas se usa una variante que suelda cada cara con su opuesta, de modo que la superficie " +
  "es continua a través de la frontera de la celda."));

children.push(h2("2.2 Remallado isotrópico"));
children.push(p(
  "Marching cubes produce una triangulación estanca pero inservible para este fin: sus " +
  "triángulos siguen la rejilla de muestreo, de modo que los grados de los vértices se dispersan " +
  "entre 3 y 9. Esto importa porque en la dualización un vértice de malla de grado d se " +
  "convierte en un anillo de carbono de tamaño d. Un vértice de grado 3 sería un anillo de tres " +
  "miembros; uno de grado 9, un hueco de nueve. Ninguno existe en carbono sp2 real."));
children.push(p(
  "La malla cruda se remalla isotrópicamente siguiendo el esquema de Botsch y Kobbelt: dividir " +
  "aristas largas, colapsar cortas, voltear aristas hacia grado 6, suavizar tangencialmente y " +
  "reproyectar sobre la superficie. El resultado concentra los grados en 6, con 5 donde la " +
  "superficie es convexa y 7 donde ensilla."));
children.push(p(
  "Dos detalles resultaron críticos y conviene mencionarlos en la sección de métodos:", { after: 80 }));
children.push(bullet("Condición de enlace en el colapso: colapsar una arista cuyos extremos comparten más que los dos vértices opuestos rompe la superficie, por lo que esos colapsos se rechazan. Toda operación preserva la manifoldidad."));
children.push(bullet("Límite de longitud en el volteo: la aceptación del volteo de aristas era puramente topológica, sin comprobación de longitud, lo que permitía aristas arbitrariamente largas y colapsaba la pared de los tubos. Con el límite (FLIP_MAX_EDGE = 1.8 Å, con convención de imagen mínima en celdas periódicas) la desviación fuera de superficie de las bobinas cayó de 0.801, 1.010 y 2.378 Å a 0.387, 0.327 y 0.361 Å."));

children.push(h2("2.3 Curvatura, censo de anillos y el presupuesto de Euler"));
children.push(p(
  "El censo de anillos no se asume: se comprueba. Para una superficie cerrada de característica " +
  "de Euler χ se cumple"));
children.push(code("   Σ (6 − n) = 6 · χ"));
children.push(p(
  "donde n recorre los tamaños de anillo. Un fulereno (χ = 2) exige exactamente doce pentágonos; " +
  "un toroide (χ = 0) exige que cada pentágono esté emparejado con un heptágono, sin necesidad " +
  "de ningún otro tamaño. χ se lee de la malla, no se supone, de modo que la comprobación es " +
  "independiente del constructor."));
children.push(p(
  "Para las superredes existe además una segunda comprobación, verdaderamente independiente. La " +
  "pared de una red de tubos es la frontera de un grafo engrosado, para el cual χ = 2·(V − E) " +
  "—un asa por cada ciclo independiente del grafo—, de modo que"));
children.push(code("   Σ (6 − n) = 12 · (V − E)"));
children.push(p(
  "queda fijado por el esqueleto antes de mallar nada. Esto detecta lo que la comprobación de " +
  "Euler sobre la propia malla no puede detectar: una malla que cerró perfectamente alrededor " +
  "del grafo equivocado. Una mezcla lo bastante ancha para fundir dos puntales en uno, o una " +
  "rejilla lo bastante gruesa para estrangular un cuello, produce exactamente eso."));

children.push(h2("2.4 Colocación de disclinaciones por Gauss-Bonnet discreto"));
children.push(p(
  "Que el censo global sea correcto no garantiza que cada disclinación esté en el sitio que la " +
  "curvatura pide. Para eso se usa Gauss-Bonnet leído localmente. El déficit angular"));
children.push(code("   K(v) = 2π − Σ θ"));
children.push(p(
  "es la curvatura gaussiana discreta en un vértice, y es geométrica y no combinatoria: un " +
  "vértice de grado 5 en una lámina plana tiene cinco ángulos de 72° que suman exactamente 2π, " +
  "de modo que su déficit es cero. Eso es lo que lo hace utilizable como objetivo: dice lo que " +
  "hace la superficie, no lo que hace la malla. Sobre una región,"));
children.push(code("   Σ (6 − grado) = (3/π) · ∫ K dA"));
children.push(p(
  "de modo que la parte correspondiente a un vértice es 3·K(v)/π. Sumado sobre una malla " +
  "cerrada devuelve 6χ exactamente (medido: 11.951, 11.943 y 11.865 frente al valor exacto 12), " +
  "por lo que una malla que igualara sus objetivos en todas partes satisfaría automáticamente la " +
  "comprobación de Euler."));
children.push(rich([
  { t: "Advertencia metodológica que conviene declarar en el artículo. ", b: true },
  { t: "La carga debe compararse sobre un entorno, nunca por vértice. La curvatura está " +
       "repartida sobre un área mientras que una disclinación es un punto: los objetivos por " +
       "vértice valen todos menos de 0.2 frente a un exceso de grado de ±1, de modo que un " +
       "heptágono «cuesta» 1.045 en cualquier posición —una razón de 2067:1— y el objetivo se " +
       "vuelve ciego a la posición. La comparación se hace por tanto sobre vecindarios de dos " +
       "anillos obtenidos por búsqueda en anchura." },
]));
children.push(p(
  "El descenso es codicioso y recalcula el objetivo en cada ronda. Dos versiones incrementales " +
  "fracasaron de forma instructiva: los vecindarios quedan obsoletos entre barridos (el " +
  "desajuste subió de 355 a 418, peor que no hacer nada) y también dentro de un mismo barrido " +
  "(45 volteos aceptados, todos «mejorantes», y el barrido en conjunto peor). Además se añadió " +
  "una restricción que impide crear disclinaciones nuevas: sin ella, el algoritmo persigue " +
  "curvatura sub-cuántica y fabrica pares 5-7 espurios."));
children.push(p("Resultados medidos tras la colocación:"));
children.push(spacer(80));
children.push(table(
  ["Estructura", "Censo antes (5 / 6 / 7)", "Censo después", "Disclinaciones bien situadas"],
  [
    ["Toroide", "68 / 560 / 68", "22 / 653 / 22", "100 %"],
    ["Unión Y", "50 / 443 / 38", "23 / 497 / 11", "100 %"],
  ], [2000, 2600, 2400, 2360]));

children.push(h2("2.5 Origen de los doce: por qué el radio no interviene"));
children.push(p(
  "En un toro, K dA = cos φ dφ dθ: los radios se cancelan. La mitad exterior integra exactamente " +
  "4π para cualquier R y r, lo que da un presupuesto de disclinaciones de exactamente +12 fuera " +
  "y −12 dentro, independientemente del tamaño. Esto se consigna porque la alternativa " +
  "intuitiva —escalar con 4πr/e— fue ensayada y da 17 o 27, valores que no corresponden a nada."));

children.push(h2("2.6 Jerarquía de superestructuras"));
children.push(p(
  "Las superredes siguen la jerarquía de Romo-Herrera, Terrones, Terrones, Dag y Meunier: " +
  "tomar un bloque unidimensional, usar operaciones de grupo puntual para formar un nodo " +
  "multiterminal, y después usar el nodo como nuevo bloque constructivo dejando que las " +
  "operaciones de traslación generen la arquitectura. Super-grafeno, super-cuadrada, " +
  "super-cúbica y super-diamante son las cuatro que construyen los autores; los mismos dos " +
  "pasos generan un superfulereno o una jaula icosaédrica, porque nada en ellos es específico " +
  "de un cristal."));
children.push(p(
  "La abstracción implementada es por tanto un grafo con un encaje: un conjunto de vértices, un " +
  "conjunto de aristas y —cuando el objeto es periódico— las imágenes de celda que esas aristas " +
  "alcanzan. Cualquier estructura de carbono terminada puede convertirse en el esqueleto de una " +
  "mayor, que es exactamente lo que el artículo original llama «el nodo como nuevo bloque " +
  "constructivo»."));

children.push(h2("2.7 Coordenadas topológicas"));
children.push(p(
  "Cuando lo único conocido de una nanoestructura es qué átomo está enlazado con cuál, se " +
  "aplican los métodos recogidos en la revisión de István László. Ciertos vectores propios de " +
  "la matriz de adyacencia son bilobulares y se comportan como las primeras ondas " +
  "estacionarias sobre la superficie, de modo que pueden leerse como ángulos:"));
children.push(bullet("Esférico (Fowler y Manolopoulos; Pisanski y Shawe-Taylor): tres vectores propios bilobulares, escalados por 1/√(λ₁ − λₖ), son las x, y, z de un fulereno."));
children.push(bullet("Toroidal (László y col.): tres no bastan —Graovac y col. demostraron que el toro sale plano desde alguna dirección— y cuatro sí. Un toro es el producto de dos circunferencias, de modo que los cuatro se dividen en dos pares degenerados: uno da el ángulo alrededor del anillo y el otro el ángulo alrededor del tubo."));
children.push(p(
  "Cuáles cuatro se midió, no se supuso: tomando un polihexágono toroidal que el paquete " +
  "construye exactamente —un tubo (5,5) curvado sobre 60 periodos, 1200 átomos, todo " +
  "hexágonos—, descartando sus coordenadas y conservando sólo sus enlaces, los vectores " +
  "propios bilobulares en orden decreciente de valor propio son 1, 2, 13, 14."));

// ---------- 3 ----------
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(h1("3. Validación y control de calidad"));

children.push(h2("3.1 Hibridación medida desde la geometría"));
children.push(p(
  "La fracción sp3 se mide directamente a partir de la suma de los tres ángulos de enlace de " +
  "cada carbono tricoordinado: 360° cuando es plano (sp2 ideal, como en grafeno) y 328.4° cuando " +
  "es tetraédrico (sp3 ideal, tres ángulos de 109.47° con el cuarto enlace fuera de la red). La " +
  "fracción"));
children.push(code("   carácter sp3 = (360 − Σθ) / (360 − 328.4)"));
children.push(p("va de 0 para plano a 1 para tetraédrico."));
children.push(rich([
  { t: "Distinción que el artículo debería declarar explícitamente: ", b: true },
  { t: "ésta es la misma suma angular que aparece en el análisis de curvatura, y los dos usos " +
       "se parecen sin ser lo mismo. El déficit 2π − Σθ nunca es negativo en un vértice " +
       "trivalente —la desigualdad del triángulo esférico lo prohíbe—, de modo que no puede " +
       "distinguir un casquete de un cuello y es inútil como signo de curvatura. Lo que sí mide " +
       "fielmente es cuánto se ha piramidalizado el carbono, y eso es hibridación. Una pared de " +
       "curvatura negativa no aparece como sp3, y no debe aparecer." },
]));
children.push(p(
  "Control experimental: una haeckelita plana, llena de pentágonos y heptágonos, mide " +
  "exactamente 0.00 de carácter sp3; un tubo (5,5) prístino mide 3.390 ± 0.000. La curvatura " +
  "vive en el censo de anillos; la hibridación, en los ángulos."));

children.push(h2("3.2 Pared colapsada: un criterio de imposibilidad, no de calidad"));
children.push(p(
  "Una suma angular inferior a 328.4° está más allá de lo tetraédrico, que ningún carbono " +
  "alcanza. No es una estructura mala: es una imposible. Por eso el constructor no la devuelve " +
  "sin más, sino que la reconstruye sujetando la pared a su propia superficie (wall_anchor) o " +
  "afinando la rejilla, y escala el remedio hasta que la suma angular mínima supera el umbral " +
  "con margen."));
children.push(p(
  "El campo implícito vale cero sobre la pared, de modo que |f(átomo)| es la desviación exacta " +
  "fuera de superficie y puede usarse como métrica sin aproximación alguna."));

children.push(h2("3.3 Estado del catálogo de superestructuras"));
children.push(p("Auditoría completa, monohilo, semilla 0, sobre la versión documentada aquí:"));
children.push(spacer(80));
children.push(table(
  ["Preset", "Átomos", "Suma angular mín.", "Veredicto", "Remedio", "Tiempo"],
  [
    ["super-hypercube", "6494", "331.6°", "sana", "—", "85 s"],
    ["super-icosahedron", "4292", "331.4°", "sana", "—", "34 s"],
    ["supertube-(6,6)", "5922", "330.7°", "sana", "—", "407 s"],
    ["superfullerene-C60", "7534", "333.0°", "sana", "wall_anchor = 1", "123 s"],
    ["super-graphene", "1180", "335.1°", "sana", "—", "35 s"],
    ["super-cubic", "836", "330.6°", "sana", "wall_anchor = 1", "71 s"],
    ["super-diamond", "3752", "331.8°", "sana", "rejilla más fina", "980 s"],
    ["super-fcc (preset)", "2982", "334.0°", "sana", "—", "413 s"],
  ], [2200, 1000, 1700, 1200, 1900, 1360]));
children.push(spacer(80));
children.push(p(
  "Ocho de ocho sanas, cero paredes colapsadas. La super-fcc, con doce tubos por nodo, es la " +
  "más densa del catálogo y da la mejor suma angular mínima de todas las redes periódicas " +
  "tridimensionales."));

children.push(h2("3.4 Conjunto de pruebas"));
children.push(p(
  "1476 pruebas automatizadas (1 omitida por ausencia de tkinter en el entorno sin pantalla). " +
  "Las pruebas no comprueban únicamente que el código corra: fijan los resultados medidos y, en " +
  "varios casos, fijan también los criterios que se ensayaron y no funcionan, para que no " +
  "vuelvan a intentarse. Éste es un punto metodológico defendible en el artículo: los " +
  "contraejemplos están versionados junto al código."));

// ---------- 4 ----------
children.push(h1("4. Reproducibilidad"));
children.push(p(
  "Para la sección de métodos o el material suplementario, la información mínima que permite " +
  "reproducir cualquier estructura del artículo es la siguiente."));
children.push(spacer(80));
children.push(table(
  ["Elemento", "Valor"],
  [
    ["Paquete", "nanocarbon_lab v0.2.0"],
    ["Python", "≥ 3.11"],
    ["Gestor de entorno y bloqueo de versiones", "uv (lockfile único para todo el workspace)"],
    ["Semilla aleatoria", "seed = 0 en todas las cifras reportadas"],
    ["Hilos de álgebra lineal", "OMP_NUM_THREADS = 1"],
    ["Parámetros por estructura", "escala de celda, radio de tubo, anchura de mezcla, resolución de rejilla, iteraciones de remallado y de relajación"],
  ], [3200, 6160]));
children.push(spacer(120));
children.push(p("Ejecución del conjunto de pruebas:"));
children.push(code("OMP_NUM_THREADS=1 uv run python -m pytest nanocarbon_lab/tests \\"));
children.push(code("    -q -n 4 --dist loadscope -m \"not slow\""));
children.push(spacer(80));
children.push(p(
  "Declare la semilla en el artículo. Los constructores son deterministas dada la semilla, pero " +
  "no entre versiones del remallador: un cambio en el criterio de volteo modifica la malla y, " +
  "con ella, el censo de anillos.", { italics: true }));

// ---------- 5 ----------
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(h1("5. Librerías utilizadas"));
children.push(p(
  "Todas son de código abierto y todas piden ser citadas. La columna de la derecha indica para " +
  "qué se usa cada una en este trabajo, que es lo que conviene declarar."));
children.push(spacer(80));
children.push(table(
  ["Librería", "Versión mínima", "Papel en este trabajo"],
  [
    ["NumPy", "1.24", "Aritmética de arreglos en todo el paquete: campos escalares, posiciones, mallas"],
    ["SciPy", "1.10", "Álgebra lineal dispersa, vectores propios para coordenadas topológicas, búsquedas espaciales"],
    ["ASE (Atomic Simulation Environment)", "3.22", "Objeto Atoms, celdas, condiciones periódicas, entrada/salida de formatos atomísticos"],
    ["NetworkX", "3.0", "Grafos: esqueletos de superredes, ciclos, detección de anillos"],
    ["scikit-image", "0.22", "Marching cubes para la extracción del conjunto de nivel cero"],
    ["Matplotlib", "—", "Visualización 3D en la interfaz gráfica"],
    ["pytest, pytest-xdist, pytest-cov", "7.4 / 3.5 / 4.1", "Conjunto de pruebas y ejecución en paralelo"],
    ["Ruff", "0.4", "Análisis estático"],
    ["uv", "—", "Resolución de dependencias y lockfile reproducible"],
  ], [2500, 1400, 5460]));
children.push(spacer(120));
children.push(p(
  "Nota sobre bpy (Blender): está declarado como conflictivo en la raíz del workspace porque " +
  "fija numpy 1.26 y arrastraría esa versión a todos los paquetes. Si el artículo incluye " +
  "figuras renderizadas con Blender, decláre­lo por separado.", { italics: true }));

// ---------- 6 ----------
children.push(h1("6. Agradecimientos"));

children.push(h2("6.1 Fuentes metodológicas"));
children.push(p(
  "Los métodos implementados provienen de trabajos publicados y deben acreditarse en el cuerpo " +
  "del artículo, no sólo en los agradecimientos:"));
children.push(bullet("Romo-Herrera, Terrones, Terrones, Dag y Meunier — Nano Letters 7 (2007) 570. La jerarquía de superredes de nanotubos: nodo multiterminal como bloque constructivo. La comprobación del censo de anillos reproduce la que describe su Información de Apoyo S2."));
children.push(bullet("Terrones y col. — Physical Review Letters 84 (2000) 1716. Haeckelitas: redes planas y tubulares de pentágonos, hexágonos y heptágonos."));
children.push(bullet("Krishnan y col. — Nature 388 (1997). Ángulos de apertura observados en nanoconos de carbono, que fijan los casos construibles."));
children.push(bullet("László, I. — Theoretical Chemistry Accounts 134 (2015) 104. Revisión de coordenadas topológicas a partir de la matriz de adyacencia."));
children.push(bullet("Fowler y Manolopoulos; Pisanski y Shawe-Taylor — caso esférico de las coordenadas topológicas."));
children.push(bullet("Graovac y col. — demostración de que tres vectores propios no bastan para el toro."));
children.push(bullet("Popović y col. — Contemporary Materials III-1 (2012) 51; Liu y col. — Nanoscale Research Letters 5 (2010) 478. Relaciones D/d de bobinas monocapa relajadas, usadas como banda de referencia por el constructor."));
children.push(bullet("Botsch, M. y Kobbelt, L. — esquema de remallado isotrópico (dividir, colapsar, voltear, suavizar, reproyectar)."));
children.push(bullet("Lorensen, W. E. y Cline, H. E. — Computer Graphics 21 (1987) 163. Algoritmo marching cubes."));
children.push(bullet("Dunlap, B. I. — criterio de disclinaciones en carbonos toroidales."));

children.push(h2("6.2 Software"));
children.push(p(
  "Formulación sugerida: «Este trabajo se apoya en el ecosistema científico de Python. Los " +
  "autores agradecen a las comunidades de NumPy, SciPy, ASE, NetworkX, scikit-image y " +
  "Matplotlib, cuyo software libre hace posible este tipo de trabajo.» Cada una de ellas tiene " +
  "una cita canónica que debe incluirse en la bibliografía, no sólo en los agradecimientos."));

children.push(h2("6.3 Asistencia de inteligencia artificial"));
children.push(p(
  "El desarrollo del generador se realizó con asistencia de Claude (Anthropic), utilizado a " +
  "través de Claude Code. Conviene declararlo, y conviene hacerlo en el lugar correcto."));
children.push(rich([
  { t: "Sobre la autoría. ", b: true },
  { t: "Las políticas del ICMJE, del COPE y de las principales editoriales (Elsevier, Springer " +
       "Nature, Wiley, ACS) coinciden en que un sistema de IA no puede figurar como autor, " +
       "porque no puede asumir responsabilidad por el trabajo ni declarar conflictos de " +
       "interés. Su uso se declara en los agradecimientos o en una sección específica de " +
       "métodos. Verifique la redacción exacta que exige la revista a la que envíe: varias " +
       "piden una declaración con formato propio." },
]));
children.push(spacer(80));
children.push(p("Texto sugerido para la sección de agradecimientos:", { bold: true }));
children.push(new Paragraph({
  spacing: { after: 160, before: 60, line: 276 },
  indent: { left: convertInchesToTwip(0.35), right: convertInchesToTwip(0.35) },
  border: { left: { style: BorderStyle.SINGLE, size: 12, color: ACCENT, space: 12 } },
  children: [new TextRun({
    text: "Los autores agradecen a Anthropic. El código de generación y análisis estructural " +
          "descrito en este trabajo se desarrolló con asistencia de Claude (Anthropic) a través " +
          "de Claude Code. La asistencia comprendió la implementación de los algoritmos, el " +
          "diseño del conjunto de pruebas y la documentación del código. El diseño de la " +
          "investigación, la elección de los métodos, la interpretación de los resultados y la " +
          "verificación de toda cifra reportada corresponden íntegramente a los autores, " +
          "quienes asumen la responsabilidad completa del contenido.",
    size: 21, font: "Calibri", italics: true, color: "1A1A1A" })],
}));
children.push(p(
  "Ajuste el alcance descrito a lo que realmente ocurrió: si la asistencia se limitó a partes " +
  "concretas, dígalo así. Una declaración que exagere o minimice el papel de la herramienta es " +
  "más problemática que una precisa.", { italics: true, color: "7F2704" }));

// ---------- 7 ----------
children.push(h1("7. Qué conviene destacar en el artículo"));
children.push(p(
  "Tres puntos de este trabajo son metodológicamente defendibles y distinguen al generador de " +
  "una simple colección de constructores:"));
children.push(bullet("La estadística de anillos es derivada, no impuesta. En las familias curvas nadie coloca un pentágono: se define una superficie, se malla, y el censo sale de la curvatura. Esto permite enunciar la comprobación de Euler como una verificación independiente y no como una tautología."));
children.push(bullet("Existe una segunda comprobación verdaderamente independiente para las superredes. Σ(6−n) = 12·(V−E) queda fijado por el esqueleto antes de mallar, y detecta el caso que la comprobación de Euler sobre la propia malla no puede detectar: una superficie impecable alrededor del grafo equivocado."));
children.push(bullet("La validez física se comprueba con un criterio de imposibilidad, no de calidad. Una suma angular por debajo de 328.4° está más allá de lo tetraédrico, de modo que no es una estructura peor sino una que no puede existir; el constructor la reconstruye en lugar de entregarla."));
children.push(spacer(140));
children.push(new Paragraph({
  border: { top: { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" } },
  spacing: { before: 160, after: 120 }, children: [],
}));
children.push(p(
  "Todas las cifras de este documento proceden de ejecuciones registradas del paquete en la " +
  "versión indicada. Antes de publicarlas, vuelva a ejecutarlas en su propio entorno y " +
  "repórtelas desde esa ejecución.",
  { italics: true, color: "595959" }));

const doc = new Document({
  creator: "nanocarbon_lab",
  title: "Generación computacional de estructuras de nanocarbono",
  description: "Metodología, validación, dependencias y agradecimientos",
  numbering: {
    config: [{
      reference: "vinetas",
      levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 460, hanging: 240 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 900, hanging: 240 } } } },
      ],
    }],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840, orientation: PageOrientation.PORTRAIT },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    children,
  }],
});

Packer.toBuffer(doc).then(b => {
  fs.writeFileSync("/home/user/Cc/docs/articulo/metodologia-nanocarbon.docx", b);
  console.log("written");
});
