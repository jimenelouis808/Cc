# Protocolo del review bibliométrico
### Defectos y dopaje en nanoestructuras de carbono 1D (CNT y nanofibras)

Documento vivo. Congélalo (tag de git + depósito en OSF) **antes** de correr las
búsquedas definitivas. Un protocolo registrado *a priori* es la diferencia entre
"revisión sistemática" y "búsqueda bibliográfica contada".

---

## 1. Pregunta de investigación y encuadre

La trampa nº 1 de los reviews bibliométricos es quedarse en estadística
descriptiva: top-10 países, top-10 revistas, top-10 autores. Ese formato está
siendo rechazado de forma creciente en revistas de materiales, porque no
produce conocimiento sobre el material. **Necesitas una pregunta con una
respuesta posible.**

**Encuadre propuesto:**

> *El dopaje con heteroátomos y la ingeniería de defectos como palanca
> estructura–propiedad en nanocarbonos 1D: ¿cómo ha evolucionado el acoplamiento
> entre predicción teórica y realización experimental, y qué pares
> (dopante × defecto) siguen sin explorar frente a qué aplicaciones?*

Esto convierte la bibliometría en **instrumento**, no en producto. Las tres
preguntas operativas:

- **RQ1 — Estructura del campo.** ¿Qué comunidades temáticas existen y cómo
  migraron (co-palabras + mapa temático + evolución temporal)?
- **RQ2 — Acoplamiento teoría↔experimento.** ¿Predice la teoría o documenta?
  ¿Cuál es el *lag* temporal por dopante entre la primera predicción DFT y la
  primera síntesis reportada? ¿Se citan mutuamente los dos subcorpus?
- **RQ3 — Cobertura y huecos.** Matriz (dopante × tipo de defecto × aplicación):
  ¿qué celdas están saturadas, cuáles vacías, y cuáles vacías *pero con
  predicción teórica favorable*?

RQ2 y RQ3 son originales y **no las he visto hechas en este tema**. RQ3 en
particular te da la sección de "perspectivas futuras" con evidencia cuantitativa
en lugar de opinión, que es exactamente lo que un editor de review quiere.

---

## 2. Alcance: las tres decisiones que me preguntaste

### 2.1 ¿Incluir grafeno y óxido de grafeno? — **No en el corpus núcleo. Sí como corpus de contexto.**

**Razón.** El problema no es conceptual, es **aritmético**. La literatura de
grafeno dopado (sobre todo N-grafeno para ORR) es, con alta probabilidad, del
orden de 2–5× la de CNT dopados. Si los fusionas:

- Los clusters de co-palabras quedarán dominados por grafeno; los CNT aparecerán
  como satélite.
- El top-20 de artículos más citados será casi todo grafeno (Qu 2010, Gong 2009,
  Dai…) y tu review dejará de ser sobre CNT.
- Las revistas núcleo por ley de Bradford cambiarán hacia revistas 2D.

Es decir: **el corpus decidirá tu narrativa por ti, y elegirá la equivocada.**

**Pero no lo tires.** Diseño de tres niveles:

| Nivel | Contenido | Uso |
|---|---|---|
| **C1 — núcleo** | CNT + nanofibras de carbono × defectos/dopaje | Todo el análisis principal |
| **C2 — extensión** | Ensamblajes 3D/macroscópicos de CNT/CNF (§2.2) | **Dentro de C1**, como subtema etiquetado |
| **C3 — contexto** | Grafeno/GO × defectos/dopaje | Corpus **separado**, solo para 3 figuras comparativas |

Las tres figuras que justifican C3, y que son de las cosas más interesantes que
puedes sacar:

1. **Curvas de crecimiento normalizadas 1D vs 2D (1991–2025).** Existe casi con
   seguridad un punto de inflexión ~2009–2011 en el que la comunidad de dopaje
   migró de nanotubos a grafeno. Cuantificarlo (y ver si los CNT se recuperan
   post-2018 vía ensamblajes 3D y fibras) es un resultado narrativo de primer nivel.
2. **Migración de autores.** Fracción de autores productivos en C1 pre-2010 que
   publican en C3 post-2012. Es una medida directa del "efecto grafeno" sobre la
   comunidad. Nadie lo ha medido para este tema.
3. **Transferencia conceptual.** ¿Los conceptos de defectos (Stone-Wales,
   pyridinic/pyrrolic/graphitic N, sitios activos) nacieron en CNT y se
   exportaron a grafeno, o al revés? Se responde con RPYS + primeras apariciones
   de término por corpus.

**Excepción — híbridos.** Los papers de híbridos CNT–grafeno **ya caen en C1**
porque mencionan CNT. No los excluyas: etiquétalos (`is_hybrid`) y trátalos como
subtema. El `classify.py` del pipeline ya lo hace.

> ⚠ **Verifícalo tú con un piloto.** No aceptes mi estimación de tamaños. Corre
> el CORE y el corpus C3 solo para contar N, **antes** de decidir, y reporta esos
> números en Métodos como justificación de la exclusión. Una decisión de alcance
> justificada con datos propios es blindaje anti-revisor.

### 2.2 ¿Redes 3D de nanotubos? — **Sí, dentro del núcleo, como subtema etiquetado.**

Aquí sí incluir, y con convicción, por una razón física: en esponjas, bosques,
aerogeles, buckypaper, fibras e hilos, **los defectos y los dopantes dejan de ser
una perturbación local y pasan a ser el mecanismo de unión**. Los dopantes de N
y B promueven uniones tipo Y y codos pentagonales/heptagonales; la conductividad
y la mecánica de la red macroscópica están dominadas por las junturas, no por el
tubo individual. Es decir: el 3D no es "una aplicación más", es **un régimen
distinto de la misma física** que estás revisando.

Además te da un eje de escala muy vendible:

> **átomo (dopante sustitucional) → tubo (defecto topológico) → juntura (unión
> Y, red) → material macroscópico (fibra, esponja, papel) → dispositivo**

Ese eje de escala es un excelente esqueleto para la parte narrativa del review, y
la bibliometría te dice cuánta literatura hay en cada peldaño (spoiler: casi
seguro está hueco en el peldaño "juntura", que es tu hallazgo).

Etiqueta con `morphology ∈ {1D-individual, bundle/array, fiber/yarn, film/buckypaper, sponge/aerogel/foam, forest/VACNT}`.
Ya implementado en `classify.py`.

### 2.3 ¿Aplicaciones, o solo efectos y caracterización? — **Ambos, pero jerarquizados.**

Solo efectos/caracterización → review estrecho, sin impacto, y compite con
revisiones de espectroscopía ya existentes.
Solo aplicaciones → te conviertes en el enésimo "N-doped carbon for ORR", tema
saturadísimo, y pierdes lo que te hace distinto (tienes `nanocarbon_lab`, tienes
la parte teórica).

**Estructura recomendada — dos ejes, y el segundo subordinado al primero:**

- **Eje primario (columna vertebral): mecanismo.**
  `tipo de dopante/defecto → cambio estructural → firma de caracterización → propiedad`
  - Dopantes: N (pyridinic/pyrrolic/graphitic/quaternary), B, P, S, F, Si, O, Se, halógenos, co-dopaje N-S / N-B / N-P, metales de transición y single-atom.
  - Defectos: vacantes (mono/di/multi), Stone-Wales, pentágono-heptágono, bordes, sp3, adátomos, intersticiales, fronteras de grano, dopaje sustitucional vs intersticial.
  - Firmas: Raman (I_D/I_G, D', 2D), XPS (deconvolución N 1s: ~398.5 / 400.1 / 401.3 eV), EELS, XANES/NEXAFS, HRTEM/STEM-HAADF, STM, EPR.
  - Teoría: energías de formación, estructura de banda, densidad de estados, transferencia de carga (Bader/Löwdin), transporte NEGF, energías de adsorción.

- **Eje secundario (resultado): aplicaciones como *clusters de destino*.**
  Electrocatálisis (ORR/HER/OER/CO2RR), almacenamiento de energía (supercaps,
  Li/Na/K-ion, Li-S), sensores y biosensores, emisión de campo, termoeléctricos y
  conductividad térmica, transistores, composites y refuerzo mecánico,
  adsorción/captura de CO2, membranas, apantallamiento EMI.

**La figura que ata los dos ejes y que sería la portada de tu review:**
una **matriz de calor (dopante × aplicación)** donde cada celda lleva
(a) nº de documentos, (b) citación normalizada por campo media, (c) fracción
teórica vs experimental. Se construye con NER por reglas sobre los abstracts —
está implementado en `classify.py` y `crosstab`. Las celdas vacías con alta
señal teórica **son tu sección de perspectivas**.

---

## 3. Criterios de inclusión / exclusión

**Inclusión**
- Documento indexado en Scopus y/o WoS Core Collection, 1991–<fecha de corte>.
- Tipo: Article, Review. (Decisión sobre Conference Paper: ver §5.)
- Idioma: inglés (declara el sesgo; ver §7).
- El documento trata de una nanoestructura de carbono 1D (CNT, CNF, nanocoil,
  nanofilamento, o ensamblaje macroscópico de éstas) **y** aborda dopaje con
  heteroátomos y/o defectos estructurales, sea experimental o computacionalmente.

**Exclusión**
- El nanocarbono 1D es solo soporte inerte y el dopaje ocurre en otra fase
  (p. ej. "TiO2 dopado con N soportado en MWCNT"). ← **Este es el falso positivo
  más frecuente y más difícil de tu tema.** `classify.py` marca candidatos con la
  heurística `dopant_host_ambiguous` para revisión manual.
- "Doping" en sentido de dopaje electrónico por transferencia de carga
  intercalada/adsorbida sin incorporación en la red (p. ej. dopaje con HNO3),
  **salvo** que se decida incluirlo como categoría propia (recomiendo:
  inclúyelo, pero etiquetado `doping_mode = charge-transfer` vs `substitutional`).
  Esa distinción es físicamente importante y casi nunca se hace en reviews.
- BNNT, nanotubos inorgánicos, nanofibras de celulosa, nanotubos de péptidos.
- Sin resumen (imposible de clasificar) → registrar en el flujo PRISMA.
- Retractados (ver §7).

---

## 4. Calibración previa: conjunto de oro (hazlo, en serio)

**Antes** de fijar la consulta, arma a mano un *gold standard* de 40–60
artículos que **tienen que** salir: los seminales (Iijima 1991; Stone & Wales
1986; primeros CNT dopados con N/B de Terrones y Ajayan, finales de los 90;
Gong/Dai 2009 sobre VA-NCNT y ORR), más 2–3 por cada dopante, por cada tipo de
defecto y por cada morfología, más los de tu propio grupo.

Luego mide, para cada variante de consulta (el brazo de precisión y el de alta
sensibilidad — la diferencia entre ambos es lo que justifica quedarse con uno):

- **Recall relativo** = (nº del gold set recuperado) / (tamaño del gold set).
  Objetivo ≥ 0.95. Si un seminal no sale, la consulta tiene un agujero
  terminológico — encuéntralo, no lo parchees a mano.
- **Precisión** = revisa 100 registros aleatorios y clasifica relevante/no.

Reporta esta tabla en Métodos. Es de las cosas que más suben la credibilidad de
un review bibliométrico y casi nadie la hace.

**Hay herramienta para esto**, no hay que hacerlo a mano:

```bash
cp queries/gold_standard_template.csv queries/gold_standard.csv    # y complétala
python -m nanocarbon_biblio.cli recall --raw data/raw --gold queries/gold_standard.csv
```

Devuelve el recall relativo, qué base encontró cada entrada, y la lista de los
no recuperados. Ver `queries/GOLD_STANDARD.md` para cómo construir el conjunto.
También está en la pestaña 7 de la GUI.

---

## 5. Decisiones a fijar (y reportar) antes de correr

| Decisión | Recomendación | Reportar |
|---|---|---|
| Fecha de corte | Año completo cerrado (p. ej. 31-12-2025) | Sí, y la fecha de ejecución |
| Ventana | 1991–corte | Sí |
| Conference papers | **Excluir** del corpus analítico; incluir solo en curva de crecimiento | Ambos N |
| Idioma | Inglés | Sí, con discusión del sesgo |
| Base para el análisis principal | **Unión Scopus ∪ WoS deduplicada** | Venn de solapamiento |
| Robustez | Repetir análisis clave en Scopus solo y WoS solo | Sí (§7) |
| Conteo de coautoría | Fraccionario para países/instituciones | Sí |
| Citación | **Normalizada por campo y año** (percentiles), no bruta | Sí |
| Tesauro de keywords | Obligatorio, versionado en el repo | Publicar el fichero |

---

## 6. Cortes temporales con sentido físico

No uses periodos iguales (2000-2005, 2006-2010…). Usa hitos, y justifícalos:

| Periodo | Hito que lo abre |
|---|---|
| 1991–1998 | Iijima 1991; descubrimiento y primeras teorías de defectos |
| 1999–2003 | Primeras síntesis de CNT dopados con N/B (CVD con precursores nitrogenados) |
| 2004–2009 | Grafeno (2004); auge de CVD escalable; caracterización XPS sistemática |
| 2010–2015 | Explosión de electrocatálisis libre de metal (Gong 2009 / Qu 2010); competencia del grafeno |
| 2016–2020 | Catalizadores de átomo único; ensamblajes 3D y fibras; DFT de alto rendimiento |
| 2021–corte | ML/potenciales aprendidos; co-dopaje racional; economía de la descarbonización |

Alimenta estos cortes a `thematicEvolution()` de bibliometrix. El mapa temático
con cortes con significado físico se lee muchísimo mejor que con cortes arbitrarios.

---

## 7. Integridad, sesgos y robustez (la sección que te distingue)

- **Solapamiento Scopus/WoS.** Reporta el diagrama de Venn con N exactos. En
  nanomateriales suele haber 20–35 % de registros exclusivos de una base. **Este
  es un resultado publicable por sí solo** y justifica usar la unión.
- **Análisis de robustez.** Corre las conclusiones principales (clusters, top
  temas, tendencias) por separado en Scopus y en WoS. Si coinciden, tu review es
  robusto y lo dices. Si no, es un hallazgo. Casi nadie lo hace.
- **Tercera fuente: OpenAlex** (gratis, vía `pyalex`). Úsala para (a) medir sesgo
  de cobertura de las dos comerciales, (b) enriquecer con ROR de instituciones,
  estado de acceso abierto y conceptos, (c) rescatar DOIs faltantes.
- **Retractaciones e integridad.** La literatura de nanocarbono para
  electrocatálisis tiene una huella no trivial de retractaciones y *paper mills*.
  Cruza los DOIs contra la base de Retraction Watch (distribuida vía Crossref,
  acceso libre). Que un review declare "se identificaron N registros retractados
  y se excluyeron" es una señal de calidad enorme y prácticamente inédita en
  este tema.
- **Sesgo idiomático y geográfico.** Buena parte de la producción china y rusa
  relevante no está en inglés ni completamente indexada. Decláralo como
  limitación explícita, no lo escondas.
- **Desambiguación de autores.** Usa Scopus AU-ID y ORCID; los nombres chinos y
  coreanos colapsan brutalmente en WoS. Si vas a hacer ranking de autores, esto
  no es opcional.
- **Deriva terminológica.** `nitrogen-doped CNT` (2000s) → `N-doped CNT` →
  `N-CNT` → `NCNT`. Sin tesauro, tu análisis de co-palabras fragmenta el mismo
  concepto en cuatro nodos. El tesauro es obligatorio (`queries/thesaurus_seed.txt`).

---

## 8. Reporte

- **PRISMA 2020** (diagrama de flujo) + **PRISMA-S** (checklist específico de
  reporte de búsquedas). Sí, aplican a reviews bibliométricos, y los editores lo
  agradecen. El diagrama lo genera el pipeline en SVG desde `manifest.json`:
  `python -m nanocarbon_biblio.cli prisma --excluded N`. Comprueba su propia
  aritmética, así que no puede llegar a un revisor con las cuentas mal.
- **Acuerdo entre codificadores.** `python -m nanocarbon_biblio.cli agreement
  --sheet <hoja codificada>` da la kappa de Cohen y precisión/recall por clase.
  Reporta el valor, el n y quién codificó.
- Registrar el protocolo en **OSF** antes de las búsquedas definitivas.
- Publicar en **Zenodo** (DOI): cadenas de búsqueda, tesauro, código Python y R,
  y — si la licencia lo permite — los identificadores (DOI/EID/UT) del corpus.
  Los metadatos completos de Scopus/WoS **no** se pueden redistribuir; los
  listados de DOIs, sí. Esto hace tu review reproducible sin violar licencias.
- Manuscrito en **Quarto** (`.qmd`) mezclando R y Python en el mismo documento.

## 9. Revistas destino candidatas

Según hacia dónde inclines el peso:

- Peso en materiales/física: *Carbon*, *Progress in Materials Science*,
  *Advanced Functional Materials*, *Journal of Materials Chemistry A*,
  *Nanoscale*, *Carbon Trends*, *FlatChem*, *Materials Today Chemistry*.
- Peso en método bibliométrico: *Scientometrics*, *Journal of Informetrics*,
  *Quantitative Science Studies*.
- Híbrido (recomendado si RQ2/RQ3 salen bien): revista de materiales, con la
  metodología completa en material suplementario.
