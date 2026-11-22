# Conjunto de oro (*gold standard*) para calibrar la búsqueda

`gold_standard_template.csv` es una **plantilla que tienes que completar tú**.
No trae DOIs: rellenarlos con datos bibliográficos que no he podido verificar
convertiría la calibración en una medida de mi memoria en vez de una medida de
tu consulta. Las dos primeras filas llevan título y año de dos trabajos
fundacionales del tema como punto de partida, y aun así están marcadas
`verified=FALSE`: compruébalas antes de usarlas.

## Por qué molestarse

Una consulta bibliográfica no se valida leyéndola. Se valida comprobando que
recupera los artículos que **sabes** que tiene que recuperar. Eso es un *test de
elemento conocido* (`known-item test`), y produce una cifra —el **recall
relativo**— que va en Métodos y que casi ningún review bibliométrico reporta.

Si un artículo seminal no sale, tu consulta tiene un agujero terminológico.
**No lo parchees añadiendo el artículo a mano**: ese mismo agujero está
escondiendo decenas de trabajos de los que no has oído hablar. Encuentra el
término que falta y arregla la consulta.

## El plan, en bloques de 20 minutos

No lo hagas de una sentada. Son cinco bloques, y los cuatro primeros se pueden
hacer en días distintos: el fichero se fusiona, nunca se sobrescribe.

### Bloque 1 — Cosecha tu propia biblioteca (20 min)

Lo que ya tienes vale, y es lo más rápido.

1. En Zotero / Mendeley / EndNote, selecciona los artículos del tema.
2. Exporta a **RIS** o **BibTeX**.
3. Vuélcalo:

```bash
python -m nanocarbon_biblio.cli gold-from ~/Downloads/mi_biblioteca.ris
```

Lee DOI, título y año, y escribe `queries/gold_standard.csv`. No teclees nada.

### Bloque 2 — Cosecha de revisiones previas (20 min)

Coge las dos o tres revisiones más citadas del tema. En Scopus, abre cada una,
ve a **References**, selecciona las relevantes y expórtalas en CSV. Luego:

```bash
python -m nanocarbon_biblio.cli gold-from ~/Downloads/refs_review1.csv
```

Acepta también exportaciones de Scopus y de WoS, no solo de gestores
bibliográficos. Fusiona por DOI y por título: lo que ya estaba no se duplica.

### Bloque 3 — Tapa los huecos a propósito (20 min)

Abre `queries/gold_standard.csv` y compáralo con la lista de categorías de más
abajo. Para cada hueco —¿tengo algo de dopaje con selenio? ¿de buckypaper?—
busca **un** artículo en Scopus y añade su DOI a mano. Suelen faltar cinco o
seis categorías; a dos minutos cada una.

Los dopantes, defectos y morfologías que el proyecto reconoce están listados en
`queries/DICCIONARIO.md`: úsalo como lista de comprobación.

### Bloque 4 — La columna `why` (20 min)

Rellena `why` en todas las entradas: *seminal*, *primer dopaje con P*, *esponja
3D*, *del grupo*. Parece burocracia y no lo es: obliga a que el conjunto esté
construido a propósito y no a golpe de memoria, y es lo que hace revisable la
selección cuando un revisor pregunta cómo la elegiste.

Aprovecha y pon `verified=TRUE` en las que hayas comprobado de verdad —que el
DOI resuelve al artículo que crees—. Las que no, déjalas en FALSE: el comando
`recall` te dirá cuántas van sin verificar.

### Bloque 5 — Mídelo (10 min)

Con la consulta ya exportada:

```bash
python -m nanocarbon_biblio.cli recall --raw data/raw --gold queries/gold_standard.csv
```

Objetivo **≥ 0.95**. Lo que no salga, no lo añadas a mano: busca el término que
le falta a la consulta.

---

## Cómo construirlo

40–60 entradas, elegidas a propósito para cubrir todo el espacio del review:

- **Seminales**: el descubrimiento, el origen del concepto de defecto, los
  primeros dopajes con N y con B.
- **Dos o tres por cada dopante**: N, B, P, S, F, halógenos, Si, O, Se, metales
  de transición, co-dopaje.
- **Dos o tres por cada tipo de defecto**: vacantes, Stone-Wales, fronteras de
  grano, bordes, sp3, inducidos por irradiación.
- **Dos o tres por cada morfología**: tubo individual, array, bosque VACNT,
  fibra o hilo, buckypaper, esponja o aerogel, juntura.
- **Los dos lados del método**: trabajos puramente computacionales y puramente
  experimentales, más alguno combinado.
- **Las revisiones previas** del tema.
- **Los tuyos**: si tu propio trabajo no sale, tu consulta está mal.

Rellena `why` en todas. Obliga a construir el conjunto deliberadamente en lugar
de a golpe de memoria, y es lo que hace revisable la selección.

## Cómo usarlo

```bash
# Comprueba el recall de la consulta ya exportada
python -m nanocarbon_biblio.cli recall \
    --raw data/raw --gold queries/gold_standard.csv \
    --out data/processed/recall_report.csv
```

O desde la GUI, pestaña **7 · Validación**.

**Objetivo: recall relativo ≥ 0.95.** Córrelo una vez por cada variante de
consulta —el brazo de precisión y el de alta sensibilidad—: la diferencia entre
ambos es justo lo que justifica quedarse con uno u otro, y esa justificación
también va en Métodos.
