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
