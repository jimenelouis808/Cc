# Celdas reportadas: qué tomar y qué evitar

Notas sobre cinco imágenes de referencia (celdas comerciales y publicadas). Las imágenes
**no se guardan en el repositorio** (varias tienen derechos reservados); aquí solo se
describen. Completar la columna de cita con el DOI de cada una.

| # | Celda | Técnica | Cita |
|---|---|---|---|
| R1 | Bloque cuadrado de PTFE, tapa roscada con ventana de cuarzo | Raman | _DOI pendiente_ |
| R2 | Celda de flujo circular en tres capas (Fig. 1 a/b) | Raman | _DOI pendiente_ |
| R3 | Celda cilíndrica para GI-XRD, WE = película polimérica metalizada (Fig. 2, ref. [14], ECS) | XRD | _DOI pendiente_ |
| R4 | Foto de un difractómetro θ-θ con potenciostato al lado | XRD | — |
| R5 | ISEC para XPS a presión ambiente, WE = ensamble con membrana (CC BY 4.0) | XPS | _DOI pendiente_ |

## R1 — Bloque de PTFE con tapa roscada (Raman)
- **Tapa roscada** que aprieta ventana y empaque: sin tornillos en la cara superior, así que el
  objetivo tiene el paso libre. Buena opción para `lid_raman` si el émbolo de `gap` cabe dentro de la rosca.
- **Todos los puertos usan la misma rosca** (conectores HPLC de fondo plano, 1/4-28 UNF, con
  férula/O-ring): RE, CE, entrada y salida son intercambiables. → `params.PORTS`. Facilita el
  adaptador de RE Ag/AgCl ↔ Hg/HgO.
- CE de alambre de Pt **enrollado** (más área en poco volumen).
- Lo que no hay que copiar: el WE entra de lado con una pinza (bandera), así que la distancia
  ventana–WE no queda definida. Nuestro diseño fija el WE en la base y referencia `gap` a z = 0.

## R2 — Celda de flujo circular en tres capas (Raman)
- Es la misma arquitectura que nuestro "sándwich": **base con contacto de cobre anular y pines guía**,
  muestra (disco), cuerpo con canal de flujo, ventana y tapa.
- El contacto del WE es un anillo **fuera del área mojada**, por abajo: el cobre nunca toca el electrolito.
- La tapa tiene un **rebaje cónico** alrededor de la ventana: deja pasar la punta del objetivo con una
  WD corta. → modelar `lid_raman` con un cono de semiángulo ≥ el del objetivo.
- CE en forma de arco alrededor del perímetro: distribución de corriente más simétrica que un alambre puntual.
- RE entra en ángulo, con capilar Luggin cerca de la muestra.
- Flujo con bomba peristáltica: pulsa. Para Raman de capa delgada (y para EPD) preferir bomba de jeringa.

## R3 — Celda para GI-XRD con WE de película metalizada
- **El WE es la pared de la celda**: una película polimérica con un recubrimiento metálico hacia el
  electrolito. Los rayos X atraviesan la película y llegan a la interfaz sin cruzar una capa de líquido.
  Es la "opción alternativa" del `CLAUDE.md` (catalizador sobre papel de carbón como pared), ya probada.
- Anillo de apriete con 6 tornillos avellanados (cabezas al ras) + O-ring: perfil bajo en la cara superior.
- Lo que no hay que copiar: los 4 conectores de manguera salen de lado a la altura del cuerpo.
  En θ-θ deben quedar por debajo de z = 0 o fuera del cono del haz (`checks/xrd_clearance.py`).
- En incidencia rasante la huella es mucho más larga que en Bragg-Brentano: si se usa GI, la holgura
  lateral manda.

## R4 — Difractómetro θ-θ
- La muestra queda en el centro de una platina pequeña, con el tubo y el detector moviéndose en un
  círculo vertical: el espacio es limitado arriba y a los lados del eje del haz.
- Los cables del potenciostato y las mangueras deben salir **perpendiculares al plano del haz**
  (hacia atrás o hacia abajo), nunca a lo largo de él.
- Pendiente de medir en el equipo: `XRD["stage_max_height"]` y `XRD["stage_max_footprint"]`.

## R5 — Celda para XPS a presión ambiente con membrana
- La tapa son **dos placas que sujetan el WE** (catalizador sobre Nafion): el mismo esquema que el
  marco doble del módulo Chladni y que la variante "WE como pared" de XRD.
- RE axial con Luggin, CE en resorte (más área), entrada y salida de líquido opuestas.
- Las placas sostienen la diferencia de presión: en XRD no hay vacío, pero la película se tensa con
  la columna de electrolito; dimensionar el anillo tensor con eso.

## Consecuencias para este proyecto
1. Puertos con una rosca estándar (`PORTS`), no agujeros a la medida de cada electrodo.
2. Contacto del WE anular, por abajo, fuera del área mojada, con pines guía (base plana).
3. Tapa Raman con rebaje cónico; nada de tornillos dentro del cono del objetivo.
4. XRD: variante "WE como pared" (R3/R5) como primera opción para catalizadores sobre papel de carbón
   o película; variante de capa delgada para electrodos macizos.
5. Conectores y cables fuera del plano del haz en XRD.
