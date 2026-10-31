# De dónde salen estos CIF

Copiados de la biblioteca de referencia de `ramancarbon`
(`ramancarbon/database/data/cif/`). Se **copian**, no se importan: los
paquetes de este repositorio son independientes y lo único que cruza
entre ellos son archivos. Así este directorio es autocontenido y nada en
`nanocarbon_lab` apunta a `ramancarbon`.

Cada uno pasó allí dos comprobaciones independientes: densidad
cristalográfica contra la literatura (valida celda y contenido a la vez)
y posición de la reflexión más intensa contra su ficha.

| archivo | fase | grupo | confianza declarada |
|---|---|---|---|
| `FeSe_tetragonal.cif` | β-FeSe, tipo PbO | P4/nmm | alta |
| `FeSe_hexagonal.cif` | δ-FeSe, tipo NiAs | P6₃/mmc | **media** |
| `Fe3C_cementita.cif` | cementita | Pnma | media |

**El hexagonal lleva un aviso que importa para estos modelos**: esa fase
es deficiente en hierro (Fe₁₋ₓSe) y a composiciones concretas ordena
como Fe₇Se₈ o Fe₃Se₄ con superestructura. Los modelos de aquí la cortan
estequiométrica 1:1, que es la celda ideal y no necesariamente la de tu
muestra.
