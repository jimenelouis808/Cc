# Superestructuras ya construidas

Un archivo `.xyz` (XYZ extendido) por cada construcción distinta de los
presets geodésicos y de Goldberg, y del catálogo
(`nanocarbon_lab/examples/ex12_catalogue.py`). Así no hay que volver a
generarlas: algunas tardan horas.

- El nombre son los parámetros: `<grafo>_m1n0_f4_r2_w1.xyz` es GC(m, n),
  frecuencia de la esfera geodésica, filas de antiprisma por brazo y
  radio de ventana. Los de Goldberg son `goldberg-<mapa>_m2n0.xyz`.
- La línea de comentario lleva la celda (`Lattice`), la periodicidad
  (`pbc`), los parámetros, el censo de anillos (`census`, tamaño:número),
  el veredicto sp2 y la fracción de disclinaciones bien colocadas
  (`placed`).
- `index.csv` resume todo y dice qué presets usan cada archivo.

Se leen con ASE (`ase.io.read(path)`), OVITO o VESTA. Para completar o
regenerar:

```bash
OMP_NUM_THREADS=1 uv run python -m nanocarbon_lab.examples.ex13_export_structures \
    catalogo/xyz --cache catalogo/cache --workers 2
```

Un archivo que ya existe no se reconstruye.
