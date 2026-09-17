# nanocarbon

Tres paquetes independientes para caracterización y simulación de
nanomateriales de carbono, en un mismo espacio de trabajo.

| Paquete | Para qué es |
|---|---|
| **`packages/ramancarbon`** | Analiza **medidas**: Raman (carbono y TMD), difracción de rayos X con Rietveld, fotoemisión (XPS) y electroquímica (CV, carga-descarga, impedancia, HER/OER). Lectores, ajuste, identificación de fases, motor de figuras y una aplicación gráfica. |
| **`packages/carbonforge`** | Prepara **cálculos**: Quantum ESPRESSO, SIESTA, LAMMPS; campos de fuerza, celdas EDLC y lectura de resultados. |
| **`packages/nanocarbon_lab`** | Genera **estructuras**: nanotubos, fulerenos, haeckelitas, schwarzitas, uniones, espumas, TMD, dopantes y defectos. |

La interfaz de usuario está en español; el código y los docstrings, en inglés.

## Empezar

```bash
uv sync --all-packages --extra dev      # instala los tres
uv run ramancarbon-gui                  # la aplicación gráfica
```

Instalación detallada, incluido Tkinter y el caso sin `uv`, en
[`INSTALACION.md`](INSTALACION.md). Lo que ha cambiado en la última ronda,
en [`NOVEDADES.md`](NOVEDADES.md).

## Pruebas

```bash
cd packages/ramancarbon
OMP_NUM_THREADS=1 uv run python -m pytest ramancarbon/tests -q -n 4 --dist loadscope
```

`OMP_NUM_THREADS=1` y `--dist loadscope` no son opcionales: el porqué está en
[`CLAUDE.md`](CLAUDE.md).

## La regla que no se rompe

**Los tres paquetes son independientes y siguen siéndolo.** Ninguno importa a
otro. Cada uno tiene su `pyproject.toml`, su versión, sus pruebas y su script
de consola, y cada uno debe poder instalarse y pasar el linter por separado.

## Dónde está el conocimiento del proyecto

Los archivos `CLAUDE.md` no son documentación de cortesía: son el registro de
**cada decisión que se tomó porque su alternativa dio un resultado
equivocado**, con el número que lo demostró. Si vas a tocar algo, léelos
primero.

- [`CLAUDE.md`](CLAUDE.md) — la regla del espacio de trabajo y cómo trabajar aquí.
- [`packages/ramancarbon/CLAUDE.md`](packages/ramancarbon/CLAUDE.md) — las barreras físicas de Raman, DRX, XPS y electroquímica.
- [`packages/nanocarbon_lab/CLAUDE.md`](packages/nanocarbon_lab/CLAUDE.md) — geometría, periodicidad y relajación.
- [`packages/carbonforge/CLAUDE.md`](packages/carbonforge/CLAUDE.md) — preparación de cálculos.

## Honestidad sobre la validación

Todo está validado contra datos **sintéticos** generados por el propio
paquete. Eso comprueba las fórmulas y la lógica, no la exactitud sobre
medidas reales. Nada aquí está contrastado contra un patrón certificado.

## Licencia

[MIT](LICENSE).
