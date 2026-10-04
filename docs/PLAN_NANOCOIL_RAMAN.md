# Plan: Raman de nanocoils 1D de carbono (5–7) con TB y GPAW

Destilado de *Nanocoil_Raman_GPAW_ClaudeCode_MasterPlan.md* (documento del
usuario, no versionado aquí): lo que se alinea con la suite y cómo se reparte.
La regla del workspace manda: **ningún paquete importa a otro**; la estructura
pasa de `nanocarbon_lab` a `tbkit` como archivo (extxyz con celda y pbc).

## Objetivo

Relación cuantitativa estructura → topología 5–7 → curvatura → estructura
electrónica → fonones → polarizabilidad → Raman a 532 nm (2.331 eV), para una
nanocoil periódica en z (~200 C/celda), después dopada (N, B, S, Se) y
funcionalizada (O, OH, COOH, NH2), comparada con el Raman experimental.

## Qué ya existe (no duplicar)

| Necesidad del plan | Dónde está |
|---|---|
| Construir el coil periódico, catálogo de coils medidos | `nanocarbon_lab/builders/periodic_coil.py`, `nanocoil.py` |
| Anillos, conteo 5/6/7, Euler | `nanocarbon_lab/dopants/rings.py`, auditorías de mallas |
| Relajar y fonones Γ con TB (Xu, xu_ch*) | `tbkit.tasks.relax`, `tasks.phonons`, `modes.py` |
| Clasificación de modos (participación, carácter, simetría, VDOS) | `tbkit/modes.py` |
| Raman no resonante, resonante, perfiles de excitación | `tbkit/raman.py`, `resonance.py` |
| Raman con fonones de QE | `tbkit/qe.py` |
| Estructura electrónica, DOS/PDOS, bandas Γ–Z | `tbkit/analysis.py`, `tasks` |
| TB frente a GPAW, campañas reanudables con supervisor | `tbkit/recipes/crystal_validation.py`, `carbon_environments.py`, `run_crystals.sh` |
| TB dependiente del entorno (curvatura, 5–7, sp2/sp3 mezclados) | `tbkit/environment.py` + `tang_carbon` (paso 6) |
| Dopaje y funcionalización con TB | conjuntos `xu_chn/chno/chnob/chnos/chnop/chnose` |

## Fases (en orden; no se escala sin validar la anterior)

1. **Estructura** (nanocarbon_lab): exportar el coil periódico C~200 con su celda
   (Lz = periodo, vacío transversal), informe de topología: anillos 5 y 7, pares
   5–7, longitudes y ángulos C–C, curvatura, torsión. Sin modificar la original.
2. **Línea base TB** (tbkit): relajar con `tang_carbon` y con Xu; comprobar que los
   pares 5–7 sobreviven (si no: `TOPOLOGY_WARNING`, no se corrige solo); fonones
   Γ; clasificar modos (radial/tangencial/torsional, localización en 5, 7,
   curvatura +/−) por vectores propios, nunca solo por frecuencia.
3. **Referencia GPAW** del mismo C~200: convergencia de k en z y de vacío,
   relajación PBE (reanudable, como las campañas existentes), y comparación
   TB↔GPAW de geometría, fuerzas, bandas y gap.
4. **Raman estático** (TB, y con fonones GPAW/QE si se calculan): espectro con
   ensanchamiento registrado (Gauss/Lorentz/Voigt), asignación por modos en
   `raman_mode_assignment.md`; D, G, D′, 2D y modos de baja frecuencia sin
   asignar por posición sola.
5. **Electrónica y óptica**: DOS/PDOS, bandas Γ–Z, ¿hay transición cerca de
   2.331 eV? (con el aviso de que las energías TB/KS no son ópticas).
6. **Raman a 532 nm**: resonante solo si la fase 5 lo justifica; para el coil
   gapless o casi, la vía es la de doble resonancia (como `graphene.py`), no la
   α estática.
7. **Comparación experimental**: cargar `experimental_raman.txt` sin alterarlo;
   posiciones, D/G, RMSE, correlación, desplazamiento óptimo explícito.
8. **Dopaje** (C199X, X = N, B, S, Se) por sitio: cerca/lejos de 5–7, curvatura
   +/−; energía de formación y Raman por configuración.
9. **Funcionalización** (O, OH, COOH, NH2), siempre relajada antes del Raman.

## Registro

`COMPUTATIONAL_MATRIX.md`, `TODO.md` (P0–P6 del documento original) y
`RESULTS.md` se crean al empezar la fase 1 en el paquete que corresponda;
cada cálculo guarda estructura, celda, parámetros, versiones, energía, fuerzas
y convergencia (los `.gpw` fuera de git).
