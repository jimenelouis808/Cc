# insitu-cells

Celdas electroquímicas modulares y paramétricas (CadQuery) para Raman in situ (Renishaw inVia),
XRD in situ (Bruker D8 Advance θ-θ), sonoelectrodeposición, EPD/DEP de CNT/CNF y decoración con
nanopartículas. Las reglas de diseño están en [`CLAUDE.md`](CLAUDE.md); las lecciones de celdas
publicadas, en [`docs/referencias.md`](docs/referencias.md).

```bash
uv venv && uv pip install -e ".[dev]"     # o: uv sync --extra dev
python coupons/tolerance_coupon.py         # → out/tolerance_coupon.step / .stl
python modules/base_flat.py                # → out/base_flat.step / .stl
python modules/chamber_ring.py             # → out/chamber_ring.step / .stl + resumen
python modules/lid_raman.py                # → out/lid_raman.step / .stl + resumen
python checks/walls.py                     # paredes, holguras, sellos e interferencias (76 checks)
python checks/raman_clearance.py           # objetivo y cono de luz vs base/anillo/tapa, gap 0.1–0.5
python checks/xrd_clearance.py             # huella y altura máxima permitida
python sim/chladni_estimate.py             # modos de la placa-WE
python -m pytest -q
```

## Estado
| Paso | Archivo | Estado |
|---|---|---|
| 1 | `coupons/tolerance_coupon.py` | listo para imprimir en PP |
| 2 | `modules/base_flat.py`, `modules/chamber_ring.py` | listos; checks de pared y Raman en verde |
| 3 | `modules/lid_raman.py` | listo; falta probar con agua (y medir el objetivo) |
| 4 | `modules/lid_xrd.py` + `checks/xrd_clearance.py` | check listo, tapa pendiente |
| 5 | Chladni, EPD/DEP, decoración | estimación de modos lista |
| 6 | Simulación (gmsh → FEniCSx / Elmer) | pendiente |

## Ensamble Raman: base + anillo + tapa (valores por defecto)
| | |
|---|---|
| Huella | 54 × 54 mm; base 6 mm, anillo 13 mm, placa de tapa 5 mm (sobre lainas) |
| Área expuesta del WE | Ø9.5 mm = 70.9 mm² (ID del O-ring sobre la muestra) |
| WE rehundido | 2.75 mm en el piso del anillo |
| O-ring sobre la muestra | 9.5 × 1.0 EPDM; tope duro → compresión fija (25 %) |
| Capa delgada | gap 0.3 mm nominal, 0.1–0.5 mm con lainas DIN 988 3×6 de 0–0.4 mm |
| Ventana | Ø6 × 0.5 (cuarzo en ácido, zafiro en KOH), pegada al ras en la punta de la nariz |
| Nariz del émbolo | Ø9.0 en el agujero de Ø9.5; pared ≥ 2 mm alrededor del cono de luz de NA 0.5 |
| O-ring de la tapa | 33 × 1.5 EPDM; compresión 12–38 % en todo el recorrido de lainas |
| Tornillos | anillo → base: 4 × M3×14 + tuerca; tapa → anillo: 4 × M3×6 en insertos (engrane ≥ 3.6 mm) |
| Pines guía | 2 × Ø3×8, posiciones asimétricas (una sola orientación) |
| Puertos | entrada −x (z 5.5), salida +x (z 9, purga), CE +y (1/4-28), RE −y (O-ring radial) |

### Cómo se modela el objetivo
Cara frontal + nariz cónica + barril (`params.RAMAN`), más `clearance_margin`. La punta queda en
`z_tip = WD + window_t·(1 − 1/n_w) + gap·(1 − 1/n_e)` = 8.23 mm: cada capa plana alarga el foco
t·(1 − 1/n). El cono de luz de la NA se traza con refracción en electrolito, ventana y aire.
El hueco de la tapa cubre objetivo y cono **en todo el recorrido de lainas**: con gap 0.5 la tapa
sube 0.2 mm, pero el objetivo, enfocado en el WE, solo 0.05 mm.

**Medir antes de fabricar** (`MEDIR` en `params.RAMAN`): Ø de la cara frontal, ángulo de la nariz,
Ø del barril, WD y NA. Con un objetivo de cara frontal Ø25 mm (el valor original) la tapa no cabe:
lo detecta `checks/walls.py` (prueba `test_barrel_sized_objective_tip_is_rejected`).

### Limitaciones conocidas
- **Caída óhmica:** RE y CE viven en el pozo; la corriente al WE pasa por la rendija de 0.25 mm
  entre la nariz y el agujero, y luego por la capa de 0.3 mm. Medir la resistencia no compensada
  (EIS a alta frecuencia) y compensarla; para electroquímica pesada, subir el gap con lainas.
- **Flujo:** la entrada y la salida renuevan el pozo, no la capa delgada (allí manda la difusión).
- **Aire atrapado:** el pozo no tiene venteo por arriba. Llenar con la celda inclinada y la salida
  (+x) en lo más alto.
- **Ventana pegada:** verificar el adhesivo contra 3 M KOH y 0.5 M H2SO4 (cupón sumergido 48 h).
- **Material de la tapa:** la nariz y las paredes del embudo son finas: PEEK o PTFE maquinado.
  PP impreso solo para pruebas de ajuste.
- **XRD:** el anillo no sirve. Su borde supera por mucho la altura permitida en el cono del haz
  (< 1 mm a 2θ bajos). El módulo XRD necesita su propio cuerpo de perfil bajo (paso 4).
