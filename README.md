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
python checks/walls.py                     # paredes mínimas e interferencias (50 checks)
python checks/raman_clearance.py           # punta del objetivo vs anillo
python checks/xrd_clearance.py             # huella y altura máxima permitida
python sim/chladni_estimate.py             # modos de la placa-WE
python -m pytest -q
```

## Estado
| Paso | Archivo | Estado |
|---|---|---|
| 1 | `coupons/tolerance_coupon.py` | listo para imprimir en PP |
| 2 | `modules/base_flat.py`, `modules/chamber_ring.py` | listos; checks de pared y Raman en verde |
| 3 | `modules/lid_raman.py` | pendiente |
| 4 | `modules/lid_xrd.py` + `checks/xrd_clearance.py` | check listo, tapa pendiente |
| 5 | Chladni, EPD/DEP, decoración | estimación de modos lista |
| 6 | Simulación (gmsh → FEniCSx / Elmer) | pendiente |

## Ensamble base + anillo (valores por defecto)
| | |
|---|---|
| Huella | 54 × 54 mm, base 6 mm + anillo 13 mm |
| Área expuesta del WE | Ø6 mm = 28.3 mm² (ID del O-ring sobre la muestra) |
| WE rehundido | 3.1 mm en el piso del anillo |
| O-ring sobre la muestra | ID 6 × cs 1.5 EPDM; tope duro → compresión fija (25 %) |
| Tornillos | 4 × M3×14 cabeza cilíndrica + tuerca embebida en la base |
| Pines guía | 2 × Ø3×8, posiciones asimétricas (una sola orientación) |
| Puertos | entrada −x, salida +x (más alta, purga burbujas), CE +y (1/4-28), RE −y (O-ring radial) |
| Tapas | 4 insertos térmicos M3 en (±15, ±15); el sello va en cada tapa |

El anillo **no sirve para XRD**: su borde (13 mm a 14 mm del centro) supera por mucho la altura
permitida en el cono del haz (< 1 mm a 2θ bajos). El módulo XRD necesita su propio cuerpo de perfil
bajo (paso 4), probablemente con la variante "WE como pared" de `docs/referencias.md`.

**Aviso para `lid_raman` (paso 3):** para llegar a `gap` = 0.3 mm, la nariz del émbolo con la
ventana debe entrar en el agujero de Ø`exposed_d` = 6 mm, que tiene 3.1 mm de profundidad. Eso deja
una apertura óptica de ~4 mm y poco espacio para el cono de luz del objetivo 50x LWD. Si al diseñar
la tapa no alcanza, la salida simple es subir `exposed_d` a 10–12 mm (y el O-ring a juego).
