# insitu-cells

Celdas electroquímicas modulares y paramétricas (CadQuery) para Raman in situ (Renishaw inVia),
XRD in situ (Bruker D8 Advance θ-θ), sonoelectrodeposición, EPD/DEP de CNT/CNF y decoración con
nanopartículas. Las reglas de diseño están en [`CLAUDE.md`](CLAUDE.md); las lecciones de celdas
publicadas, en [`docs/referencias.md`](docs/referencias.md).

```bash
uv venv && uv pip install -e ".[dev]"     # o: uv sync --extra dev
python coupons/tolerance_coupon.py         # → out/tolerance_coupon.step / .stl
python checks/xrd_clearance.py             # huella y altura máxima permitida
python sim/chladni_estimate.py             # modos de la placa-WE
python -m pytest -q
```

## Estado
| Paso | Archivo | Estado |
|---|---|---|
| 1 | `coupons/tolerance_coupon.py` | listo para imprimir en PP |
| 2 | `modules/base_flat.py`, `modules/chamber_ring.py` | pendiente |
| 3 | `modules/lid_raman.py` | pendiente |
| 4 | `modules/lid_xrd.py` + `checks/xrd_clearance.py` | check listo, tapa pendiente |
| 5 | Chladni, EPD/DEP, decoración | estimación de modos lista |
| 6 | Simulación (gmsh → FEniCSx / Elmer) | pendiente |
