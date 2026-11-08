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
python modules/chamber_clamp.py            # → out/chamber_clamp.step / .stl (WE en pinza de Pt)
python modules/chamber_xrd.py              # → out/chamber_xrd.step / .stl (celda XRD)
python modules/lid_xrd.py                  # → out/lid_xrd + out/xrd_pocket_cover
python checks/walls.py                     # paredes, holguras, sellos e interferencias (200 checks)
python checks/raman_clearance.py           # objetivo y cono de luz vs cada pieza, gap 0.1–0.5
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
| 3b | `modules/chamber_clamp.py` (WE en pinza de Pt) | listo; falta medir el portaelectrodo |
| 4 | `modules/chamber_xrd.py` + `modules/lid_xrd.py` + `checks/xrd_clearance.py` | listos; falta medir el haz y la platina |
| 5 | Chladni, EPD/DEP, decoración | estimación de modos lista |
| 6 | Simulación (gmsh → FEniCSx / Elmer) | pendiente |

## Dos formas de montar el WE para Raman
La tapa Raman es la misma; cambia lo que va debajo.

| | Muestra plana sellada (`base_flat` + `chamber_ring`) | Pinza de Pt (`chamber_clamp`) |
|---|---|---|
| Cómo se sujeta | O-ring sobre la muestra; contacto por debajo con lámina | Portaelectrodo con pinza por la cara −x (1/4-28), muestra sobre un pedestal |
| Área | Definida: Ø9.5 = 70.9 mm² (solo cara superior) | Toda la tira fuera del agarre: 72 mm² arriba, 170 mm² en total; enmascarar reverso y bordes si se necesita un área definida |
| Pt en el electrolito | No | Sí: la pinza se moja (suma su propia corriente; en HER el Pt puede disolverse y redepositarse en el WE) |
| Piezas | 2 + tapa | 1 + tapa |

### Electrodos en la cámara para pinza
| Electrodo | Cara | Altura del eje | Sello |
|---|---|---|---|
| WE (portaelectrodo con pinza) | −x | z = −0.5 (plano medio de la muestra) | conector 1/4-28 sobre el cuerpo Ø4 |
| RE Ag/AgCl | −y | z = 7 | vástago Ø6 con O-ring radial |
| CE (Pt o grafito) | +y, a 6 mm del centro | z = 6 | conector 1/4-28 |
| Entrada de electrolito | +y, a 6 mm del centro, del otro lado | z = 6 | conector 1/4-28 |
| Salida | +x | z = 9 | conector 1/4-28 |

Montaje: pasar el portaelectrodo desde afuera (la pinza cerrada debe caber por el puerto, lo
verifica `checks/walls.py`), poner la tira en la pinza desde arriba con el pozo abierto, cerrar con
la tapa. **Medir el portaelectrodo** (`MEDIR` en `params.WE_CLAMP`): Ø del cuerpo, ancho de la pinza,
espesor y largo de las mandíbulas, espesor máximo de muestra.

## Ensamble Raman con muestra sellada: base + anillo + tapa (valores por defecto)
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
- **XRD:** el anillo no sirve (su borde supera la altura permitida por el haz). La celda XRD
  es un cuerpo aparte, descrito abajo.

## Celda XRD (D8 Advance θ-θ): `chamber_xrd` + `lid_xrd`
| | |
|---|---|
| Haz | a lo largo de x. Nada en las caras ±x |
| Cuerpo | 66.6 (x) × 63.8 (y) × 14 (z) mm; cara superior plana en z = 0, al ras de la muestra |
| Muestra | 18 × 18 × 1 mm por defecto; la huella cabe en ella desde 2θ = 9.4° |
| Película | Kapton (ácido) o PP/PE (KOH) sobre la cara superior, sellada con cordón EPDM bajo el marco |
| Marco tensor | aluminio, 5 mm fuera del corredor del haz; dentro, biselado a 4.5° desde 0.3 mm en |x| = 13.95 |
| Tornillos del marco | 10 × M3 en insertos, todos fuera del corredor (|y| > 8 mm) y lejos de los puertos |
| WE | portaelectrodo con pinza de Pt por +y; toma una lengüeta (tira Pt/Ti bajo la muestra, o la muestra si es flexible) 4.5 mm bajo la cara, en un bolsillo con tapa al ras |
| RE, CE, entrada, salida | cara −y, a z = −8, en un reservorio interno unido a la capa delgada por una ranura |

**Cómo se usa.** Inflar la película con electrolito para la electroquímica; para medir, retirar
electrolito por un puerto hasta que la película quede sobre la muestra (capa ≤ ~50 µm: el agua
atenúa mucho Cu Kα). La cara de la muestra se lleva al plano del goniómetro con lainas bajo la
celda o con la platina Z.

**Cómo se verifica el haz.** `checks/xrd_clearance.forbidden_region` construye el volumen que
ninguna pieza puede tocar: dentro del corredor del haz (|y| ≤ largo de la línea/2 + margen) y por
encima de la altura permitida a cada |x| para todo 2θ ≥ `two_theta_min`. `checks/walls.py`
intersecta con él el cuerpo, el marco, la tapa del bolsillo, la pinza y la muestra.

**Medir:** `XRD["beam_length_y"]` (largo de la línea sobre la muestra, depende de la rendija),
radio del goniómetro, divergencia, y la altura y huella libres de la platina.
