# Estado del trabajo en curso (tbkit: heteroátomos, IR/Raman)

Nota de traspaso entre sesiones: qué está hecho, qué corre y qué sigue, para
que una sesión nueva retome sin reconstruir el contexto. Se actualiza en cada
commit del plan y se borra al terminarlo (la documentación final la reemplaza).

## Hecho
- `xu_chno` reajustado con hessianas y `h_tail`; `xu_chnob/s/p` instalados
  (RMS de frecuencias 46-149 cm⁻¹ por molécula salvo los ésteres B(OMe)3 y
  PO(OMe)3, que colapsan).
- Receta `recipes/active_learning.py`; datos: `parameters/references/gpaw_b_al.json`
  (8 estructuras: B(OMe)3, B2(OH)4) y `gpaw_p_al.json` (12: PO(OMe)3, MePO3H2, H3PO4).
- Familia Se definida en `recipes/xu_bsp.py` (U, d, pares); moléculas en
  `recipes/bsp_references.py`.

## En curso
- Instalados (51240a2): `xu_chnop` (2 rondas AL, éster corregido) y `xu_chnose`.
- Reajuste de B sin término angular, con AL + torsiones (si se reinició, relanzar):
```bash
cd packages/tbkit; export OMP_NUM_THREADS=1; R=tbkit/parameters/references
uv run python -m tbkit.recipes.xu_bsp B $R/gpaw_b.json $R/gpaw_b_al.json $R/gpaw_b_torsion.json /tmp/claude-0/fit/xu_chnob4.json --workers 4 --hessians $R/gpaw_b_hessians.json
```
Criterio: B(OMe)3 sin colapso (RMS < 150 cm⁻¹), resto no peor; instalar con nota de
validez (B-OH como P-OH si gira) y quitar la nota de ésteres.

## Siguiente
1. Comprobar B tras el reajuste de arriba: relajar B(OMe)3 y PO(OMe)3 (sin colapso), RMS de frecuencias;
   instalar, quitar la nota de ésteres de `validity_notes`, tests.
2. (hecho) Se instalado; falta sección de Se en README.
3. Polarizabilidad extra de B, S, P, Se (`chn_polarizability`, conjuntos b/s/p + Se).
4. Términos angulares generales + factores de escala (Witek 2004).
5. SCC periódico con Ewald (Elstner 1998).
6. TB dependiente del entorno (Tang 1996).
7. ML: Δ-learning de repulsión (Stöhr 2020), fonones híbridos MACE con α/μ de TB,
   DeePTB; GFN2-xTB como motor de comparación opcional.
8. Validación en sistemas reales; anomalía de Kohn (Piscanec 2004).
9. Al final: documentación de métodos/validación y guía de usuario.

## Diagnóstico abierto: H de OH que migra al O vecino (B, P, Se)
- Se ajustado (`/tmp/claude-0/fit/xu_chnose2.json`): enlaces < 0,08 Å, frecuencias
  64-124 cm⁻¹; único espurio CH3SeO2H: el H del OH forma puente con el O de Se=O
  (GPAW +0,32 eV, TB −0,31 eV). Mismo patrón en MePO3H2 y H3PO4.
- No lo arregla ni pesar ×30 las energías de active learning (destroza fuerzas)
  ni la repulsión Se-H larga (probado con la resolución lineal a parámetros fijos).
- Hipótesis SCC descartada (medido): el amortiguamiento γʰ de DFTB3 lo empeora
  (−0,31 → −0,51 eV) y suavizar γ de pares con H no lo mueve. Las repulsiones de par
  a esas distancias son cero.
- Causa real: rigidez angular en el centro hipervalente. El modelo cierra O-Se-O de
  109° a 94° y gira el OH (diedro 178° → 86°) hasta dejar H a 2,16 Å del otro O.
  Base sp mínima sin orbitales d (el mismo problema que DFTB 3ob con P/S hipervalentes).
- Arreglo: paso 4 del plan adelantado: términos angulares generales centrados en
  el heteroátomo (extender AcuteAngleTerm a X-centrado, ajustado con datos), sin
  tocar xu_chno. Luego reajustar B, P, Se (S si lo necesita).
