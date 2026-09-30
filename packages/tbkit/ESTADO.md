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

## Hecho en esta etapa
- Instalados: `xu_chnob` (AL + torsiones; éster 949 -> 95 cm⁻¹), `xu_chnop` (2 rondas
  AL; éster 1220 -> 86 cm⁻¹), `xu_chnose` (nuevo). Límite declarado: X-OH de ácidos
  fosfónicos/seleníninicos/borónicos arílicos gira al relajar.

## Hecho: los conjuntos en cristales frente a GPAW (paso 5, validación)
- 13 cristales (`recipes/crystal_validation.py`, `validation/crystals_tb_vs_gpaw.json`,
  referencias `parameters/references/gpaw_crystals.json`, GPAW 26.7): la extrapolación
  aguanta; fuerzas 0,3-1,4 veces el error molecular del mismo conjunto, enlaces dentro de
  lo que cada `validity` ya declaraba. Nuevo: altura de S/P/Se sobre la hoja (0,06-0,12 Å),
  red 0,5-1 % corta (h-BN +1,1 %). Cada `validity` lo dice (`crystal_notes` en la receta).
- De paso: el C-O simple de alcoholes sale 0,06-0,08 Å corto en moléculas (en validity).
- GPAW sin MPI en un venv aparte: `apt install libxc-dev libopenblas-dev g++`,
  `CC=g++ uv pip install gpaw`. Checkpoints en `out/crystals/` (git lo ignora).

## Siguiente
1. (hecho) B/P/Se instalados. Antes: relajar B(OMe)3 y PO(OMe)3 (sin colapso), RMS de frecuencias;
   instalar, quitar la nota de ésteres de `validity_notes`, tests.
2. (hecho) Se instalado; falta sección de Se en README.
3. (hecho) Polarizabilidad extra: B 0.873, S 2.642, P 2.647, Se 3.496 Å³ (GPAW FD).
4. Términos angulares: probado O-X-O solo (`--angular-ligands O`) en Se: activo pero no
   arregla CH3SeO2H (deriva 0,87 -> 0,83 Å; SeO2 mejora, DMSeO empeora). Descartado; no se
   extiende a P/B. La causa es la base sin d. Factores de escala (Witek 2004): en curso.
5. (hecho) SCC periódico con Ewald (`ewald.py`): Madelung, caja grande = finito, fuerzas FD.
   Hecho también: validación en 13 cristales (arriba). Falta: α de respuesta lineal SCC periódica.
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
