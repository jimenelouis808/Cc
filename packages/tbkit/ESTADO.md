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

## En curso (si el contenedor se reinició, relanzar: todo retoma de sus partes/checkpoints)
```bash
cd packages/tbkit
# referencias GPAW de Se (falta coronene_SeH); partes en /tmp/claude-0/se/gpaw_se.parts
uv run python -m tbkit.recipes.bsp_references Se /tmp/claude-0/se/gpaw_se.json --workers 4
# reajustes B y P con active learning + repulsión X-H hasta 2,4 Å + hessianas
R=tbkit/parameters/references
uv run python -m tbkit.recipes.xu_bsp B $R/gpaw_b.json $R/gpaw_b_al.json /tmp/claude-0/fit/xu_chnob.json --hessians $R/gpaw_b_hessians.json
uv run python -m tbkit.recipes.xu_bsp P $R/gpaw_p.json $R/gpaw_p_al.json /tmp/claude-0/fit/xu_chnop.json --hessians $R/gpaw_p_hessians.json
```
(con `OMP_NUM_THREADS=1`; las partes en /tmp se pierden al reiniciar el contenedor.)

## Siguiente
1. Comprobar B/P: relajar B(OMe)3 y PO(OMe)3 (sin colapso), RMS de frecuencias;
   instalar, quitar la nota de ésteres de `validity_notes`, tests.
2. Se: copiar `gpaw_se.json` a references, hessianas (`frequency_references`),
   ajuste `xu_chnose`, active learning, instalar; `chnose` en CLI y GUI; tests; README.
3. Polarizabilidad extra de B, S, P, Se (`chn_polarizability`, conjuntos b/s/p + Se).
4. Términos angulares generales + factores de escala (Witek 2004).
5. SCC periódico con Ewald (Elstner 1998).
6. TB dependiente del entorno (Tang 1996).
7. ML: Δ-learning de repulsión (Stöhr 2020), fonones híbridos MACE con α/μ de TB,
   DeePTB; GFN2-xTB como motor de comparación opcional.
8. Validación en sistemas reales; anomalía de Kohn (Piscanec 2004).
9. Al final: documentación de métodos/validación y guía de usuario.
