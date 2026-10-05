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
   extiende a P/B. ¿Son los orbitales d? Prueba con GPAW relajando desde el mínimo dzp con
   la base sin d solo en el heteroátomo (`dz(dzp)`): CH3SeO2H no se mueve (0,05 Å; con d
   0,07) -> en Se NO son los d; H3PO4 tampoco (0,13 frente a 0,06); MePO3H2 sí (los OH giran
   ~150°, P-O +0,05 Å; con d 0,09 Å), pero TB gira en otra dirección. B: no es el OH sino el
   B(OH)2 coplanar con el fenilo (GPAW 23°), probable falta de repulsión H-H (H···H 1,95 Å
   frente a 2,13). Hecho: `HHContactTerm` (pared H2···H2 de GPAW) instalado en los 6 conjuntos
   sin reajuste (corte quíntico 2,0-2,4 Å): PhB(OH)2 0,61 -> 0,07 Å, coroneno-SH 1,08 -> 0,14,
   coroneno-SeH 1,18 -> 0,26, PEt3 0,16 -> 0,13. OH de ácidos selenínicos/fosfónicos: causa medida = base mínima (en el
   giro rígido la repulsión TB es plana y la parte electrónica hunde 240-300°; GPAW con base
   sz hace lo mismo). Sin arreglo dentro de una base sp mínima: limitación declarada. Factores de escala (Witek 2004): hechos
   (λ 1,003-1,037 frente a GPAW, en cada archivo; `tasks.phonons` da las escaladas aparte).
5. (hecho) SCC periódico con Ewald (`ewald.py`): Madelung, caja grande = finito, fuerzas FD.
   Hecho también: validación en 13 cristales (arriba) y α de respuesta lineal SCC periódica
   (campos locales de carga; los dipolares en cristales siguen sin Ewald).
6. (hecho) TB dependiente del entorno: `environment.py` (Tang et al. 1996) y conjunto
   `tang_carbon` (`recipes/tang_fit.py`): parte electrónica publicada; escala de Δe (×0,25), corte
   (3,0-3,6 Å) y repulsión ajustados a GPAW (`gpaw_carbon_env.json`). Fuerzas frente a Xu (eV/Å):
   Stone-Wales 0,93 -> 0,38 y amorfo 3,2 1,69 -> 0,98 (ambos fuera del ajuste), C60 0,82 -> 0,42,
   vacante 0,79 -> 0,65; grafeno 0,12 -> 0,14. Diamante a = 3,515 Å; energías entre fases poco
   fiables (diamante 0,37 eV/átomo). En GUI y CLI como «tang». Los parámetros publicados tal cual
   no dan el diamante del artículo (`tang1996_published.json`, solo pruebas).
   Planes nuevos: `docs/PLAN_NANOCOIL_RAMAN.md` (siguiente proyecto, usa tang_carbon) y
   `docs/PLAN_XPS_FUTURO.md` (a futuro).
   Nanocoil (204 átomos, `recipes/nanocoil.py`, `validation/nanocoil204_tb.json`): fases 2-3
   hechas (TB Xu/Tang, relajación PBE, gap PBE 0,22 eV frente a TB sin gap, enlaces TB-PBE
   RMS 0,015 Å, modos proyectados TB 2-6 % sobre PBE) y Raman resonante Tang (609 modos;
   posiciones robustas, alturas relativas dependientes de η). Pendiente de consistencia:
   los fonones TB de la coil usan 4 k en z en un sistema sin gap; no se ha comprobado su
   convergencia en k (ver punto 8).
   Herramientas generales: `sites.py` (anillos, grupos, frecuencia proyectada),
   `phonopy_bridge.py` (pestaña «Fonones (ZB)»), SCC sí/no y modos por sitio en la GUI,
   `recipes/site_screening.py`, `resonant_raman(select=..., cache_dir=...)`.
7. Hecho (medido, no adoptado): MACE-MP-0 sin ajuste fino (moléculas como xu_chno; coil
   11-18 % blanda) y Δ-learning lineal SOAP (no generaliza con los datos guardados).
   Infraestructura probada: `hybrid.py`, `delta.py`, `recipes/delta_fit.py`. No hechos, con
   razón: DeePTB (otro modelo entero, no una mejora de estos conjuntos) y GFN2-xTB
   (comparador; solo si hace falta un tercer método). Lo que sí ayudaría: más datos DFT
   (campaña de active learning) o ajuste fino de MACE con nuestras referencias.
8. Validación en sistemas reales; anomalía de Kohn (Piscanec 2004). Hecho: malla k de los
   fonones sin gap (grafeno con Xu: G 1572 con 12 k -> 1674 convergida; casi sin
   dependencia con kT una vez convergida: la anomalía de Xu es débil). Aviso automático en
   `modes.vibrations`/`tasks.phonons`, `tasks.kmesh_convergence`, fonones Xu del grafeno a
   6×6 × 8×8 k. Falta: pendiente de Kohn física (requiere DFT con k densa o GW), tubos
   metálicos.
9. Hecho: `docs/GUIA_USUARIO.md`, `docs/METODOS.md` y `docs/VALIDACION.md` (este, generado
   desde los datos por `recipes/validation_report.py`, con test). README corregido: tabla de
   módulos (SCC periódico, Raman/IR, Tang, sitios/phonopy), límites (había GUI y Raman
   resonante sin mencionar), I(2D)/I(G) a 2,80 eV 13,0 (no 13,1), regla ω > 100 cm⁻¹ del IR.

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
