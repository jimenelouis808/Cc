# Plan de la suite: carbonforge unificado, Raman y tight binding

Documento vivo. Cada fase se propone con detalle y se aprueba antes de empezar;
aquí se tachan las que están hechas y se anotan las decisiones, para que
cualquier sesión futura sepa qué se decidió y por qué.

---

## Decisiones tomadas (26-09-2026)

| # | Pregunta | Decisión |
|---|---|---|
| 1 | ¿vibspec en su propia ventana o dentro de la de carbonforge? | **Dentro de carbonforge.** Una sola GUI; vibspec es un subpaquete de carbonforge, así que no toca la regla de independencia. |
| 2 | ¿Dónde vive el tight binding (TB)? | **Cuarto paquete independiente** (`packages/tbkit`, nombre provisional). Intercambia estructuras por archivo (extxyz), como vibspec con nanocarbon_lab. Integrar su pestaña en la ventana de carbonforge exigiría una dependencia entre paquetes: se decide cuando TB tenga GUI, no antes. |
| 3 | Toolkit gráfico | **Tk para carbonforge** (unificación incluida). Para TB, GUI con gráficos avanzados (isosuperficies de orbitales, mapas de densidad): se elige el toolkit al empezar su GUI; candidato PySide6 + pyvista. |
| 4 | "Magnetización frente a la energía" | **Las tres:** (a) m(E) = ∫^E [ρ↑ − ρ↓] dE′ y su derivada dm/dE = ρ↑ − ρ↓; (b) m frente al campo (energía Zeeman) y la susceptibilidad; (c) m frente al dopaje o nivel de Fermi. |
| 5 | Parámetros de TB | **Las dos vías:** leer archivos `.skf` de DFTB (mio/3ob; revisar licencia antes de redistribuir) y ajustar parámetros propios a cálculos GPAW de vibspec. |

La plataforma de la suite completa es **Linux** (GPAW solo funciona ahí).
Windows sigue sirviendo para construir, preparar y analizar, con las pestañas
de cálculo desactivadas y explicando por qué.

---

## Fase A — Salud de carbonforge (en curso)

- [x] `find_sites` contaba el H como vecino: un C–H de borde salía "basal", y el
      dopado o los grupos basales podían caer en un borde.
- [x] `introduce_vacancies` podía quitar hidrógenos en estructuras pasivadas.
- [x] Dopante colocado junto a un grupo funcional (y grupo invertido respecto al
      plano), visto por el usuario al construir desde la GUI.
- [x] Control de colocación: borde armchair / zigzag, basal, anillos de 5 y 7
      (defecto Stone-Wales), vecinos de un defecto, índices explícitos, y una
      separación mínima entre heteroátomos y grupos.
- [x] Advertencias corregibles en la propia GUI: cada aviso lleva, cuando la
      hay, una corrección que se aplica con un botón.
- [x] Catálogo de capacidades por código (QE, SIESTA, GPAW):
      `carbonforge/codes/`, parámetros avanzados con tipo, valores válidos y
      descripción, editables en la GUI (`gui/advanced.py`) y pasados a los
      escritores; importación de `INPUT_PW.def` (QE), `siesta.tex` (SIESTA) e
      introspección de GPAW. El TB registrará su propio catálogo en su paquete
      (misma forma, sin importar carbonforge).
- [x] `make_pyridinic_n` con un solo N deja dos carbonos colgantes: avisar.
- [x] Los 2 errores de ruff preexistentes en CI; ruta `dopants.base` inexistente
      citada en `packages/carbonforge/CLAUDE.md`.
- [x] Encontrado de paso: `stone_wales_defect` colapsaba los dos átomos del
      enlace en el mismo punto (normales de signo opuesto promediadas a cero).
- [x] Partir `gui/app.py` (1400 líneas) por pestañas: primer paso de la fase B.
      `gui/tabs/` (builder, preview, importing, edlc, analysis); `app.py` queda
      en ~200 líneas con lo compartido (cola de trabajos, formularios, errores).

## Fase B — GUI unificada

- Secciones por tarea: **Estructura** (construir, importar, biblioteca,
  comprobaciones; una "estructura actual" compartida), **Preparar** (QE,
  SIESTA, LAMMPS, EDLC), **Calcular** (GPAW IR; después Raman y TB),
  **Resultados** (bandas, DOS, espectros QE, IR frente a FTIR, modos).
- Generalizar la cola de trabajos y `record.json` de vibspec a todos los
  motores: cada cálculo lanzado desde la GUI queda reproducible y cancelable.
- Retirar la ventana separada de vibspec cuando todo esté en la principal
  (`carbonforge vibspec gui` pasa a abrir la ventana principal en su sección).

Hecho (primer bloque):

- [x] Ventana por secciones: Estructura (Construir, Importar, Modelo finito
      (IR)), Preparar (Celda EDLC), Calcular (IR con GPAW), Resultados
      (Bandas y espectros, IR frente a FTIR). Las páginas de vibspec se
      construyen al abrirlas por primera vez.
- [x] "Estructura actual" compartida (`gui/session.py`): se publica al
      construir, importar o armar un modelo; se toma con un botón. vibspec
      acepta estructuras en memoria (`core.load_atoms`) con las mismas
      comprobaciones que un archivo.
- [x] `carbonforge vibspec gui` abre la ventana principal en Modelo finito.

Pendiente:

- [x] Exportación QE/SIESTA/LAMMPS en Preparar → Cálculo, sobre la
      estructura actual (receta, ajustes, avanzados, correcciones, formatos).
      La celda EDLC también usa la estructura actual.
- [x] Una cola para todos los motores (`carbonforge/jobs/`): `job.json` con
      los pasos de cada exportación QE/SIESTA/LAMMPS, runner reanudable
      (`python -m carbonforge.jobs.run DIR`), página Calcular → Trabajos, y
      "Encolar al exportar" en Preparar. vibspec usa la misma cola.
- [x] Resultados: DOS/PDOS en la ventana y modos de QE (desde `dynmat.axsf`,
      clic en una banda → animación), con los mismos ayudantes de modos que
      vibspec (`results/modes.py`).
- [x] Proyectos de receta con `job.json` (relajación → `update-geometry` →
      propiedad, con pools de puntos k).

Fase B cerrada.

## Fase C — Raman en vibspec

- Primer paso barato: conectar la exportación DFPT de QE (`ph.x` con
  `lraman`), que carbonforge ya escribe, al modelo finito de vibspec
  (requiere pseudopotenciales norm-conserving y gap).
- Con GPAW: derivadas de la polarizabilidad por campo finito × desplazamientos
  (caro); resonante con `ase.vibrations.ResonantRaman` más adelante.
- Análisis: el mismo de la fase 3 de vibspec (ensanchado, factor de escala,
  emparejamiento), con el factor de Bose y el (ν_láser − ν)⁴ que ya aplica
  `results.spectra.broaden`.

Hecho:

- [x] QE conectado al modelo finito: `asr='zero-dim'` en `dynmat.in` para
      estructuras finitas; "Frente al experimento" y `vibspec plot` aceptan un
      `dynmat.out` como espectro calculado (se quitan los modos rígidos < 100 cm⁻¹).
- [x] Raman en el flujo de vibspec (`core/raman.py`): `CalcSpec.raman` =
      `field` (polarizabilidad DFT por ±E, 36N SCF) o `bond` (Lippincott–Stuttman,
      empírico); 6N geometrías con caché reanudable; actividad 45a'²+7γ'² en
      Å⁴/amu y razón de despolarización sobre los modos del IR.
- [x] Análisis y GUI: tipo IR/Raman, Raman experimental como intensidad,
      factores de láser y Bose en el análisis (nunca almacenados), `vibspec
      plot --kind raman --raman-exp`.

Pendiente de la fase C:

- [ ] Raman resonante (`ase.vibrations.ResonantRaman`/Placzek con TDDFT): caro;
      solo si el láser cae cerca de una transición del modelo.
- [ ] Probar `field` con GPAW real (aquí no hay GPAW): comparar con la
      referencia de benceno o coroneno publicada antes de usarlo en cintas.

## Fase D — Paquete de tight binding

Motor, por capas y en este orden:

1. TB π ortogonal (pz, t ≈ 2,7 eV). Casos de prueba analíticos: bandas del
   grafeno, niveles de acenos.
2. Slater–Koster sp³ parametrizable: ssσ, spσ, ppσ, ppπ; energías on-site;
   escalado con la distancia (Harrison, GSP); cortes. Con solapamiento
   (Hc = ESc).
3. Parámetros: lector de `.skf` y ajuste a GPAW (fuerzas, niveles, bandas).
4. Carga autoconsistente (Mulliken, U tipo Hubbard).
5. Espín: Hubbard de campo medio (magnetismo de bordes zigzag).
6. Sistemas periódicos con puntos k (nanotubos periódicos válidos aquí).

Salidas: orbitales atómicos y proyecciones, orbitales moleculares (energías,
coeficientes, isosuperficies), cargas Mulliken/Löwdin, matriz de densidad
P = Σ fᵢ cᵢ cᵢ† (con S), órdenes de enlace, DOS/PDOS, bandas, y las tres
magnetizaciones de la decisión 4.

Hecho (`packages/tbkit`, 0.1.0):

- [x] 1. Modelo π con orbital π local (sirve en tubos y fullerenos), Hückel para
      N/B/O, hopping con deformación opcional. Grafeno, benceno, acenos y
      nanotubos zigzag contra fórmulas cerradas.
- [x] 2. Slater–Koster s/p con leyes constante, exponencial, Harrison, GSP y tabla;
      solapamiento (H c = E S c); carbono sp³ de Xu–Wang–Chan–Ho.
- [x] 3. Lector `.skf` (formato simple) y ajuste por mínimos cuadrados a niveles de
      referencia, con lector de `gpaw.txt`.
- [x] 4. Cargas autoconsistentes (DFTB2, Klopman–Ohno), solo sistemas finitos.
- [x] 5. Hubbard de campo medio con las tres magnetizaciones; teorema de Lieb
      comprobado en el trianguleno.
- [x] 6. Periódico con puntos k (suma de Bloch, mallas, caminos de ASE).
- [x] Salidas: P, Mulliken/Löwdin, Mayer/Wiberg/Coulson, DOS/PDOS, bandas, orbitales
      en rejilla → `.cube`. CLI `tbkit`.

Pendiente de la fase D:

- [x] Parte repulsiva: energía libre total, fuerzas de Hellmann–Feynman
      (ortogonal, no ortogonal, periódico, SCC), repulsión de Xu y spline de los
      `.skf`, calculadora ASE, relajación y fonones en Γ. Validado contra
      experimento sin reajustar (diamante, grafeno; ver README de tbkit).
- [ ] SCC periódico (suma de Ewald).
- [ ] Hubbard + SCC combinados; espín no colineal.
- [ ] Verificar la convención `sp` heteronuclear de los `.skf` contra DFTB+.
- [ ] Catálogo de parámetros de tbkit con la forma del de carbonforge (sin
      importarlo) y GUI (toolkit por decidir).

### Directriz científica del usuario (27-09-2026)

Adoptado de la "Directriz maestra" para tbkit: parámetros en JSON con unidad,
fuente, sistema y validez; colas suaves en los cortes; comprobación de H = H†,
S > 0 y C†SC = I; registro reproducible y `tbkit run`; informe de cada ajuste
(antes → después, métricas); nombres honestos ("tipo DFTB2" solo con integrales
de dos centros + SCC + repulsión); validación en tres niveles (unidad, física,
regresión) y protocolo requisito → formulación → implementación → test →
validación → documentación.

Anotado para más adelante (no forma parte de los objetivos actuales, salvo que
se decida): acoplamiento espín–órbita (validar primero en un átomo), MoS₂/WS₂ con
orbitales d, funciones de Green y NEGF (cadena 1D como primera validación),
transporte I(V), hamiltonianos asistidos por ML, matrices dispersas y
benchmarks. No adoptado: reorganizar tbkit en las carpetas de la directriz
(ya separa estructura, hamiltoniano, parámetros, solver y análisis).

## Objetivo (28-09-2026)

Calcular **IR y Raman de carbono dopado y funcionalizado** con los heteroátomos
más estudiados (N, O, B, S, P; grupos hidroxilo, epoxi, carbonilo, carboxilo,
amino). Descartado: FeSe y su interfaz, potenciales clásicos dentro de tbkit.
La auditoría del modelo está en `packages/tbkit/MODEL_AUDIT.md`; el orden de
trabajo, en su sección 9: validación con CNT prístinos → análisis de modos →
IR → oxígeno → B, S, P.

## Fase E — Raman con tight binding

| Paso | Método | Alcance |
|---|---|---|
| 1er orden no resonante | Polarizabilidad de enlace, o derivadas de la polarizabilidad TB | Asumible; fonones de TB (repulsivo tipo DFTB) o importados de QE |
| 1er orden resonante | Elementos de matriz ópticos y electrón–fonón en TB | Bastante más trabajo |
| 2º orden (banda 2D, doble resonancia) | Perturbaciones de 4º orden con suma en k en toda la zona de Brillouin | Nivel artículo; meses; solo periódico |


Hecho (paso 1, `packages/tbkit`):

- [x] Polarizabilidad TB: suma sobre estados (finitos y cristales, con el operador
      velocidad ∂ₖH en la convención de fases con posiciones), respuesta lineal
      SCC con apantallamiento para finitos (comprobada contra campo finito), ε∞.
- [x] Raman de primer orden no resonante sobre los fonones del propio modelo:
      tensores dα/dQ, actividades, despolarización, espectro con láser y Bose.
- [x] Validación por reglas de selección (diamante T₂g; C₆₀ 2A_g + 8H_g) y
      contra experimento (ε∞ del diamante, α de C₆₀), con el error declarado.

Pendiente de la fase E:

- [x] Parámetros con H y N (paso 2): convención `.skf` verificada contra
      DFTB+; bandera `scc` en el modelo; U de H, C, N calculadas con el átomo
      de GPAW (= DFTB mio); conjunto `xu_chn` (C–C de Xu intacto; H y N
      ajustados a niveles, fuerzas y energías de GPAW, ajuste conjunto con
      repulsión por proyección variable), con referencias y receta en el
      paquete. Validado: enlaces ≤ 0,02 Å (aromáticos, X–H), frecuencias
      50–70 cm⁻¹, reglas de selección del benceno, modos de anillo de la
      piridina. Pendiente: α baja (base mínima), C–C/C–N simples ~0,08 Å,
      O y B.
- [x] Fonones importados de QE (paso 3): lector propio en tbkit (`qe.py`:
      dynmat/matdyn en Γ, desplazamientos o autovectores, fase y subespacios
      degenerados), `raman(phonons=...)`, `raman_from_qe`, `tbkit raman --modes`.
- [x] Dipolos intraatómicos en la óptica (paso 4): ⟨s|r|p⟩ del átomo libre de
      GPAW, signo desde la convención del modelo, apantallamiento con dipolos
      atómicos derivado del γ de Klopman–Ohno; campo finito como prueba.
      Desaparecen α⊥ = 0 y ρ = 1/3; α de C₆₀ 61,7 → 69,0 Å³ (exp. 76,5 ± 8).
- [x] Más allá de la base mínima para la α (paso 5): polarizabilidad atómica
      extra por elemento, apantallada con el resto, ajustada a tensores α de
      GPAW FD (12 moléculas). Prueba ±2 %; C₆₀ 84,5 Å³ (exp. 76,5 ± 8); ε∞ del
      diamante 5,07 (exp. 5,7). Pendiente: campos locales en cristales (Ewald).
- [x] Raman resonante de primer orden (paso 6): ∂α(ω_L + iη)/∂Q con la α
      compleja apantallada (finitos) o interbanda (cristales), perfiles de
      excitación; `tbkit raman --resonant`. Sin estructura vibrónica.
- [x] Segundo orden (paso 7): G, 2D y 2D′ del grafeno por doble resonancia
      (`graphene.py`); dispersión de la 2D 112 cm⁻¹/eV (exp. ~100) con fonones
      GPAW; posiciones altas por los fonones PBE. Sin banda D (defectos).
- [x] Validación con CNT prístinos (paso 8): RBM de 11 tubos a −1,3/−4,0 % de
      227/d; G con desplazamiento y ΔG pequeño (sin anomalía de Kohn).
- [x] Análisis de modos (`modes.py`).
- [x] IR (paso 9): `infrared.py`; frente a GPAW, factor típico ~2 por modo.
- [x] Oxígeno: `xu_chno` (grupos funcionales, epóxido basal cerrado con la
      corrección de ángulos agudos y barridos de apertura; α ±10 % frente a GPAW).
- [x] B, S, P: `xu_chnob`, `xu_chnos`, `xu_chnop`, cada uno `xu_chno` fijo más un
      elemento (fuerzas 0,38 / 0,43 / 0,46 eV/Å). Falta su α extra.
- [ ] α extra de B, S, P (GPAW FD, `chn_polarizability --set b|s|p`).
- [ ] Validación en sistemas reales (copos y tubos dopados) antes de meter las
      frecuencias en el objetivo (orden acordado el 29-09-2026).
- [ ] Frecuencias en la función objetivo; U calculada en `xu_carbon`.

## Fase F — GUI de tbkit (decidido el 29-09-2026)

Decisiones 2 y 3 cerradas: **ventana propia** (`tbkit-gui`, dentro de tbkit;
carbonforge solo gana un botón «Abrir en tbkit» que la lanza como proceso
externo con la estructura actual) y **PySide6 + pyvista** (dependencias
opcionales, extra `gui`). Objetivo: todas las capacidades de tbkit desde la
ventana (modelos y parámetros, niveles/DOS/PDOS/bandas, orbitales e
isosuperficies, Hubbard y magnetización, relajación, fonones y modos, Raman
no resonante y resonante, IR, grafeno 2D, QE, registros reproducibles).

## Pendiente, por decisión del usuario (29-09-2026)

- H₂ fisisorbido en superestructuras (2000+ átomos): flujo GCMC en carbonforge
  (LAMMPS `fix gcmc`, LJ H₂–C, corrección de Feynman–Hibbs a 77 K). Aplazado.

Ideas tomadas de la literatura del usuario (Papaconstantopoulos, Mehl, Chronis,
Sigalas, "Tight-binding method in electronic structure", Encyclopedia of
Condensed Matter Physics, 2ª ed., 2024; el método NRL-TB), para después:

- [ ] Formas NRL como leyes de tbkit: hopping `(e + f r + g r²) exp(-t² r) F(r)`,
      solapamiento `(δ + p r + q r² + s r³) exp(-u² r) F(r)`, corte de Fermi
      `F(r) = 1/(1 + exp((r - R0)/l))`.
- [ ] On-site dependiente del entorno, `h = a + b ρ^{2/3} + c ρ^{4/3} + d ρ²` con
      `ρ_i = Σ_j exp(-λ² R_ij) F(R_ij)`: sin término repulsivo (el desplazamiento
      V0 de los autovalores DFT lo absorbe), y la transferibilidad entre
      coordinaciones que a Xu le falta. Sus fuerzas incluyen ∂h/∂R.
- [ ] Ajuste conjunto de energías totales y autovalores (su ec. 11, energías con
      peso ~200 veces el de una banda), ocupados y algunos vacíos.
- [ ] Lector de los parámetros NRL publicados para C (Papaconstantopoulos,
      Mehl, Erwin, Pederson 1998): su fonón Γ₂₅′ del diamante es 39.3 THz frente
      a 39.9 medido (1311 frente a 1332 cm⁻¹), mejor que los 1224 cm⁻¹ de Xu.
- [ ] Validar con propiedades fuera del ajuste, como hacen ellos: constantes
      elásticas, vacantes, superficies, fonones en toda la zona.
---

## Reglas que siguen valiendo

- Los paquetes no se importan entre sí (ver `CLAUDE.md` de la raíz).
- Lógica separada de los widgets: todo lo que decide va en módulos sin Tk y
  con tests; la ventana solo coloca controles y llama.
- Ningún cálculo largo dentro de los tests; los que usan un motor real van
  marcados `slow`.
- Advertir de parámetros físicos dudosos antes de ejecutar, no después.
