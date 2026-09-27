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

## Fase E — Raman con tight binding

| Paso | Método | Alcance |
|---|---|---|
| 1er orden no resonante | Polarizabilidad de enlace, o derivadas de la polarizabilidad TB | Asumible; fonones de TB (repulsivo tipo DFTB) o importados de QE |
| 1er orden resonante | Elementos de matriz ópticos y electrón–fonón en TB | Bastante más trabajo |
| 2º orden (banda 2D, doble resonancia) | Perturbaciones de 4º orden con suma en k en toda la zona de Brillouin | Nivel artículo; meses; solo periódico |

---

## Reglas que siguen valiendo

- Los paquetes no se importan entre sí (ver `CLAUDE.md` de la raíz).
- Lógica separada de los widgets: todo lo que decide va en módulos sin Tk y
  con tests; la ventana solo coloca controles y llama.
- Ningún cálculo largo dentro de los tests; los que usan un motor real van
  marcados `slow`.
- Advertir de parámetros físicos dudosos antes de ejecutar, no después.
