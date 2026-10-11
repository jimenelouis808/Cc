# Coil 5-7 de 204 átomos: todo lo calculado

Coil periódica con «rodillas» (12 pentágonos, 78 hexágonos y 12 heptágonos por celda, periódica
en z, Lz = 15 Å). Aquí está todo lo calculado sobre ella sin dopar, con N grafítico y con amina
(−NH₂ + H), copiado de `out/` sin recalcular nada. `MANIFIESTO.json` dice de qué archivo de
`out/` sale cada uno, con su SHA-256. Las carpetas de trabajo completas (trayectorias, cachés,
registros, 291 MB) no se versionan.

## Cómo abrirlo en tbkit

| Qué | Cómo |
|---|---|
| Una estructura | «Abrir…» → `estructuras/<sistema>/<modelo>.extxyz` |
| Los modos (animarlos, ordenarlos por sitio, Raman/IR con ellos) | Geometría y modos → «Abrir modos…» → `modos/<sistema>/hessian.npy`. Se diagonaliza la hessiana guardada (< 1 s) y su estructura queda activa |
| Raman resonante e IR de los tres | «Abrir resultado…» → `raman_ir_dopados/report.json` (o `tbkit abrir raman_ir_dopados`) |
| Validación con GPAW de los modos | «Abrir resultado…» → `validacion_gpaw_modos/report.json` |
| Estructura electrónica tbkit vs GPAW | «Abrir resultado…» → `electronica/report.json` |
| Bandas D y 2D (doble resonancia) | «Abrir resultado…» → `doble_resonancia/report.json` |
| Cualquier otro .json o espectro .npz | «Abrir resultado…» o «Abrir espectro guardado…» |

Las páginas y figuras ya hechas están en `paginas/`: HTML que se abre sin internet; figuras para
revista a una columna (85 mm), PDF y PNG a 600 dpi, en `*_figuras/`; y los datos de cada figura y
tabla en CSV, en `*_datos/`. Se regeneran con
`tbkit report <carpeta> --figuras pdf,svg,png --ancho doble`.

## Paso a paso

Comandos desde `packages/tbkit`. Cada receta se reanuda donde quedó; los números de abajo salen
de los archivos de esta carpeta.

### 0. La estructura de partida

1. **Construcción** (nanocarbon_lab, pasada como archivo):
   `build_knee_periodic_coil(coil_radius=7.75, pitch=15.0, sides_per_turn=6, circumference=6,
   relax=True)` → `estructuras/00_entrada_nanocarbon_lab.extxyz`.

### 1. Coil sin dopar (receta `nanocoil`, fase 2)

1. **Relajación TB** con dos modelos (`nanocoil tb --model xu_carbon` y `--model tang_carbon`).
   Malla 1×1×4, kT 0.05 eV, celda fija; se exige que la topología no cambie (enlaces C–C < 1.85 Å).
   - Topología intacta y sin modos imaginarios en los dos.
   - Los dos dan la coil sin gap en su propia geometría.
   - → `estructuras/sin_dopar/tb_xu_carbon.extxyz`, `tb_tang_carbon.extxyz`;
     `sin_dopar_fase2/tb_*.json`.
2. **Convergencia en k con GPAW** (`nanocoil gpaw-k`): PBE, LCAO dzp, h 0.2 Å, Fermi-Dirac 0.05 eV.
   - kz = 2 da fuerzas a 6×10⁻⁴ eV/Å RMS de kz = 4.
   - → `sin_dopar_fase2/gpaw_convergencia_k.json`.
3. **Relajación PBE** (`nanocoil gpaw-relax`), kz = 2, 6 Å de vacío.
   - → `estructuras/sin_dopar/pbe_gpaw.extxyz`.
4. **Fase 3** (`nanocoil gpaw-phase3`).
   - Gap PBE: 0.223 eV.
   - Enlaces de tbkit frente a PBE: RMS 0.016 Å (Xu) y 0.015 Å (Tang).
   - Frecuencias de modos tbkit proyectadas en PBE.
   - → `sin_dopar_fase2/gpaw_fase3_gap_geometria_modos.json`.
5. **Raman resonante de todos los modos** (`nanocoil raman`), Xu y Tang.
   - → `sin_dopar_fase2/raman_resonante_*`.
   - Fue el primer cálculo, con el modelo de solo carbono. El que se compara con los dopados es
     el del paso 6.
6. **Mismo modelo que los dopados** (`doped_raman`, ver §4): relajada otra vez con xu_chn + SCC.
   - → `estructuras/sin_dopar/tb_xu_chn.extxyz`, hessiana completa en `modos/sin_dopar/`.
7. **Doble resonancia, bandas D y 2D** (`coil_double_resonance`, §5).

### 2. N grafítico

1. **Cribado de sitios** (`site_screening`, modelo chn = xu_chn con SCC, kz 4).
   - Los 204 carbonos se agrupan en 51 clases equivalentes (por anillos y entorno).
   - Se calcula un punto simple por clase y se relajan las 5 mejores.
   - El mejor es el **sitio 0, en anillos [5, 5, 6]**: un N entre dos pentágonos. Los sitios 37 y
     66, también [5, 5, 6], quedan a +0.037 y +0.072 eV.
   - → `cribado_sitios/N_grafitico/` (`report.txt`, los 5 mejores relajados en `mejores/`).
2. **Comprobación con GPAW** (puntos simples PBE en las geometrías de tbkit de 4 sitios).
   - El sitio 0 también es el más bajo en PBE; los sitios 50, 3 y 8 quedan +0.40, +0.43 y +0.59 eV
     arriba.
   - → `cribado_sitios/N_grafitico/gpaw_puntos_simples/`.
3. **Relajación TB** del sitio 0 (`doped_raman relax N`).
   - → `estructuras/N_grafitico/tb_xu_chn.extxyz`.
4. **Hessiana** (`doped_raman hessian N`): se calculan las filas de los átomos a menos de 6 Å del
   N; el resto se toma de la coil sin dopar (embebido).
   - `doped_raman check N` comprobó el embebido con el modelo completo.
   - → `modos/N_grafitico/hessian.npy`.
5. **Raman resonante, IR y validación con GPAW**: §4.
6. **Electrónica**: con 817 electrones por celda la coil con N es metálica en los dos métodos (§6).

### 3. Amina (−NH₂ + H, adición 1,2)

1. **Candidatos**: 13 posiciones del grupo y su H, construidas con nanocarbon_lab.
2. **Cribado** (`structure_screening`, xu_chn, kz 4, relajación local de los átomos a menos de
   6 Å del grupo).
   - Mejor: **a048_h059**, el grupo en un C de anillos [6, 6, 7] y el H en el C vecino. El
     siguiente queda a +0.48 eV.
   - → `cribado_sitios/amina/`.
3. **Comprobación con GPAW** (puntos simples PBE de 3 candidatos).
   - a048_h059 también es el más bajo; a000_h200 y a048_h049 quedan +1.05 y +2.63 eV arriba.
   - → `cribado_sitios/amina/gpaw_puntos_simples/`.
4. **Relajación, hessiana embebida, Raman e IR**: igual que el N.
   - → `estructuras/amina/tb_xu_chn.extxyz`, `modos/amina/hessian.npy`.

### 4. Raman resonante e IR de los tres (receta `doped_raman`)

Ajustes: xu_chn con SCC, kz 4, kT 0.05 eV, láseres de 1.96, 2.33 y 2.54 eV, η 0.1 eV.
Diferencias finitas: δ 0.01 Å para la hessiana y 0.005 Å para α y las cargas de Born.

1. **Raman**: α(ω_L + iη) derivada a lo largo de cada modo por encima de la región baja (282
   modos sin dopar y con N, 290 con la amina).
   - La línea más fuerte a 2.33 eV es la G, a 1612–1620 cm⁻¹ en los tres.
2. **Banda D de primer orden**: intensidad de cada modo pesada por su carácter de respiración de
   anillos. Es una estimación; la banda D de verdad es de doble resonancia (§5).
3. **IR**: cargas de Born de los componentes transversales al eje (x, y), con la regla de suma
   impuesta.
   - Las del N se calcularon completas, porque embebidas se desviaban hasta 26 %.
   - Las de la amina se calcularon embebidas, con un error menor al 5 % comprobado contra el
     modelo completo.
4. **Incertidumbre** con η 0.05 y 0.2 eV y con el modelo Xu: `validation/coil204_raman_uncertainty.json`.
5. **Resultado**: `raman_ir_dopados/` y su página `paginas/raman_ir_dopados.html`
   (espectros a 1.96, 2.33 y 2.54 eV, banda D e IR).
6. **Validación con GPAW de los modos que importan** (receta `doped_gpaw`).
   - GPAW-PBE en x_PBE ± s·L a lo largo de cada modo de tbkit elegido: las 3 líneas Raman más
     fuertes en G, las 2 de la D, las 3 IR más fuertes, el modo de C más alto y los del dopante.
   - El N y la amina se relajaron antes en PBE (→ `estructuras/*/pbe_gpaw.extxyz`).
   - Cociente de frecuencias PBE/tbkit:

     | Sistema | Modos | Mediana | Rango |
     |---|---|---|---|
     | Sin dopar | 9 | 0.951 | 0.94–1.00 |
     | N grafítico | 12 | 0.950 | 0.95–1.00 |
     | Amina | 13 | 0.977 | 0.95–1.02 |

     Los modos de G salen ~5 % altos en tbkit: de ahí el factor de escala 0.95 que se usa en las
     tablas escaladas.
   - → `validacion_gpaw_modos/`.

### 5. Bandas D y 2D por doble resonancia (receta `coil_double_resonance`; solo sin dopar)

1. **Constantes de fuerza** Φ(R) en una supercelda de 3 celdas (612 átomos, xu_carbon, kz 2,
   ±0.01 Å).
   - D(q = 0) reproduce los 612 modos Γ a 0.013 cm⁻¹.
2. **Acoplamiento electrón–fonón** ⟨k+q|∂H|k⟩ en 24 puntos k y q, con γ = 0.1 eV.
   - La 2D suma todos los pares de ramas.
   - El defecto de la D es un sitio con la energía de sitio de un N.
3. **Resultado**:

   | Láser | D (cm⁻¹) | 2D (cm⁻¹) | I_D/I_G | I_2D/I_G |
   |---|---|---|---|---|
   | 1.96 eV | 1294 | 2654 | 0.68 | 0.55 |
   | 2.33 eV | 1295 | 2656 | 0.53 | 0.31 |
   | 2.54 eV | 1301 | 2660 | 0.75 | 0.42 |

   - La dispersión es de unos 10 cm⁻¹/eV, frente a ~50 y ~100 en grafeno.
   - Las intensidades dependen de γ.
   - → `doble_resonancia/`.

### 6. Estructura electrónica, tbkit frente a GPAW (receta `electronic_compare`)

Bandas y DOS de los tres sistemas en la geometría PBE, más grafeno y moléculas de referencia.

| Sistema | Gap PBE | Gap tbkit |
|---|---|---|
| Sin dopar | 0.223 eV | 0.186 eV |
| Amina | 0.228 eV | 0.184 eV |
| N grafítico | metálico | metálico |

- **Banda del N** (semillena en el nivel de Fermi): 23 meV de ancho en GPAW y 83 meV en tbkit.
- **Forma de la DOS**: correlación 0.80–0.86 entre los dos métodos en ±2 eV del nivel de Fermi.
  Lejos de él no coincide: las bandas σ salen ~4 eV demasiado hondas en tbkit.
- → `electronica/` y `validation/electronic_tb_vs_gpaw.json`.

### 7. N piridínico y pirrólico (solo cribado, no usado después)

1. **Cribado tbkit**: puntos simples de todos los motivos y relajación local de los 4–5 mejores.
   - → `cribado_sitios/piridinico/`, `pirrolico/`.
2. **Puntos simples GPAW** de 3 candidatos de cada uno.
   - En las geometrías de tbkit, las fuerzas PBE son de 2.6 a 7.9 eV/Å: esas geometrías están
     lejos del mínimo PBE.
   - Por eso el orden de estos sitios **no es confiable** y no se siguió con ellos.
3. **Construcción del pirrólico**: la primera versión tenía los tres H encimados (error del
   constructor) y se rehízo.

## Figuras con incertidumbre (`incertidumbre/`)

Una segunda versión de las figuras de Raman, con barras y bandas de incertidumbre. Las
primeras versiones siguen sin tocar en `paginas/`. No se recalculó nada: todo sale de lo que
ya estaba en caché.

```bash
cd packages/tbkit
uv run python resultados/coil204/incertidumbre/hacer_figuras.py datos     # necesita out/
uv run python resultados/coil204/incertidumbre/hacer_figuras.py figuras   # solo datos/
```

| Figura (`incertidumbre/figuras/`, PDF y PNG 600 dpi) | Qué incertidumbre muestra |
|---|---|
| `raman_banda_eta_1_96`, `_2_33`, `_2_54` | Espectro con η = 0.1 eV (línea) y la banda η 0.05–0.2 eV. Tres estructuras; cada curva va normalizada a su máximo de G. |
| `centroide_G_eta`, `centroide_D_eta`, `id_ig_eta` | Valor con η = 0.1 eV y barra sobre η 0.05–0.2 eV, por láser y estructura. |
| `modelos_sin_dopar` | Coil sin dopar con los tres modelos (xu_chn, Xu carbono, Tang): la dispersión entre modelos. |
| `cociente_pbe_tbkit` | Cociente de frecuencias PBE/tbkit de los modos verificados con GPAW, con la línea de la mediana. Su dispersión es la incertidumbre de cualquier posición escalada. |

- **Base de los espectros con η**: solo los 40 modos más fuertes de cada estructura, que son los
  que tienen tensores con η 0.05 y 0.2.
- **Datos**: CSV y JSON en `incertidumbre/datos/`.
- **Qué muestran**:
  - η mueve sobre todo las alturas relativas. La posición de G cambia pocos cm⁻¹, salvo la coil
    sin dopar a 1.96 eV: ~1567–1590 cm⁻¹.
  - I_D/I_G del N grafítico queda en ~0.01 con cualquier η.
  - El modelo pesa más que η en I_D/I_G sin dopar: Tang da 0.10–0.13; xu_chn, 0.08–0.10.
  - PBE/tbkit: mediana 0.951, intervalo 0.941–1.023 (34 modos). Una posición escalada con 0.95
    arrastra una incertidumbre de −1 % / +8 %.

## Lo que no está aquí

- **Cálculos de GPAW**: archivos `.gpw`, trayectorias y cachés de desplazamientos. Pesan cientos
  de MB y solo sirven para reanudar.
- **Validaciones generales de los modelos** (no son de la coil): están en `validation/` del
  paquete y en `docs/VALIDACION.md`.
