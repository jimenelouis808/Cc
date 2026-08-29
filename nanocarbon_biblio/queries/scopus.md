# Estrategias de búsqueda — Scopus (Advanced Search)

> Pegar en **Advanced document search**. Registrar SIEMPRE: fecha de ejecución,
> nº de resultados por bloque, y la versión final usada. Ver `docs/PROTOCOL.md`.

---

## 0. Decisiones de sintaxis (por qué está escrito así)

| Decisión | Motivo |
|---|---|
| **No usar `dop*`** | Recupera `dopamine`/`dopaminergic`. Los sensores electroquímicos de dopamina con CNT son miles de registros irrelevantes. Se enumeran las formas: `doped OR doping OR dopant*`. |
| **No usar `CNF` suelto** | `CNF` = *cellulose nanofibril/nanofiber* en ciencia de materiales. Contaminación masiva. Se usa `"carbon nanofib*"`. |
| **`CNT` suelto solo en el brazo de alta sensibilidad** | `CNT` = *classical nucleation theory*. En la práctica casi todo artículo escribe "carbon nanotube" al menos una vez en TI/AB/KW. |
| **Comillas = *loose phrase*** | Scopus ignora puntuación dentro de `"..."` pero **no** une tokens: `"carbon nanotube"` no captura `carbon nano tube`. Por eso se añaden variantes con guion. |
| **`*` = 0 o más caracteres** | `"carbon nanofib*"` cubre fiber/fibre/fibers/fibrous. |
| **`W/n` = proximidad sin orden**, `PRE/n` = orden fijo | Usado en el bloque de dopantes para capturar `nitrogen-doped`, `doped with nitrogen`, `nitrogen and boron co-doping`. |
| **Exclusiones solo en `TITLE`** | Excluir en TITLE-ABS-KEY tiraría composites CNT/celulosa o comparativas CNT vs BNNT que sí son relevantes. |

---

## 1. CORE — consulta principal (brazo de precisión)

Este es el corpus del review: **nanoestructuras 1D de carbono × defectos/dopaje**.

```
( TITLE-ABS-KEY ( "carbon nanotube*"  OR  "carbon nano-tube*"  OR  swcnt*  OR  mwcnt*
      OR  dwcnt*  OR  swnt*  OR  mwnt*  OR  "single-wall* carbon nanotube*"
      OR  "double-wall* carbon nanotube*"  OR  "multi-wall* carbon nanotube*"
      OR  "carbon nanofib*"  OR  "carbon nano-fib*"  OR  "carbon nanofilament*"
      OR  "vapor grown carbon fib*"  OR  "vapour grown carbon fib*"
      OR  "carbon nanocoil*"  OR  "carbon nanotube yarn*"  OR  "carbon nanotube fib*" ) )
AND
( TITLE-ABS-KEY ( doped  OR  doping  OR  dopant*  OR  codoped  OR  codoping
      OR  "co-doped"  OR  "co-doping"  OR  "dual-doped"  OR  "tri-doped"
      OR  heteroatom*  OR  "hetero-atom*"  OR  "heteroatom-doped"
      OR  defect*  OR  vacanc*  OR  "stone-wales"  OR  "stone wales"
      OR  "topological defect*"  OR  "point defect*"  OR  "line defect*"
      OR  "grain boundar*"  OR  "dangling bond*"  OR  "sp3 defect*"
      OR  substitutional*  OR  adatom*  OR  interstitial*
      OR  "defect engineering"  OR  "defect-free"  OR  "pentagon-heptagon" ) )
AND  PUBYEAR  >  1990
AND  ( DOCTYPE ( ar )  OR  DOCTYPE ( re )  OR  DOCTYPE ( cp ) )
AND  LANGUAGE ( english )
AND NOT  TITLE ( "boron nitride nanotube*"  OR  bnnt*  OR  "cellulose nanofib*"
      OR  "nanofibrillated cellulose"  OR  "bacterial cellulose" )
```

> **Nota sobre `DOCTYPE`/`LANGUAGE`:** funcionan dentro de la caja de *Advanced search*.
> Si prefieres filtrar en la interfaz, quita esas dos líneas y usa los checkboxes
> `LIMIT-TO(DOCTYPE,"ar")` etc. — el resultado es idéntico pero la cadena queda
> más limpia para reportar en PRISMA-S.
>
> **Sobre `cp` (conference paper):** decide con datos. En nanocarbono hay mucho
> congreso de baja señal y sin referencias citadas completas. Corre la consulta
> con y sin `cp` y reporta ambos N. Mi recomendación: **excluir `cp`** del corpus
> analítico principal y usarlo solo para la curva de crecimiento.

---

## 2. CORE — brazo de alta sensibilidad (para test de recall)

Solo para medir cuánto pierde el brazo de precisión. **No es el corpus final.**

```
( TITLE-ABS-KEY ( "carbon nanotube*" OR "carbon nano-tube*" OR swcnt* OR mwcnt* OR dwcnt*
      OR swnt* OR mwnt* OR "carbon nanofib*" OR "carbon nano-fib*" OR cnt OR cnts
      OR "carbon nanofilament*" OR "carbon nanocoil*" OR "carbon nanohorn*"
      OR "bamboo-like carbon*" OR "herringbone carbon*" OR "platelet carbon nanofib*" ) )
AND
( TITLE-ABS-KEY ( doped OR doping OR dopant* OR codop* OR "co-doped" OR "co-doping"
      OR heteroatom* OR defect* OR vacanc* OR disorder* OR "stone-wales"
      OR substitutional* OR functionaliz* OR "nitrogen incorporation"
      OR "boron incorporation" OR irradiat* OR "ion implantation" ) )
AND  PUBYEAR  >  1990
AND NOT  TITLE ( "boron nitride nanotube*" OR bnnt* OR "cellulose nanofib*"
      OR "nanofibrillated cellulose" OR dopamin* )
```

Compara: `N_sensible − N_precisión` y revisa manualmente una muestra aleatoria
de 100 registros del delta. Si >70 % son irrelevantes, quédate con el brazo de precisión.

---

## 3. FACETAS (se añaden con `AND` sobre el CORE)

No cambian el corpus: **etiquetan subconjuntos**. Corre cada una y guarda el N
para tu tabla de composición del corpus. En el pipeline de Python estas mismas
facetas se aplican por léxico sobre título+resumen+keywords (`classify.py`),
lo que te permite tener las etiquetas *dentro* del data frame de bibliometrix.

### 3.1 Facet TEÓRICO / COMPUTACIONAL

```
AND TITLE-ABS-KEY ( "density functional theor*"  OR  dft  OR  "first-principle*"
      OR  "first principles"  OR  "ab initio"  OR  "ab-initio"
      OR  "molecular dynamic*"  OR  "tight-binding"  OR  "tight binding"
      OR  "monte carlo"  OR  "reactive force field"  OR  reaxff  OR  airebo
      OR  "machine learning potential*"  OR  "machine-learned potential*"
      OR  "neural network potential*"  OR  "non-equilibrium green"
      OR  "nonequilibrium green"  OR  negf  OR  "boltzmann transport"
      OR  "many-body perturbation"  OR  "bethe-salpeter"  OR  "GW approximation"
      OR  "band structure calculation*"  OR  "formation energy" )
```
> ⚠ `dft` también es *discrete Fourier transform*. Dentro del CORE el ruido es
> bajo, pero valida una muestra.

### 3.2 Facet EXPERIMENTAL (síntesis + caracterización)

```
AND TITLE-ABS-KEY ( "chemical vapor deposition"  OR  "chemical vapour deposition"  OR  cvd
      OR  "arc discharge"  OR  "laser ablation"  OR  "floating catalyst"
      OR  "raman spectroscop*"  OR  "raman spectra"  OR  "D band"  OR  "G band"
      OR  "ID/IG"  OR  xps  OR  "x-ray photoelectron"  OR  "N 1s"
      OR  "transmission electron microscop*"  OR  hrtem  OR  haadf  OR  stem
      OR  "electron energy loss"  OR  eels  OR  "scanning tunneling microscop*"
      OR  xanes  OR  nexafs  OR  "near edge x-ray"  OR  "electron paramagnetic"
      OR  "electron spin resonance"  OR  "thermogravimetric"
      OR  "ion irradiation"  OR  "electron irradiation"  OR  "post-annealing"
      OR  "plasma treatment"  OR  "ammonia treatment" )
```

### 3.3 Facet DOPANTE ESPECÍFICO (proximidad)

```
AND TITLE-ABS-KEY ( ( nitrogen  OR  boron  OR  phosphorus  OR  phosphorous
        OR  sulfur  OR  sulphur  OR  fluorine  OR  chlorine  OR  bromine  OR  iodine
        OR  silicon  OR  oxygen  OR  selenium  OR  "transition metal*" )
      W/3  ( doped  OR  doping  OR  dopant*  OR  codoped  OR  "co-doped" )
   OR  "n-doped"  OR  "b-doped"  OR  "p-doped"  OR  "s-doped"  OR  "f-doped"
   OR  "si-doped"  OR  "o-doped"  OR  "se-doped"  OR  "cl-doped"
   OR  "n,s-codoped"  OR  "n,b-codoped"  OR  "n,p-codoped"  OR  "b,n-codoped"
   OR  "pyridinic"  OR  "pyrrolic"  OR  "graphitic nitrogen"  OR  "quaternary nitrogen" )
```
> ⚠ `"p-doped"` colisiona con **p-type doping** (semiconductores) y `"n-doped"`
> con **n-type**. En CNT esto es una ambigüedad real y molesta. En el pipeline de
> Python (`classify.py`) se desambigua exigiendo co-ocurrencia con
> `phosphorus/phosphorous` o `nitrogen` en el mismo registro. **Recomiendo confiar
> en la clasificación de Python, no en esta faceta de Scopus**, precisamente por esto.

### 3.4 Facet ESTRUCTURAS 3D / ENSAMBLAJES MACROSCÓPICOS

```
AND TITLE-ABS-KEY ( "carbon nanotube sponge*"  OR  "CNT sponge*"  OR  aerogel*
      OR  "carbon nanotube forest*"  OR  "vertically aligned carbon nanotube*"
      OR  vacnt*  OR  buckypaper  OR  "bucky paper"  OR  "carbon nanotube array*"
      OR  "carbon nanotube network*"  OR  "three-dimensional network*"
      OR  "3D network*"  OR  "3D architecture*"  OR  foam*  OR  "nanotube junction*"
      OR  "y-junction*"  OR  "carbon nanotube yarn*"  OR  "carbon nanotube fib*"
      OR  "macroscopic assembl*"  OR  "hierarchical carbon*"  OR  "self-assembled network*" )
```

### 3.5 Facet APLICACIONES

```
AND TITLE-ABS-KEY ( electrocataly*  OR  "oxygen reduction"  OR  orr  OR  "hydrogen evolution"
      OR  her  OR  "oxygen evolution"  OR  oer  OR  "CO2 reduction"  OR  "nitrogen reduction"
      OR  supercapacitor*  OR  "electrochemical capacitor*"  OR  "lithium-ion"
      OR  "sodium-ion"  OR  "potassium-ion"  OR  "lithium-sulfur"  OR  "metal-air"
      OR  batter*  OR  "fuel cell*"  OR  sensor*  OR  biosensor*  OR  "gas sensing"
      OR  "field emission"  OR  thermoelectric*  OR  "thermal conductivity"
      OR  "field-effect transistor*"  OR  "field effect transistor*"
      OR  photocataly*  OR  "hydrogen storage"  OR  "CO2 capture"  OR  adsorption
      OR  membrane*  OR  "water treatment"  OR  "capacitive deionization"
      OR  "polymer composite*"  OR  "mechanical reinforcement"  OR  "EMI shielding"
      OR  "electromagnetic interference"  OR  "microwave absorption" )
```
> ⚠ `her` es una palabra inglesa común. En Scopus, dentro de TITLE-ABS-KEY sin
> comillas es un token exacto, no un truncamiento, pero aun así aparece en frases
> ("her group"). Es raro en abstracts técnicos; aun así, prefiero clasificar
> aplicaciones en Python con reglas contextuales.

---

## 4. CORPUS DE CONTEXTO — grafeno / GO (corpus SEPARADO, no mezclar)

Ver `docs/PROTOCOL.md` §2 para la justificación. Este corpus **no entra** en el
análisis principal; se usa para curvas comparativas de crecimiento y para el
análisis de migración temática 1D→2D.

```
( TITLE-ABS-KEY ( graphene  OR  "graphene oxide"  OR  "reduced graphene oxide"  OR  rgo
      OR  "graphene nanoribbon*"  OR  "few-layer graphene"  OR  "graphene quantum dot*" ) )
AND
( TITLE-ABS-KEY ( doped OR doping OR dopant* OR codoped OR codoping OR "co-doped"
      OR heteroatom* OR defect* OR vacanc* OR "stone-wales" OR "grain boundar*" ) )
AND  PUBYEAR  >  2003
AND  ( DOCTYPE ( ar )  OR  DOCTYPE ( re ) )
AND  LANGUAGE ( english )
```

### 4.1 Corpus de HÍBRIDOS (este SÍ suele entrar en el core)

Los híbridos CNT–grafeno ya caen dentro del CORE porque contienen términos de CNT.
Para cuantificarlos explícitamente:

```
<CORE>  AND  TITLE-ABS-KEY ( graphene  OR  "graphene oxide"  OR  rgo )
```

---

## 5. Exportación desde Scopus — checklist crítico

1. Selecciona todos → **Export → CSV**.
2. En el panel de campos marca **obligatoriamente**:
   - *Citation information* (incluye **Year, Cited by, DOI, Source title, Document Type**)
   - *Bibliographical information* (incluye **Affiliations**, necesario para `C1`/países)
   - *Abstract & keywords* (necesario para `AB`, `DE`, `ID`)
   - ✅ **`References`** ← **imprescindible**. Sin este campo NO hay co-citación,
     ni acoplamiento bibliográfico, ni RPYS. **No se puede añadir a posteriori.**
   - *Funding details* (si vas a hacer análisis de financiación)
3. Límite: **20 000 registros por exportación CSV**. Si superas eso, trocea por
   `PUBYEAR` (p. ej. 1991-2010, 2011-2016, 2017-2020, 2021-2025) y concatena.
   El pipeline de Python acepta varios ficheros y deduplica.
4. Guarda cada fichero en `data/raw/scopus/` con nombre
   `scopus_<arm>_<rango-años>_<YYYYMMDD>.csv`.
