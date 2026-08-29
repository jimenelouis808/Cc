# Estrategias de búsqueda — Web of Science Core Collection

> Pegar en **Advanced Search** (pestaña *Documents*). Registrar colección,
> índices activos y fecha. Ver `docs/PROTOCOL.md`.

---

## 0. Diferencias de sintaxis frente a Scopus (importante)

| Aspecto | Scopus | Web of Science |
|---|---|---|
| Campo tema | `TITLE-ABS-KEY(...)` | `TS=(...)` → título + resumen + *Author Keywords* + **KeyWords Plus** |
| Solo título | `TITLE(...)` | `TI=(...)` |
| Comodines | `*` (0+), `?` (1) | `*` (0+), `?` (exactamente 1), `$` (0 o 1) |
| Proximidad | `W/n`, `PRE/n` | `NEAR/n` (por defecto `NEAR` = 15), `SAME` |
| Año | `PUBYEAR > 1990` | `PY=(1991-2025)` |
| Tipo doc | `DOCTYPE(ar)` | `DT=(Article)` |
| Idioma | `LANGUAGE(english)` | `LA=(English)` |

**Dos avisos que cambian los resultados:**

1. **KeyWords Plus.** `TS=` incluye KeyWords Plus, términos que WoS genera
   automáticamente a partir de los títulos de las referencias citadas. Esto
   **aumenta el recall y mete ruido** respecto a Scopus. Si quieres máxima
   comparabilidad Scopus↔WoS, usa `TI=(...) OR AB=(...) OR AK=(...)` en lugar de
   `TS=`. Recomiendo correr **ambas** y reportar la diferencia: es un resultado
   metodológico interesante por sí mismo.
2. **Lematización.** WoS lematiza por defecto (`nanotube` ↔ `nanotubes`). Puedes
   desactivarla con comillas exactas. Al usar `*` explícitos el efecto es menor,
   pero decláralo en el protocolo.

---

## 1. CORE — consulta principal (equivalente al brazo de precisión de Scopus)

```
TS=("carbon nanotube*" OR "carbon nano-tube*" OR SWCNT* OR MWCNT* OR DWCNT*
    OR SWNT* OR MWNT* OR "single-wall* carbon nanotube*"
    OR "double-wall* carbon nanotube*" OR "multi-wall* carbon nanotube*"
    OR "carbon nanofib*" OR "carbon nano-fib*" OR "carbon nanofilament*"
    OR "vapor grown carbon fib*" OR "vapour grown carbon fib*"
    OR "carbon nanocoil*" OR "carbon nanotube yarn*" OR "carbon nanotube fib*")
AND
TS=(doped OR doping OR dopant* OR codoped OR codoping OR "co-doped" OR "co-doping"
    OR "dual-doped" OR "tri-doped" OR heteroatom* OR "hetero-atom*"
    OR defect* OR vacanc* OR "Stone-Wales" OR "Stone Wales"
    OR "topological defect*" OR "point defect*" OR "line defect*"
    OR "grain boundar*" OR "dangling bond*" OR "sp3 defect*"
    OR substitutional* OR adatom* OR interstitial*
    OR "defect engineering" OR "defect-free" OR "pentagon-heptagon")
NOT
TI=("boron nitride nanotube*" OR BNNT* OR "cellulose nanofib*"
    OR "nanofibrillated cellulose" OR "bacterial cellulose")
```

Después, en el panel de refinado:
```
AND PY=(1991-2025)
AND DT=(Article OR Review)          [decide si añades "Proceedings Paper"]
AND LA=(English)
```

> **`NOT` en WoS** se aplica al conjunto completo, no al bloque anterior.
> Si tu interfaz protesta, guarda cada bloque como set (`#1`, `#2`, `#3`) y combina:
> `#1 AND #2 NOT #3`. Es la forma más limpia y la que deja trazabilidad en el
> *Search History* — expórtalo, lo necesitas para PRISMA-S.

---

## 2. Variante comparable a Scopus (sin KeyWords Plus)

```
(TI=(<bloque A>) OR AB=(<bloque A>) OR AK=(<bloque A>))
AND
(TI=(<bloque B>) OR AB=(<bloque B>) OR AK=(<bloque B>))
```
donde `<bloque A>` y `<bloque B>` son los del §1. Reporta N con `TS=` y N con
esta variante: la brecha te dice cuánto aporta KeyWords Plus en tu tema.

---

## 3. CORE — brazo de alta sensibilidad

```
TS=("carbon nanotube*" OR "carbon nano-tube*" OR SWCNT* OR MWCNT* OR DWCNT*
    OR SWNT* OR MWNT* OR "carbon nanofib*" OR "carbon nano-fib*"
    OR "carbon nanofilament*" OR "carbon nanocoil*" OR "carbon nanohorn*"
    OR "bamboo-like carbon*" OR "herringbone carbon*")
AND
TS=(doped OR doping OR dopant* OR codop* OR heteroatom* OR defect* OR vacanc*
    OR disorder* OR "Stone-Wales" OR substitutional* OR functionaliz*
    OR "nitrogen incorporation" OR irradiat* OR "ion implantation")
NOT
TI=("boron nitride nanotube*" OR BNNT* OR "cellulose nanofib*" OR dopamin*)
```

---

## 4. FACETAS (añadir con `AND` sobre el CORE)

Traducción directa de `queries/scopus.md` §3, cambiando `TITLE-ABS-KEY(` por `TS=(`.

### 4.1 Teórico / computacional
```
AND TS=("density functional theor*" OR DFT OR "first-principle*" OR "first principles"
    OR "ab initio" OR "ab-initio" OR "molecular dynamic*" OR "tight-binding"
    OR "tight binding" OR "Monte Carlo" OR "reactive force field" OR ReaxFF OR AIREBO
    OR "machine learning potential*" OR "neural network potential*"
    OR "nonequilibrium Green" OR "non-equilibrium Green" OR NEGF
    OR "Boltzmann transport" OR "many-body perturbation" OR "Bethe-Salpeter"
    OR "band structure calculation*" OR "formation energy")
```

### 4.2 Experimental
```
AND TS=("chemical vapor deposition" OR "chemical vapour deposition" OR CVD
    OR "arc discharge" OR "laser ablation" OR "floating catalyst"
    OR "Raman spectroscop*" OR "Raman spectra" OR "D band" OR "G band"
    OR XPS OR "X-ray photoelectron" OR "transmission electron microscop*"
    OR HRTEM OR HAADF OR STEM OR "electron energy loss" OR EELS
    OR "scanning tunneling microscop*" OR XANES OR NEXAFS
    OR "electron paramagnetic" OR "electron spin resonance"
    OR "ion irradiation" OR "electron irradiation" OR "plasma treatment")
```

### 4.3 Dopantes por proximidad
```
AND TS=((nitrogen OR boron OR phosphorus OR phosphorous OR sulfur OR sulphur
    OR fluorine OR chlorine OR iodine OR silicon OR oxygen OR selenium)
    NEAR/3 (doped OR doping OR dopant* OR codoped)
    OR "pyridinic" OR "pyrrolic" OR "graphitic nitrogen" OR "quaternary nitrogen")
```

### 4.4 Estructuras 3D
```
AND TS=("carbon nanotube sponge*" OR "CNT sponge*" OR aerogel*
    OR "carbon nanotube forest*" OR "vertically aligned carbon nanotube*"
    OR VACNT* OR buckypaper OR "bucky paper" OR "carbon nanotube array*"
    OR "carbon nanotube network*" OR "three-dimensional network*"
    OR "3D network*" OR foam* OR "nanotube junction*" OR "Y-junction*"
    OR "carbon nanotube yarn*" OR "macroscopic assembl*" OR "hierarchical carbon*")
```

### 4.5 Aplicaciones
```
AND TS=(electrocataly* OR "oxygen reduction" OR ORR OR "hydrogen evolution"
    OR "oxygen evolution" OR "CO2 reduction" OR supercapacitor*
    OR "electrochemical capacitor*" OR "lithium-ion" OR "sodium-ion"
    OR "lithium-sulfur" OR "metal-air" OR batter* OR "fuel cell*"
    OR sensor* OR biosensor* OR "gas sensing" OR "field emission"
    OR thermoelectric* OR "thermal conductivity" OR "field-effect transistor*"
    OR photocataly* OR "hydrogen storage" OR "CO2 capture" OR adsorption
    OR membrane* OR "polymer composite*" OR "EMI shielding" OR "microwave absorption")
```

---

## 5. Corpus de contexto — grafeno / GO (separado)

```
TS=(graphene OR "graphene oxide" OR "reduced graphene oxide" OR rGO
    OR "graphene nanoribbon*" OR "few-layer graphene" OR "graphene quantum dot*")
AND
TS=(doped OR doping OR dopant* OR codoped OR heteroatom* OR defect* OR vacanc*
    OR "Stone-Wales" OR "grain boundar*")
AND PY=(2004-2025) AND DT=(Article OR Review) AND LA=(English)
```

---

## 6. Exportación desde WoS — checklist crítico

1. *Export* → **Plain text file** (`.txt`, formato ISI/tagged) o **Tab-delimited**.
   `bibliometrix::convert2df` acepta ambos; **el plain text tagged es el más fiable**.
2. Record Content: **`Full Record and Cited References`** ← **imprescindible**.
   Sin las referencias citadas no hay co-citación ni RPYS. No se recupera después.
3. Límite: **1 000 registros por exportación**. Trocea por rangos (`1-1000`,
   `1001-2000`, …). El pipeline de Python concatena y deduplica automáticamente.
4. Guarda en `data/raw/wos/` como `wos_<arm>_<inicio>-<fin>_<YYYYMMDD>.txt`.
5. Exporta también el **Search History** completo (botón *History* → export)
   para el reporte PRISMA-S.

### Índices de la Core Collection
Declara explícitamente cuáles tienes activos: `SCI-EXPANDED`, `SSCI`, `A&HCI`,
`CPCI-S`, `ESCI`, `BKCI`. La suscripción de tu institución cambia el N — dos
personas con la "misma" consulta obtienen números distintos si sus índices difieren.
**Esto se reporta en Métodos, siempre.**
