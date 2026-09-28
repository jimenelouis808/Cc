# Auditoría del modelo — tbkit

Qué calcula tbkit, con qué ecuaciones, qué parámetros usa y de dónde vienen,
qué está comprobado y qué no. Escrito a partir del código (no de la
documentación previa); cada afirmación señala el módulo donde se puede
verificar. Unidades: eV, Å, amu, cm⁻¹ salvo que se diga otra cosa.

Objetivo del proyecto al que sirve: **IR y Raman de carbono dopado y
funcionalizado** con los heteroátomos más estudiados.

---

## 1. Flujo de cálculo

```mermaid
flowchart TD
    S[Estructura ASE<br/>archivo .xyz/.extxyz] --> SYS[System.build<br/>hamiltonian.py]
    P[Parámetros JSON<br/>params.py / skf.py] --> SYS
    SYS --> H["H(k), S(k)<br/>bloques Slater–Koster"]
    H --> SOL{¿modelo SCC?}
    SOL -- no --> DIAG[diagonalizar<br/>solver.solve]
    SOL -- sí --> SCC[cargas autoconsistentes<br/>scc.self_consistent]
    DIAG --> EF[energía libre y fuerzas<br/>forces.py]
    SCC --> EF
    EF --> CALC[TBCalculator<br/>calculator.py]
    CALC --> RELAX[relajación BFGS<br/>tasks.relax]
    CALC --> VIB[fonones Γ por diferencias finitas<br/>ase.vibrations → raman._model_phonons]
    QE[modos de QE<br/>qe.py] -.-> VIB
    VIB --> RAMAN[∂α/∂Q<br/>raman.py / resonance.py]
    SOL --> ALPHA[α estática o α(ω+iη)<br/>optics.py + dipoles.py]
    ALPHA --> RAMAN
    RAMAN --> SPEC[actividades, ρ, espectro]
    GR[grafeno: modelo π + fonones en toda la ZB<br/>graphene.py] --> SPEC2[G, 2D, 2D′]
```

---

## 2. Hamiltoniano

**Base.** Por elemento, orbitales `s, px, py, pz` (sp³) o un orbital π local
`"pi"` (modelo π). No hay orbitales d (`basis.py`, `params.py`).

**Elementos de matriz** (`slater_koster.py`): para un enlace A→B con cosenos
directores c:

- ⟨s_A|s_B⟩ = V_ssσ
- ⟨s_A|p_B,α⟩ = c_α V_spσ(s en A, p en B)
- ⟨p_A,α|s_B⟩ = −c_α V_spσ(s en B, p en A)
- ⟨p_A,α|p_B,β⟩ = c_α c_β V_ppσ + (δ_αβ − c_α c_β) V_ppπ
- ⟨π_A|π_B⟩ = V_ppπ(d), sin factor angular (orbital π normal a la superficie).

**H(k)** (`hamiltonian.py`): H_ij(k) = Σ_R h_ij(R) e^{2πi k·R_frac}; en la
óptica periódica, con fases que incluyen posiciones (`optics._bloch_ii`).
Solapamiento opcional (H c = E S c); se comprueba H = H† y S > 0 en cada
construcción.

**Leyes de distancia** (`params.py`): constante, exponencial
v0 e^{−β(d/d0−1)}, Harrison v0 (d0/d)ⁿ, GSP
v0 (r0/d)ⁿ exp{n[−(d/rc)^nc + (r0/rc)^nc]}, tabla (spline cúbico, `.skf`),
polinomio de corte Σ c_k (rc−d)^k (k ≥ 3), y `Tail`: la ley interior hasta r1
y un cúbico hasta rm que iguala valor y pendiente en r1 y se anula con
pendiente cero en rm.

## 3. Modelo de Xu (`parameters/xu_carbon.json`)

Referencia: C. H. Xu, C. Z. Wang, C. T. Chan, K. M. Ho, J. Phys.: Condens.
Matter 4, 6047 (1992).

| Término | Forma en tbkit | Valores |
|---|---|---|
| On-site | E_s, E_p | −2.99, 3.71 eV |
| Hoppings ssσ, spσ, ppσ, ppπ | GSP con r0 = 1.536329 Å, n = 2, nc = 6.5, rc = 2.18 Å | v0 = −5.0, 4.7, 5.5, −1.55 eV |
| Cola | cúbica entre 2.45 y 2.6 Å | calculada de las condiciones de continuidad |
| Repulsión | E_rep = Σ_i f(Σ_j φ(r_ij)), φ GSP (v0 8.18555 eV, r0 1.64, n 3.30304, nc 8.6655, rc 2.1052), f polinomio de grado 4 | coeficientes de la tabla I |
| Base | ortogonal, s + p | |

**Energía** (`forces.py`): F = Σ_{k,n} w_k f_n ε_n − T S_el + E_rep, con
T S_el la entropía de Fermi–Dirac. **Fuerzas**: Hellmann–Feynman,
∂E/∂R = Tr[ρ ∂H] − Tr[W ∂S] + ∂E_rep; las derivadas de los bloques por
diferencias centrales (1e-5 Å); probadas contra diferencias finitas de la
energía.

**Discrepancias y cosas por verificar frente al artículo de Xu:**

1. Los números se transcribieron de la tabla I y se validaron solo de forma
   indirecta (constante del diamante 3.555 frente a 3.567 Å medidos, enlace
   del grafeno 1.42 Å, cohesión 7.24 frente a 7.37 eV). **No se han cotejado
   línea a línea con el PDF.** Si tienes el artículo, esa comprobación es la
   primera tarea pendiente.
2. Xu da coeficientes explícitos para las colas; tbkit los recalcula de
   cuatro condiciones (valor y pendiente en 2.45 Å, cero y pendiente cero en
   2.6 Å). Es la misma cúbica si la de Xu cumple esas condiciones; por
   verificar con la tabla.
3. **Añadidos que no son de Xu**, declarados en el JSON: la U de Hubbard del C
   (10.0 eV, «orden de magnitud»; el valor calculado con el átomo de GPAW es
   9.9228 eV), el dipolo intraatómico d(C) = 0.4952 Å (GPAW) y la
   polarizabilidad extra α(C) = 0.953 Å³ (ajustada a GPAW FD). Solo afectan a
   SCC y a la respuesta óptica, no a energías ni fuerzas no SCC.

## 4. Conjunto C/H/N (`parameters/xu_chn.json`)

- C–C idéntico a Xu. H (s) y N (s, p): on-site, hoppings C–H, N–H, C–N, N–N
  en la forma GSP de Xu (r0 por par, nc = 6.5, rc y cola escalados por
  r0/1.536), repulsión por pares con polinomio de corte.
- Ajuste a GPAW (PBE, LCAO dzp): niveles con un desplazamiento común;
  después niveles + fuerzas + energías con la repulsión resuelta exactamente
  (lineal). 118 geometrías de 16 moléculas; 6 de prueba (incluido un
  coroneno con N piridínico).
- U de H, C, N: dε/dn del átomo libre con GPAW (= DFTB mio).
- **Estado fundamental SCC** (DFTB2, γ de Klopman–Ohno), solo en finitos.
- Validado: enlaces X–H y aromáticos ≤ 0.02 Å de GPAW; frecuencias con RMS
  de 50–70 cm⁻¹ (CH₄, NH₃, benceno, piridina). Límites: C–C/C–N simples junto
  a heteroátomos ~0.08 Å; energías solo entre geometrías de igual
  composición; sin H–H.

## 5. Fonones

- **Moléculas y celdas (Γ)**: `ase.vibrations`, diferencias centrales de las
  fuerzas con δ = 0.005 Å (`tasks.phonons`, `raman._model_phonons`),
  calculadora con kT = 0.02 eV por defecto. Hessiana ponderada por masas
  (masas estándar de ASE), diagonalizada; modos L = e/√m con e ortonormal
  (convención de ASE, comprobada en su código). Frecuencias imaginarias
  como negativas.
- Se descartan 3 (cristal), 5 (lineal) o 6 modos rígidos por su |ω| más
  pequeño (`raman.internal_modes`); se avisa de frecuencias imaginarias
  < −20 cm⁻¹ y de fuerzas residuales > 0.05 eV/Å.
- **Grafeno en toda la zona** (`graphene.GraphenePhonons`): constantes de
  fuerza en supercelda n×n, imagen mínima con pesos repartidos, D(q) con fases
  de posiciones.
- **Fonones externos**: modos de QE en Γ (`qe.py`).
- **No hay**: dispersión q de sistemas que no sean grafeno (CNT, cintas),
  DOS vibracional, análisis de participación por especie, separación masa /
  efecto químico, IR.

## 6. Polarizabilidad y Raman

- α finita apantallada: δM = (1 − χK)⁻¹ χ V con M = (cargas, dipolos
  intraatómicos, dipolos extra) y el núcleo de Klopman–Ohno y sus derivadas;
  comprobada contra campo finito y, sin apantallar, contra la suma sobre
  estados.
- α de cristales: suma interbanda con ∂ₖH (+ dipolos intraatómicos), sin
  campos locales.
- Raman no resonante: tensores ∂α/∂Q por diferencias centrales (δ = 0.01 Å),
  actividad 45a′² + 7γ′² (Å⁴/amu), ρ = 3γ′²/(45a′² + 4γ′²).
- Raman resonante: ∂α(ω_L + iη)/∂Q (invariantes complejos).
- Grafeno: G de tercer orden y 2D/2D′ por doble resonancia (cuarto orden).
- Validado: reglas de selección (diamante T₂g, C₆₀ 2A_g + 8H_g, benceno
  2A₁g + 4E₂g + E₁g), α de C₆₀ 84.5 frente a 76.5 ± 8 Å³, tensores α a ±5 %
  de GPAW.

## 7. Tests (`tbkit/tests`, 152 rápidos + lentos nocturnos)

| Archivo | Qué garantiza |
|---|---|
| test_core | grafeno, Hückel, nanotubos con el orbital π, bandas |
| test_magnetism | Hubbard de campo medio, teorema de Lieb |
| test_parameters | `.skf`, ajuste, convención heteronuclear |
| test_energy | fuerzas = −∇E (ortogonal, no ortogonal, periódico, SCC) |
| test_reproducibility | cada número con unidad y fuente, registros reproducibles |
| test_raman | reglas de selección, α, ε∞, C₆₀ (lento) |
| test_chn | conjunto C/H/N: huellas de referencias, geometría, frecuencias |
| test_qe | modos de QE importados = modos propios |
| test_dipoles | respuesta lineal = campo finito, invariancia de la convención de p |
| test_resonance | límite estático, polieno resonante, prohibidos siguen prohibidos |
| test_graphene | acoplamiento e-fonón = suma explícita, simetrías, 2D (lento) |

## 8. Clasificación

| Parte | Estado |
|---|---|
| Álgebra Slater–Koster, H(k), fuerzas, SCC, respuesta lineal | **Exacta dentro del modelo**; comprobada con relaciones cerradas y diferencias finitas |
| Parámetros de Xu | Publicados; **transcripción sin cotejar con el PDF** |
| Conjunto C/H/N | Ajustado a GPAW; validado en moléculas; **sin validar en nanotubos ni en carbono extendido dopado** |
| Fonones en Γ | Correctos en método; precisión limitada por el modelo (Xu: G 1653–1667 frente a 1582 cm⁻¹; diamante 1224 frente a 1332) |
| Raman de moléculas | Reglas de selección y tendencias fiables; intensidades absolutas orientativas |
| Nanotubos: RBM, G, dependencia con el diámetro | **No validado** (siguiente paso) |
| IR | **No implementado** |
| O, B, S, P | **Sin parámetros** |

## 9. Propuesta (en este orden)

1. **Validación con CNT prístinos**: ω_RBM(d) y la G (G⁺/G⁻) para zigzag,
   armchair y quirales; errores frente a relaciones de la literatura. Con
   `xu_carbon` en celdas periódicas (sin SCC) y, para los finitos pasivados,
   `xu_chn`.
2. **Análisis de modos** (nuevo módulo `modes.py`): participación por especie
   P_X(ν) = Σ_{i∈X} |e_i(ν)|², proyecciones radial/tangencial/axial en
   tubos, DOS vibracional total y proyectada con ensanchamiento configurable,
   y separación masa / efecto químico (recalcular con la masa del C en el
   dopante y las mismas constantes de fuerza). Guardar siempre los modos.
3. **IR**: intensidades |∂μ/∂Q|² con μ del estado fundamental (cargas SCC +
   dipolos intraatómicos) en moléculas y fragmentos finitos.
4. **Oxígeno** (hidroxilo, epoxi, carbonilo, carboxilo, éter), con la misma
   receta que el N: referencias GPAW, ajuste conjunto, validación.
5. **B, S, P** después, con la misma receta y su propia validación.
6. **Frecuencias en la función objetivo** del ajuste (L = w_E L_E + w_F L_F +
   w_ω L_ω), con peso moderado.
7. Cambios menores detectados: poner en `xu_carbon` la U calculada (9.9228 eV)
   en vez del «orden de magnitud» 10.0.

Descartado (por decisión del usuario o por no encajar): FeSe y su interfaz;
potenciales clásicos (Tersoff, REBO, ReaxFF) dentro de tbkit — son del lado
LAMMPS de carbonforge; reorganizar carpetas o mover geometría entre paquetes;
un «Xu no ortogonal» propio (los `.skf` de DFTB y, más adelante, NRL-TB cubren
ese papel).
