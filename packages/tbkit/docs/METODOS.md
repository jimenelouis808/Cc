# Métodos de tbkit

Qué calcula cada módulo, con qué ecuaciones, cómo se comprueba y dónde deja de
valer. Unidades: eV y Å en todo; los `.skf` se convierten al leerlos. Las
ecuaciones son las de las docstrings de cada módulo, que mandan si algo
difiere. Los números de validación están en [`VALIDACION.md`](VALIDACION.md)
(generado desde los datos) y el uso en [`GUIA_USUARIO.md`](GUIA_USUARIO.md).

## 1. Hamiltoniano y solución

**H(k) y S(k)** (`hamiltonian`). Bloques de dos centros de Slater–Koster
(ssσ, spσ, ppσ, ppπ; `slater_koster`) con leyes de distancia constante,
exponencial, Harrison, GSP o tabla, y colas suaves (`Tail`) para que fuerzas y
fonones sean continuos. En periódicos, `H_ij(k) = Σ_R h_ij(R) e^{ik·R}` sobre
las imágenes de la lista de vecinos de ASE. El modelo π usa un orbital π local
normal a la superficie, sin factor angular (un `pz` cartesiano rompe los tubos;
`test_core`). Se comprueba H = H† y S > 0 en cada construcción.

**Ocupaciones** (`solver`). Fermi–Dirac de anchura kT; el nivel de Fermi fija
el número de electrones. `Solution.gap()` vale 0 si algún estado tiene ocupación
fraccionaria: es la definición de «sin gap» que usan Raman y los avisos.

**Malla k** (`kpoints`). Mallas Γ-centradas solo en los ejes periódicos (los
puntos K del grafeno caen en la malla si n es múltiplo de 3) y caminos de
bandas de ASE.

## 2. Cargas autoconsistentes (SCC)

`scc`: DFTB2. Cada átomo con carga Δq_A desplaza los orbitales,

    V_A = Σ_B γ_AB Δq_B,     H_μν += ½ S_μν (V_A + V_B),
    γ_AB = e² / sqrt(r² + (e²/Ū)²),  Ū = (U_A + U_B)/2   (Klopman–Ohno).

En los xu_ch* las U son calculadas (átomo de GPAW, dε/dn), no ajustadas. Mezcla de Anderson;
las fuerzas SCC necesitan carga convergida a 1e-10.

**Periódicos** (`ewald`; Elstner 1998): C/r por Ewald, la asíntota −C a²/2r³
por un segundo Ewald (su término G = 0, una constante, se descarta y se dice),
y el resto, que cae como r⁻⁵, en espacio real con un corte suave. Las
direcciones sin periodicidad son una supercelda con su vacío. Comprobado
(`test_ewald`): constante de Madelung del NaCl, una molécula en una caja grande
igual al resultado finito, fuerzas frente a diferencias finitas (h-BN).

## 3. Energía total y fuerzas

`forces`: energía libre de Mermin

    F = Σ w_k f_n ε_n − T S_el + E_rep (+ E_SCC),

y fuerzas de Hellmann–Feynman en base no ortogonal,
`∂E_band/∂R = Σ_k w Re Tr[ρ ∂H/∂R − W ∂S/∂R]`, con ρ = Σ f c c† y
W = Σ f ε c c†. Repulsión (`repulsive`): por pares, `½ Σ V_ab(r)` (spline de los
`.skf`), o embebida, `Σ_i f(Σ_j φ(r_ij))` (Xu); términos de ángulo agudo para
anillos de tres miembros (cero desde 80°) y el contacto H···H (pared de GPAW
H₂···H₂, corte quíntico 2,0–2,4 Å). Toda vía de fuerzas se compara con
diferencias finitas de la energía (`test_energy`, `test_environment`,
`test_hh_contact`). El modelo π no tiene repulsión y se niega a dar energías.

## 4. Carbono dependiente del entorno (Tang 1996)

`environment`: cada salto, el desplazamiento on-site Δe y la repulsión siguen

    h(r_ij) = α1 R_ij^−α2 exp(−α3 R_ij^α4) (1 − S_ij),

con apantallamiento S_ij = tanh ξ_ij por los átomos cercanos a la línea i–j y
distancia escalada R_ij por las coordinaciones efectivas de ambos extremos.
Cortes suaves (no están en el artículo; declarados). Las coordinaciones del
artículo se reproducen a 1e-3. Los parámetros publicados no dan el diamante
del artículo (`tang1996_published.json`, solo pruebas); el conjunto usable,
`tang_carbon`, conserva la parte electrónica publicada y ajusta a GPAW la escala
de Δe, el corte y la repulsión (`recipes/tang_fit.py`), con amorfo 3,2 y
Stone–Wales fuera del ajuste.

## 5. Magnetismo (Hubbard de campo medio)

`hubbard`: `H_σ = H_0 + Σ_μ U_μ (n_{μ,−σ} − n0_μ/2) − σh`, autoconsistente.
Magnetización frente a la energía (m(E), dm/dE), al campo (M(h), χ) y al
dopaje. Comprobado: teorema de Lieb (triangulene M = 2), bordes de cintas
zigzag, grafeno sin momento por debajo de U_c (`test_magnetism`). Los momentos
son el parámetro de orden del campo medio, no un estado correlacionado.

## 6. Respuesta óptica

**Polarizabilidad** (`optics`). Suma sobre estados,

    α_ab(ω) = e² Σ_k w_k Σ_{n≠m} (f_n − f_m) Re[r^a_nm r^b_mn] E_mn / (E_mn² − (ħω)²),

con r = C†RC en finitos y r_nm = i⟨n|∂_k H|m⟩/E_mn en cristales (α por celda;
ε∞ = 1 + 4πα/V). Cada par de estados se cuenta una vez (con ensanchamiento un
estado puede estar «ocupado» y «vacío»). Respuesta lineal SCC apantallada: en
finitos, cargas y dipolos; en cristales, campos locales de carga vía Ewald (los
dipolares aún no). Comprobado: la respuesta lineal es la de campo finito; una
molécula en caja da la α finita con su campo de Lorentz; el h-BN prístino da
la α sin apantallar.

**Dipolos intraatómicos** (`dipoles`). `⟨s|r_α|p_β⟩ = d δ_αβ`, d calculado del
átomo libre de GPAW (no ajustado), con el signo tomado de la convención del
propio modelo (cambiar el signo de todo spσ deja α igual; probado). La única
cantidad óptica ajustada es una α extra por elemento (`extra_polarizability`,
frente a tensores de GPAW).

**Infrarrojo** (`infrared`). μ = Σ Q_A R_A − Σ p_A con el mismo operador de
posición que α; cargas de Born por diferencias centrales; intensidades
|∂μ/∂Q|² en km/mol. Las cargas de Born de una molécula neutra suman cero.

**Corrección de cargas para el IR** (`charge_model`, receta `recipes/ir_charge_fit.py`,
medida, no adoptada por defecto). Se suma a las cargas de Born de TB una corrección
de clase IV: `BondFlux`, de tipo CM3, con 2 parámetros por par de elementos (6 en C/H/N y
12 con O); `EnvironmentFlux`, lineal en SOAP (aprendizaje automático); o ambas. Las tres
son neutras por construcción y nulas en carbono puro. Se ajustan con ridge a las cargas
de Born de GPAW de 44 moléculas y 3 recortes de la coil (`recipes/born_references.py`),
validando con exclusión de una estructura a la vez. Medido
(`validation/ir_charge_models_{chn,chno}.json`): `BondFlux` baja el error de las cargas
de Born de 0,098 a 0,084 e (C/H/N) y de 0,100 a 0,083 e (con O); el IR de las
estructuras apartadas pasa del 46 % al 52–54 % de modos dentro de un factor 2, y en los
modos de la coil comparados con GPAW la mediana de |log₁₀ I/I_GPAW| pasa de 0,225 a
0,202, con la misma fracción dentro de ×2 (65 %). El modelo SOAP se sobreajusta: queda
peor que TB sin corregir en validación cruzada (0,104–0,116 e). La mejora de `BondFlux`
es real pero pequeña; el IR por defecto sigue sin corrección y su incertidumbre medida
(por banda, ±40 % en la coil) es la que se declara.

## 7. Raman

**No resonante** (`raman`). Tensor dα/dQ_k = Σ (∂α/∂x_ai) L_k[a,i], actividad
S = 45a′² + 7γ′² y despolarización ρ = 3γ′²/(45a′² + 4γ′²); el espectro añade
(ν_L − ν)⁴ y el factor de Bose. Exige gap: rechaza metales, semimetales, capa
abierta y láseres a menos del 20 % del gap. Primera comprobación de cualquier
cambio: reglas de selección (diamante T2g, C60 2Ag + 8Hg).

**Resonante** (`resonance`; Gillet, Giantomassi y Gonze 2013). Lo mismo con la
α compleja α(ω_L + iη), η > 0; su límite estático bajo el gap es el no
resonante (probado). Con `select` deriva α a lo largo de cada modo,
(α(x + hL) − α(x − hL))/2h, el mismo tensor con 2 α por modo (probado en
benceno y diamante, δ = 0,002 Å; con δ grande el error es mayor que átomo por
átomo). Las energías de resonancia son las del modelo, no ópticas.

**Grafeno** (`graphene`; Thomsen y Reich 2000, Venezuela, Lazzeri y Mauri
2011). G de tercer orden y 2D/2D′ de cuarto orden por doble resonancia:
electrones del modelo π, luz en el gauge de velocidad, acoplamiento
electrón–fonón analítico (t′(a) = −βt/a, comprobado contra una suma explícita
de ΔH) y fonones interpolados en toda la zona (GPAW guardados, o Xu).

**Cualquier modelo periódico** (`double_resonance`, receta
`coil_double_resonance`). El mismo esquema para un Hamiltoniano de tbkit
cualquiera. El vértice electrón–fonón ⟨k+q|∂H|k⟩ sale de las derivadas de los
saltos por enlace, en el gauge de posiciones fijas: con las posiciones
desplazadas en la fase de Bloch aparece un término espurio ∝ k. Un elemento se
comprobó contra el valor analítico del grafeno. Los fonones en q vienen de
Φ(R) en una supercelda (D(0) reproduce los modos Γ); el defecto de la D es un
vértice de un sitio. La 2D suma todos los pares de ramas (q,ν), (−q,ν′): con
solo las diagonales salía unas 100 veces más débil en celdas grandes. El
último denominador se factoriza con una serie de Taylor de orden 7 en
ω_ν + ω_ν′ − 2ω̄. La misma maquinaria en grafeno da I_2D/I_G = 12 a 2.33 eV.
En la coil (xu_carbon, γ 0.1 eV) la D y la 2D dispersan solo unos 10 cm⁻¹/eV,
frente a ~50 y ~100 en grafeno. Las intensidades dependen de γ y las energías
de resonancia son las del modelo.

**Modos de otras fuentes** (`qe`): modos de Quantum ESPRESSO en q = 0
(dynmat/matdyn), con la fase de cada modo fijada y bases reales de los
subespacios degenerados; el modelo solo da α.

## 8. Vibraciones

**Γ** (`modes`, `tasks.phonons`): diferencias finitas de las fuerzas
(`ase.vibrations`), hessiana guardada, L = e/√m. Análisis: participación por
grupos, carácter cilíndrico, solapamiento con la respiración, simetría
rotacional, DOS vibracional. Factores de escala de frecuencias por conjunto
(Scott–Radom, frente a GPAW; `recipes/frequency_scaling.py`) siempre aparte de
las crudas.

**Sistemas sin gap**: las frecuencias de los modos que acoplan con la superficie
de Fermi dependen de la malla k. En grafeno con Xu y kT = 0,05 eV, G vale 1572
cm⁻¹ con 12 k por eje y 1674 convergida (≥ 48). `modes.vibrations` y
`tasks.phonons` avisan; `tasks.kmesh_convergence` sube la malla.

**Toda la zona** (`phonopy_bridge`, extra `phonons`): phonopy elige los
desplazamientos, tbkit da las fuerzas. Dispersión, DOS por elemento,
termodinámica armónica y representaciones irreducibles en Γ. En diamante, Γ
coincide con los fonones directos a 0,2 cm⁻¹ y el modo Raman es T2g (probado).

**Sitios** (`sites`). Anillos = ciclos simples de hasta 7 átomos que cierran en
el espacio (se siguen los desplazamientos de red: un ciclo que da la vuelta a la
celda no es un anillo). Grupos: anillos de 5, 6 y 7, cada heteroátomo, sus C
vecinos, H. `projected_frequency`: frecuencia de un patrón de desplazamiento con
cualquier calculadora, ω² = u·K·u / u·M·u (cociente de Rayleigh: exacto para un
modo propio de esa calculadora, cota superior si no; probado en H₂O).

## 9. Parámetros y ajustes

Los conjuntos viven en `tbkit/parameters/*.json`: cada número con unidad y
fuente, cada archivo con referencia, sistema y validez (lo exige
`test_reproducibility`). Ajustes reproducibles en `recipes/` a referencias GPAW
(PBE, LCAO dzp) guardadas en `parameters/references/`; cada xu_ch* guarda el SHA-256 de
sus datos. Un ajuste
guarda un archivo nuevo y reporta cada cambio. `xu_chn` conserva el C–C de Xu;
`xu_chno` añade O; `xu_chnob/s/p/se` añaden un elemento sobre `xu_chno` fijo
(sin ese elemento dan exactamente lo de `xu_chno`; probado). Mínimos espurios:
`recipes/active_learning.py`.

## 10. Aprendizaje automático (medido, no adoptado)

`hybrid`: geometría y modos de cualquier calculadora de ASE (DFT, un potencial
de aprendizaje automático) y α/μ de TB por la vía `phonons=(frecuencias, L)`
que ya usan los modos de QE (probado: con la calculadora de tbkit da los modos
de tbkit). `delta`: corrección lineal en SOAP sobre un conjunto TB,
E = E_TB + Σ_i (w_Z · x_i + b_Z), ajustada por mínimos cuadrados con ridge
(elegido por validación cruzada dejando fuera un grupo), con una medida de
novedad por átomo (1 − similitud coseno máxima con el entrenamiento). Probado:
fuerzas = −∇E y reproducción exacta de una corrección lineal.

Medido frente a GPAW (`VALIDACION.md`): MACE-MP-0 sin ajuste fino no mejora a TB
en moléculas y ablanda la coil un 11–18 %; la corrección Δ no generaliza con los
datos guardados (en moléculas no vistas empeora). Ninguno se usa por defecto;
ambos quedan para cuando haya un potencial ajustado o muchos más datos DFT.

## 11. Límites de fondo

- Base mínima s+p: sin d ni polarización (los OH de ácidos fosfónicos y
  selenínicos giran al relajar; medido como error de base, no corregible con
  términos repulsivos).
- Conjuntos ajustados en moléculas: en cristales, extrapolación medida en 13
  casos; fuera de ellos, no comprobada.
- Energías relativas solo entre estructuras de la misma composición y los mismos
  tipos de enlace (no se ajustaron energías de atomización): dimetil éter frente a
  etanol sale −1,44 eV con xu_chno y +0,38 con GPAW. `tang_carbon` compara entre
  estructuras de carbono, con poca fiabilidad entre fases muy distintas. Un
  ranking de sitios o isómeros con TB se confirma con DFT.
- Sin excitones; gaps del modelo (y de Kohn–Sham) menores que los ópticos.
- Sin banda D de defectos; la anomalía de Kohn física no está en Xu (débil).
