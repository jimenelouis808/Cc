# Celdas electroquímicas modulares para mediciones in situ

Proyecto de diseño paramétrico (CadQuery, Python) de celdas electroquímicas modulares para:
1. Raman in situ (Renishaw inVia, microscopio confocal)
2. XRD in situ (Bruker D8 Advance, Bragg-Brentano θ-θ)
3. Sonoelectrodeposición (baño ultrasónico o sonda)
4. Electrodeposición electroforética (EPD) de CNT/CNF
5. Decoración de nanotubos con nanopartículas (electrodeposición, 3 electrodos)

Electrolitos objetivo: 0.5 M H2SO4 y 3 M KOH. EPD: etanol / IPA (y posiblemente acetona).

## Reglas de diseño (no negociables)

- **Todo es paramétrico.** Ninguna dimensión hardcodeada fuera de `params.py`. Cada módulo importa de ahí.
- **Unidades: mm.** Origen: centro de la superficie del electrodo de trabajo (WE) = plano de referencia (z = 0).
  El plano z = 0 es el plano del goniómetro en XRD y el plano focal en Raman.
- Cada script de módulo exporta STEP (para simulación/maquinado) y STL (para impresión) a `out/`.
- Cada módulo tiene una función `build(p=params) -> cq.Workplane` y un bloque `if __name__ == "__main__"` que exporta.
  `p` es el módulo params; para variantes usar `modules._geom.variant(p, CELL={...})`, nunca editar params en pruebas.
- Cada módulo expone además `features(p) -> {nombre: sólido}` (lo que se resta del cuerpo):
  `checks/walls.py` mide paredes entre esos elementos. Al agregar un módulo, agregar sus checks ahí.
- Geometría compartida (huella, tornillería, sellos, puertos) en `modules/_geom.py`.
- Antes de dar por terminado un módulo: correr `checks/` (interferencias, espesores mínimos de pared, holgura de haz XRD)
  y `python -m pytest -q`.
- Lecciones de celdas publicadas en `docs/referencias.md`: consultarlas antes de diseñar un módulo nuevo.

## Materiales

| Uso | Material | Notas |
|---|---|---|
| Prototipo de ajuste (sin electrolito) | PLA / PETG | Solo para verificar ensambles |
| Cuerpo final impreso | **PP** | Resiste H2SO4, KOH y alcoholes. 100 % relleno, ≥4 perímetros |
| Cuerpo final maquinado | PTFE o PEEK | Si el PP impreso fuga o se deforma |
| O-rings | **EPDM** | NO usar FKM/Viton (se degrada en KOH concentrado) |
| Ventana Raman en ácido | Cuarzo / sílice fundida, ≤0.5 mm | |
| Ventana Raman en KOH | **Zafiro** ≤0.5 mm o película FEP | La sílice se ataca en KOH 3 M |
| Ventana XRD en ácido | Kapton 8–25 µm | |
| Ventana XRD en KOH | Película PP o PE (tipo XRF) | **Kapton se hidroliza en KOH** |
| Prohibidos con KOH | PLA, PETG, PC, resinas SLA estándar | Hidrólisis / lixiviación |
| Prohibidos con acetona (EPD) | PMMA, ABS | |

## Electrodos (ver `params.py`, valores marcados MEDIR deben confirmarse con calibrador)

- RE: Ag/AgCl sat. KCl (Camray). En KOH considerar Hg/HgO → puerto de RE con adaptador intercambiable.
- CE: alambre de Pt (en HER ácida preferir barra de grafito: el Pt se disuelve y redeposita en el WE).
- WE A: glassy carbon Ø3 mm (entra **por abajo**, cara hacia arriba, sello radial).
- WE B: sustrato plano sellado con O-ring (área = ID del O-ring), contacto con lámina por debajo
  (`base_flat` + `chamber_ring`).
- WE C: tira sujeta con portaelectrodo de pinza de Pt que entra por un puerto lateral, sumergida,
  sobre un pedestal que fija su cara en z = 0 (`chamber_clamp`, como la celda R1). La pinza se moja.

## Arquitectura modular ("sándwich")

```
[ tapa / módulo de ventana ]   ← intercambiable por técnica
[ anillo-cámara con puertos ]  ← RE (Luggin), CE, entrada/salida electrolito
[ base de WE ]                 ← intercambiable: plano (pinza/placa) o GC desde abajo
```
- Anillo-cámara y base se unen con tornillos M3 (insertos o tuercas embebidas) + O-ring de cara.
- El área expuesta del WE plano la define el diámetro interior del O-ring de cara → reportar como parámetro.
- La cámara tiene una **zona de capa delgada** sobre el WE y un **reservorio lateral más profundo** donde viven RE y CE.

### Módulo Raman (inVia)
- Tapa con "émbolo" que baja la ventana hasta una distancia `gap` del WE (0.1–0.5 mm, ajustable con lainas).
- Restricción: altura de todo lo que rodea la ventana < distancia de trabajo del objetivo (WD), con margen ≥1 mm.
- La cara superior alrededor de la ventana debe dejar libre el diámetro de la punta del objetivo.
- Ventanas delgadas (≤0.5 mm) para minimizar aberración esférica en confocal.
- El objetivo se modela como cara frontal + nariz cónica + barril (no como cilindro de Ø barril).
  Punta en `z_tip = WD + Σ tᵢ(1 − 1/nᵢ)` sobre el WE (ventana y electrolito alargan el foco).
  Código en `modules/_optics.py`; checks en `checks/raman_clearance.py` y `checks/walls.py`.
- Implementado (`modules/lid_raman.py`): tapa de una pieza (placa + émbolo hueco + nariz con
  ventana pegada al ras). La placa apoya sobre lainas en los tornillos; el hueco cubre el objetivo
  y el cono de luz en todo el rango `gap_range_raman`. `exposed_d` ≥ 9.5 para que la nariz tenga
  pared de 2 mm alrededor del cono de NA 0.5.

### Módulo XRD (D8 Advance)
- θ-θ: la muestra queda horizontal (el electrolito no se derrama).
- **Perfil bajo obligatorio**: nada puede sobresalir del plano z = 0 dentro del cono del haz.
  Usar `checks/xrd_clearance.py` para la altura máxima permitida vs distancia al centro.
  Bordes de la celda biselados a lo largo del eje del haz.
- Película (Kapton o PP) sostenida con anillo tensor, electrolito ≤ ~50 µm sobre la muestra durante la medición
  (la capa de agua atenúa fuertemente Cu Kα).
- Opción alternativa a evaluar: catalizador sobre papel de carbón como pared de la celda, con el catalizador
  hacia el electrolito (abajo) y el haz entrando por la cara trasera. Ojo: C(002) del papel ~26° 2θ.
- Ajuste de altura (tornillo o lainas) para llevar la superficie del WE al plano del goniómetro.
- Implementado (`modules/chamber_xrd.py`, `modules/lid_xrd.py`): cara superior plana en z = 0,
  película sellada con cordón bajo un marco que se bisela sólo dentro del corredor del haz
  (|y| ≤ `beam_length_y`/2 + margen). Fuera del corredor la altura no importa: tornillos,
  pinza y marco grueso van ahí. Puertos sólo en ±y (el cuerpo se niega a construir uno en ±x).
- WE con pinza de Pt en XRD: la pinza toma una lengüeta 4.5 mm bajo la muestra (decisión del
  usuario). Con la pinza a la altura de la muestra el sello de la película tendría que subir
  sobre el portaelectrodo.
- Verificación del haz: `forbidden_region` + `invasion` en `checks/xrd_clearance.py`.

### Módulo electrodeposición Chladni (WE = placa vibrante)
- El WE es una placa delgada excitada en modos de flexión (piezo pegado o bocina con varilla al centro).
- **Objetivo: patronado intencional** (un patrón distinto por frecuencia/modo).
- **Diseño elegido:** placa CUADRADA con borde empotrado entre dos marcos rígidos atornillados
  + disco piezo pegado por debajo **fuera del centro** (≈ L/4 de cada borde) para excitar también modos no
  axisimétricos. El O-ring de sello va sobre la placa, justo dentro de la línea de empotramiento
  (zona de desplazamiento casi nulo → poco amortiguamiento).
- Descartado por ahora: placa montada al centro con bordes libres (patrones clásicos más ricos, pero
  contener el electrolito sin amortiguar los bordes libres es muy difícil). Reconsiderar solo si los
  modos del diseño empotrado resultan insuficientes.
- Punto de partida: Ti grado 2 o Ni, L = 30–40 mm, h = 0.2–0.5 mm (ver `sim/chladni_estimate.py`;
  los modos caen en rango de audio → generador de funciones + amplificador de audio/piezo).
- Identificar resonancias reales con barrido de impedancia del piezo, con la celda llena.
- Para máximo contraste del patrón: régimen limitado por transporte de masa (baja concentración de ion
  metálico), sin agitación externa.
- Parámetros extra: posición del piezo, diámetro del piezo, ancho del marco de empotramiento,
  torque/espesor de los marcos.
- **Aislamiento eléctrico obligatorio entre piezo y WE** (capa de Kapton/epoxi): evitar lazo de tierra
  con el potenciostato. Blindar cables del driver; verificar ruido en CV sin y con excitación.
- Solo la cara superior de la placa expuesta; reverso y bordes aislados.
- Parámetros: material y espesor de placa, diámetro/lado, tipo de montaje, frecuencia, amplitud,
  modo objetivo, altura de electrolito (carga de masa: baja las frecuencias respecto a placa seca).
- Modo fijo → depósito patronado. Barrido de frecuencia (chirp) → homogeneización.
- Calibrar modos primero en seco (arena/sal) y luego en líquido con trazadores.

### Requisitos para depósitos homogéneos y reproducibles (todos los módulos de deposición)
- Distribución primaria uniforme: WE y CE paralelos en un canal cuyas paredes aislantes coinciden
  con los bordes de los electrodos, o WE ligeramente rehundido en pared aislante (evitar bordes expuestos).
- CE de área ≥ WE, simétrico, alejado; RE con Luggin fuera de la línea WE–CE.
- Transporte de masa controlado (flujo laminar, chorro, vibración) y temperatura controlada (chaqueta).
- Posición fija con pines guía; volumen fijo; desgasificación; control por carga (coulombimetría).
- Orientación que permita liberar burbujas si hay co-evolución de H2.

### Módulo EPD / DEP selectivo por diámetro
- Concepto: canal con flujo laminar (bomba de jeringa preferible al goteo) y campo transversal.
  La posición de captura a lo largo del canal codifica la movilidad / respuesta dieléctrica.
- La movilidad electroforética casi no depende del tamaño (Smoluchowski); la selectividad real viene de
  dielectroforesis (DEP, AC), cuya fuerza escala con el volumen de la partícula y con ∇|E|².
- Electrodos interdigitados o almenados con gap que varía a lo largo del canal (gradiente de campo),
  depositados por sputtering con máscara de sombra sobre vidrio.
- Medio de baja conductividad para DEP. Si es EPD DC en agua: considerar electrólisis (>1.23 V).
- Simulación: electrostática + trazado de partículas (arrastre de Stokes + EP + DEP + sedimentación).

### Módulo decoración con NPs
- Reutiliza base + anillo-cámara; tapa con puerto para agitación y purga de N2.

## Sellos (ver `params.py: oring_gland()`)
- Sello de cara estático: profundidad ≈ 0.75·cs, ancho ≈ 1.5·cs.
- Sello radial sobre vástago de electrodo: profundidad radial ≈ 0.80·cs, ancho ≈ 1.4·cs.
- Valores de partida; verificar contra el manual de O-rings (Parker) y calibrar con cupón impreso.

## Tolerancias FDM
- Agujeros: +0.2–0.3 mm sobre nominal (calibrar con cupón).
- Pared mínima en contacto con electrolito: 2.0 mm.
- Orientar piezas para que las ranuras de O-ring queden en caras superiores al imprimir.

## Orden de trabajo
1. `coupons/tolerance_coupon.py`: agujeros Ø 3/6/8 mm con offsets 0–0.4 mm y ranuras de O-ring
   con 3 profundidades. Imprimir en PP y medir.
2. `modules/base_flat.py` (WE plano) y `modules/chamber_ring.py`.
3. `modules/lid_raman.py` → prueba con agua antes de electrolito.
4. `modules/lid_xrd.py` + `checks/xrd_clearance.py`.
5. Chladni, EPD/DEP, decoración.
6. Simulación: STEP → gmsh → FEniCSx o Elmer
   - modos propios de la placa Chladni (seca y con carga de fluido)
   - distribución de corriente secundaria (número de Wagner) en celdas de deposición
   - campo eléctrico y trayectorias de partículas en EPD/DEP

## Estructura
```
params.py
export.py       # export(wp, nombre) → out/<nombre>.step y .stl
modules/        # un archivo por módulo
checks/         # verificaciones geométricas
coupons/        # piezas de calibración de impresión
sim/            # mallado y simulación
docs/           # referencias.md: lecciones de celdas publicadas
tests/          # python -m pytest -q
out/            # STEP / STL generados (no editar, no versionar)
```

## Relación con nanocarbon (repositorio `Cc`)
Proyectos separados: ninguno importa código del otro. Si la simulación de EPD/DEP necesita
propiedades de los CNT (diámetro, longitud, polarizabilidad, movilidad) calculadas con
`nanocarbon_lab`, se intercambian **por archivo** (JSON/CSV con unidades explícitas), con lector y
pruebas de cada lado. Mismo criterio que el workspace `Cc` sigue entre sus paquetes.
