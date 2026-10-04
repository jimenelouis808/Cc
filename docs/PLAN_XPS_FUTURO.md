# Plan a futuro: XPS teórico centrado en sitios (GPAW + TB)

Resumen de *GPAW_XPS_Site_Centric_Workflow.md* (documento del usuario, no
versionado aquí). **No empezado**: queda en el plan, después del Raman de
nanocoils. ramancarbon ya lee y ajusta XPS experimental; esto añade el lado
teórico.

## Idea central

La unidad es el **sitio atómico**, no el pico: sitio → entorno químico local →
estructura electrónica → nivel de core → desplazamiento (ΔE respecto de un
sitio de referencia) → componente XPS → espectro sintético. Cada pico debe
poder rastrearse hasta sus sitios.

## Piezas

- `Site`, `SiteDescriptor`, `ReferenceSite`, `CoreLevelResult`, `XPSComponent`.
- Clasificador de entornos (C: C–C, C–N, C–S, C–Se, C–O, C=O, defecto; N:
  piridínico, pirrólico, grafítico, amina, oxidado; S, Se, Fe) con los rasgos
  que llevaron a cada etiqueta, y agrupación de sitios equivalentes (huella de
  distancias, no solo el elemento).
- Selección de sitios para core-hole (`rank_sites_for_xps`): dopantes, primeros
  y segundos vecinos, defectos, referencias prístinas; nunca todos los C.
- ΔSCF con la API de la versión instalada de GPAW (verificar antes; no inventar
  APIs), convergencia del **desplazamiento**, no solo de la energía total.
- Espectro sintético (Voigt por defecto; dobletes S 2p y Se 3d; Fe 2p fuera de
  un modelo de una partícula), intensidad por población primero.
- Bases acumulativas: sitios, entornos, componentes. Alineamiento con el
  experimento siempre explícito; nunca ajustar el teórico al experimental.
- Después: descriptores TB (cargas, LDOS de tbkit) → modelo ΔE_core(descriptores)
  ajustado a ΔSCF → XPS de estructuras grandes.

## Orden

Grafeno prístino (todos los C iguales) → grafeno con vacante → grafeno con N
(C 1s y N 1s) → CNT → N/S/Se-CNT → FeSe/C → integración TB → modelo sustituto.
