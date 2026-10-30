# GCD: FeSe 900 °C en KOH 3 M

Masa activa 20 µg sobre un disco de carbón vítreo de 3 mm (0.070686 cm²), referencia Ag/AgCl_sat, ventana −0.6 a +0.4 V.

## Un punto por corriente

corriente | C (µF) | C (F/g) | C (µF/cm²) | R (Ω) | eficiencia (%) | fuga (µA) | asimétrica
--- | --- | --- | --- | --- | --- | --- | ---
5 µA | 77.22 | 3.861 | 1092 | 76.16 | 110.68 | 0.254 | 
10 µA | 55.99 | 2.8 | 792.2 | 80.69 | 104.48 | 0.219 | 
20 µA | 45.41 | 2.27 | 642.4 | 72.35 | 101.95 | 0.193 | 
60/−20 µA | 39.54 | 1.977 | 559.4 | 75.39 | 101.53 | 0.226 | sí
40 µA | 38.46 | 1.923 | 544.1 | 85.52 | 100.96 | 0.19 | 

## Lo que mil ciclos le hacen a cada uno

corriente | ciclos | C inicial (µF) | C final (µF) | retención (%) | pérdida (%/ciclo) | eficiencia media (%) | forma
--- | --- | --- | --- | --- | --- | --- | ---
5 µA | 1001 | 391.4 | 77.22 | 19.7 | 0.0527 | 109.65 | raiz
10 µA | 1001 | 59.08 | 55.99 | 94.8 | 0.0053 | 105.13 | exponencial
20 µA | 1001 | 46.03 | 45.41 | 98.6 | 0.0013 | 102.06 | exponencial
60/−20 µA | 1001 | 40.99 | 39.54 | 96.5 | 0.0035 | 101.57 | raiz
40 µA | 1001 | 38.54 | 38.46 | 99.8 | 0.0002 | 100.98 | exponencial

## El archivo asimétrico, comparado donde se puede

curva | corriente de descarga (mA) | medido (F/g) | interpolado de los simétricos (F/g) | diferencia
--- | --- | --- | --- | ---
60/−20 µA | 0.02 | 1.977 | 2.27 | -12.9 %
