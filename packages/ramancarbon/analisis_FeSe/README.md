# Rietveld de las dos muestras FeSe/C — ESTADO PROVISIONAL

Difractogramas de Rigaku (`.ras`), Cu Kα, Bragg-Brentano, D/teX Ultra 250,
10–90° en pasos de 0.01°, 4.27 °/min (≈0.14 s por punto).

- **S1** — 900 °C. Declaradas: T-FeSe, H-FeSe, Fe₃C, α-Fe y carbono.
- **S2** — 1020 °C. Declaradas: carbono, Fe₃C, α-Fe y T-FeSe.

## Lo que hay aquí

| archivo | qué es |
|---|---|
| `rietveld_FeSe.png` / `.pdf` | la figura: observado, calculado, fondo, ticks por fase y diferencia debajo en la misma escala |
| `S*_ajuste.csv` | las seis columnas del ajuste, para redibujar sin volver a refinar |
| `S*_ticks.npz` | posiciones de reflexión permitidas por fase, con el cero aplicado |
| `S*_resumen.json` | factores R, fracciones, celdas, coordenadas refinadas, avisos |
| `*.py` | los guiones que produjeron todo, para rehacerlo |

## Por qué dice PROVISIONAL

**Las fracciones en peso de estos dos refinamientos NO son citables**, y la
razón está medida, no supuesta.

1. **El orden del polinomio de fondo por defecto (6) es inadecuado para
   estas muestras.** En S2, con orden 6 el refinamiento dice 99.1 % de
   carbono y 0.1 % de Fe₃C con GOF 2.44; con orden 10 dice 74.4 % y
   25.1 % con GOF 1.22. El polinomio de orden 6 no puede seguir la
   curvatura del fondo, así que la fase de carbono se lo come y con ella
   desaparecen las fases de hierro. La figura de S2 que hay aquí está
   hecha con el orden por defecto y se ve: la curva diferencia tiene un
   hueco sistemático entre 18° y 35°.
2. **El fondo es el 97 % (S1) y el 87 % (S2) del patrón calculado.** Las
   fases explican el resto, y de ese resto salen todas las fracciones.
3. **La causa del fondo es instrumental.** Hierro medido con ánodo de
   cobre: el Cu Kα está justo por encima del borde K del hierro, el
   hierro fluoresce y eso cae en el detector. No deforma ni desplaza
   ningún pico, así que no se arregla analizando mejor: hace falta
   monocromador secundario, discriminación en energía en el detector
   (la D/teX Ultra puede), o ánodo de cobalto.
4. **La anchura del carbono choca con el techo del modelo.** La banda 002
   de S2 mide 2.26° de FWHM y el perfil de Caglioti del paquete no pasa
   de 1.53°, así que la calculada sale estrecha y alta. Y subir el techo
   no lo arregla: barrido en seis valores entre 2 y 25, W del carbono
   acaba igual al techo en los seis y su fracción sube monótonamente del
   59 % al 84 %. Es degenerado con el fondo.

## Lo que SÍ es resultado

- **Identificación de fases**, que va por posiciones y no depende del fondo.
- **El carbono no es grafito 2H**: d₀₀₂ = 3.435 Å contra los 3.354 Å del
  grafito. Los cuatro candidatos de la biblioteca convergen al mismo
  valor; al grafito le cuesta deformar la celda +2.33 % para llegar y al
  turbostrático de 3.44 Å solo −0.12 %.
- **Parámetros de red y coordenadas refinadas** (ver `S*_resumen.json`).
- De las fracciones, solo las que el barrido marca estables: en S1 α-Fe
  (varía 7.7 puntos) y T-FeSe (2.4); en S2 α-Fe (1.4) y T-FeSe (0.5).
  Fe₃C y el carbono varían entre 25 y 35 puntos en las dos muestras.

## Qué falta

Queda corriendo un barrido del orden del fondo en órdenes 8, 10, 12, 14,
16 y 20 sobre las dos muestras, para dar los números finales en el orden
donde dejen de moverse en vez de en el que trae el paquete por defecto.
