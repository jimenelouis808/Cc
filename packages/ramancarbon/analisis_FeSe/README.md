# Rietveld de las dos muestras FeSe/C

Rigaku `.ras`, Cu Kα₁ (1.540593 Å), Bragg–Brentano, D/teX Ultra 250,
10–90° en pasos de 0.01°, 8001 puntos, 4.27 °/min (≈0.14 s por punto).

- **S1** — 900 °C: T-FeSe, H-FeSe, Fe₃C, α-Fe y carbono.
- **S2** — 1020 °C: carbono, Fe₃C, α-Fe y T-FeSe.

Refinados con **fondo de orden 12**, no con el 6 que trae el paquete por
defecto. Ver «El orden del fondo» más abajo: no es un ajuste fino, es la
diferencia entre un resultado y un sinsentido.

## Resultado

**S1 se cuantifica. S2 no.** Las dos frases están respaldadas por el
mismo barrido, en `tablas.md`.

En S1 las cinco fases son estables sobre los órdenes de fondo 8 a 20
(Rwp plano en 3.94–3.99, GOF ≈ 0.94) y ningún parámetro de anchura
termina en su techo. En S2, Fe₃C varía 22 puntos y el carbono 20 sobre
el mismo barrido, la `b` de la cementita acaba en su cota de red del
+5 % y sus anchuras en el techo: en S2 la «cementita» está haciendo de
joroba, no de fase. Se ve en la figura — el fondo de S2 lleva una
joroba justo debajo de la banda 002, porque el modelo se reparte esa
banda entre el carbono y el polinomio.

Las fracciones de S1, con la mitad de la dispersión del barrido como
incertidumbre (no el error propagado del refinamiento, que es diez veces
menor y está condicionado a que el modelo de fondo sea el correcto, que
es justo lo que se está poniendo a prueba):

| fase | % en peso | dominio (nm) | a (Å) | c (Å) |
|---|---:|---:|---:|---:|
| Fe₃C cementita | 49.4 ± 3.4 | 30.0 | 5.0807 | 4.5229 |
| C turbostrático | 28.9 ± 5.0 | 5.1 | 2.4066 | 6.8719 |
| α-Fe | 10.8 ± 1.0 | 16.6 | 2.8680 | — |
| H-FeSe | 7.6 ± 1.1 | 21.1 | 3.6168 | 5.8935 |
| T-FeSe | 3.3 ± 0.2 | 29.1 | 3.7825 | 5.4934 |

Son de la parte **cristalina e identificada**. Una fase sin modelar no
baja el total de 100 %: su intensidad se reparte entre las demás, y el
amorfo no aparece en absoluto.

## El carbono no es grafito

d₀₀₂ = **3.436 Å** contra los 3.354 Å del grafito 2H. Los cuatro
candidatos de la biblioteca convergen al mismo valor; al grafito le
cuesta deformar la celda **+2.33 %** para llegar y al turbostrático de
3.44 Å solo **−0.12 %**. Dominio de 5.1 nm en S1 y 4.7 nm en S2.

## El orden del fondo

Las muestras llevan hierro y están medidas con ánodo de cobre. El Cu Kα
está justo por encima del borde K del hierro, el hierro fluoresce y eso
cae en el detector: el fondo es el **96 %** del patrón calculado en S1 y
el **83 %** en S2. No deforma ni desplaza ningún pico, así que no se
arregla analizando mejor — hace falta monocromador secundario,
discriminación en energía en el detector (la D/teX Ultra puede) o ánodo
de cobalto.

Lo que sí hay que hacer al analizar es darle al polinomio orden
suficiente para seguir esa joroba. Con el orden 6 por defecto, S2
devuelve **99.1 % de carbono y 0.1 % de cementita con GOF 2.44**; con el
orden 10, 74.4 % y 25.1 % con GOF 1.22. El carbono se estaba comiendo el
fondo.

**Y esto invalida cualquier barrido de sensibilidad corrido sobre un
fondo inadecuado**, cosa que pasó dos veces durante este análisis: un
barrido del techo del perfil hecho con orden 6 parecía demostrar que la
fracción del carbono era una función monótona de una cota arbitraria
(59 % → 84 %), y repetido con orden 12 la degeneración desaparece — el
Rwp se mueve 3.953 → 3.947 en todo el barrido y la fracción se queda en
33 ± 3 %.

## Lo que queda sin explicar

- **S2, 2θ = 43.260°, d = 2.0897 Å.** Ninguna fase declarada lo produce.
  Probé austenita γ-Fe con carbono disuelto, que pondría ahí su 111, y
  **la medida la refuta**: añadirla baja el Rwp 0.04 puntos en S2 y 0.085
  en S1 —el paquete tiene calibrado que un efecto inexistente compra
  0.3 %— y su celda se va a a = 3.771 Å, su cota superior, absorbiendo
  otra cosa.
- **S2 no tiene pico donde debería estar la 110 del α-Fe** (44.67°). Con
  este fondo la ausencia de un pico débil no es prueba fuerte, pero es
  coherente con que la α-Fe de S2 salga al 3.7 ± 3.6 %.
- En S1, en cambio, los seis picos que el buscador deja sin explicar son
  **todos Fe₃C y T-FeSe**: fases que ya estaban declaradas, enterradas
  bajo la fluorescencia. No hay fase sorpresa en S1.

## Archivos

| archivo | qué es |
|---|---|
| `rietveld_FeSe.png` / `.pdf` | la figura: observado, calculado, fondo, ticks por fase, diferencia debajo en la misma escala |
| `tablas.md` | las cuatro tablas y los avisos de cada refinamiento |
| `S*_ajuste.csv` | las seis columnas del ajuste, para redibujar sin refinar |
| `S*_ticks.npz` | posiciones de reflexión por fase, con el cero aplicado |
| `S*_resumen.json` | factores R, fracciones, celdas, coordenadas, avisos |
| `*.py` | los guiones que lo produjeron |

## Si hace falta cuantificar S2

No con esta medida. Hace falta controlar el fondo de fluorescencia
—monocromador, discriminación en energía o ánodo de Co— y contar más
tiempo. S2 tiene cinco picos detectables sobre una banda de carbono que
domina el patrón; S1 tiene muchos picos nítidos de fases de hierro, y
esa es toda la diferencia entre las dos.
