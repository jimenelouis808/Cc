# Voltametría cíclica — S1 (900 °C) en KOH 0.5 M

Cinco `.mpr` de EC-Lab sobre la muestra **S1**, el FeSe/C tratado a
900 °C que ya analizamos aquí por Raman y por difracción (α-Fe, Fe₃C,
T-FeSe, H-FeSe y carbono turbostrático).

| | |
|---|---|
| electrodo de trabajo | carbono vítreo, 3 mm de diámetro (0.0707 cm²) |
| material activo | 20 µg |
| electrolito | KOH 0.5 M |
| referencia | Ag/AgCl, KCl saturado |
| contraelectrodo | alambre de Pt |
| ventana | −0.600 a +0.400 V vs Ag/AgCl |
| velocidades | 5, 10, 20 y 50 mV/s, más una repetición a 50 |

Todo eso lo dijiste tú y nada sale de los archivos: **el bloque de
comentarios de un `.mpr` no se lee**, ni aquí ni en el paquete. EC-Lab lo
arrastra de un experimento al siguiente, nadie lo edita, y en estos
describe una medida que no es ésta.

## Lo primero: dónde está esta ventana

Con Ag/AgCl saturado (+0.197 V vs SHE) en KOH 0.5 M (pH 13.56, contando
el coeficiente de actividad y no pOH = 0.30):

    E(RHE) = E + 0.197 + 0.05916 × 13.56 = E + 0.999 V

así que la ventana medida es **+0.399 a +1.399 V vs RHE**, y el límite
positivo queda **169 mV por encima del equilibrio del oxígeno**
(1.23 V vs RHE). Sobre un material que lleva α-Fe, Fe₃C y FeSe en
álcali —o sea, un catalizador de OER— eso no es una ventana libre de
faradaico: es una medida de OER a sobrepotencial bajo con un
voltamperograma alrededor.

Por el otro lado, +0.399 V vs RHE está dentro de la región de reducción
de oxígeno. Si el electrolito no se purgó con N₂ o Ar, la corriente
catódica del límite negativo lleva ORR dentro.

Eso explica, por sí solo, todo lo que sigue.

## Lo que la medida dice

Tablas en [`tablas.md`](tablas.md); figuras en `cv_capacitancia.{png,pdf}`
y `cv_cinetica.{png,pdf}` (600 dpi y vector).

### La eficiencia coulómbica baja a 0.48

De la carga de oxidación vuelve el 48 % a 5 mV/s, el 69 % a 10, el 82 % a
20 y el 90 % a 50. **A la velocidad más lenta, más de la mitad de lo que
la integral llama capacitancia no se almacena: se gasta.** Cuanto más
dura el ciclo, más tiempo pasa el electrodo por encima de 1.23 V.

### Lo que se gasta son 0.41 µA, y es oxígeno

La carga neta de un ciclo, `ΔQ = Q_ox − Q_red = ∫i dt`, es por definición
lo que pasó y no volvió. Contra la duración del ciclo da una recta:

    ΔQ = 0.41 µA · t,   ordenada −8.9 µC,   R² = 0.996

sobre cargas de 10 a 150 µC. Son 5.8 µA cm⁻² promediados sobre el ciclo;
en el límite positivo la ley `i = k_cap·ν + i_par` pide 2.3 µA, o sea
**33 µA cm⁻² a η = 169 mV**, que es exactamente lo que da un catalizador
de OER de hierro decente a ese sobrepotencial. Es el 54 % de Q_ox a
5 mV/s y el 13 % a 50.

Con eso, la «retención del 51 % entre 5 y 50 mV/s» del convenio del lazo
no es una limitación de transporte. La capacitancia de la carga de
reducción, que esa corriente anódica no contamina, va de 7.39 a
5.52 F/g.

### El electrodo perdió un 16 % durante la sesión

CV_01 y CV_05 son los dos a 50 mV/s, el primer experimento y el último, y
dan 6.55 y 5.52 F/g de `C_red` con 22 ciclos entre medias. Esa repetición
es lo único de la serie que separa cinética de degradación, porque las
velocidades se midieron en orden y por tanto el tiempo y ν van juntos.
**Hiciste bien en medirla.**

Dentro del propio CV_01, en cambio, la `C_red` apenas baja (6.61 → 6.55
F/g en diez ciclos) mientras la `C_lazo` cae de 7.82 a 7.17: en esos diez
ciclos lo que se va es la fuga, no lo almacenado. Un Ag/AgCl en KOH
además deriva —el cloruro se fuga y el potencial se mueve—, así que parte
de ese 16 % puede ser la referencia y no la muestra; en alcalino lo que
no deriva es un Hg/HgO.

## ¿Se pueden sacar el b y Dunn de estas medidas?

**No.** Y se puede decir con números por qué, que es mejor que decirlo a
secas.

El b baja de 0.96 en 0.70 V vs RHE a 0.46 en 1.30 V, y 0.46 se lee como
«controlado por difusión, tipo batería». Ajustando a las mismas cuatro
velocidades tres leyes de dos parámetros cada una —`a·ν^b`, la de Dunn
`k₁ν + k₂√ν`, y `k_cap·ν + i_par`— las tres dan R² entre 0.975 y 0.998 en
todos los potenciales. **Ninguna gana**: sobre una década de ν, una
corriente constante y un término en √ν son casi la misma función, y con
cuatro puntos y dos parámetros no hay con qué elegir.

La prueba que lo zanja es quitar la fuga MEDIDA —la del balance de carga,
que no mira cómo escala i con ν, así que la corrección no es circular— y
ver si el b vuelve a 1:

| E / V vs RHE | b medido | b sin los 0.41 µA |
|---|---|---|
| 0.50 | 0.892 | 1.144 |
| 0.72 | 0.876 | 1.140 |
| 0.90 | 0.783 | 1.020 |
| 1.07 | 0.690 | 0.858 |
| 1.25 | 0.526 | 0.600 |

Una sola constante **sobrecorrige** en el extremo negativo —b pasa de 1,
que es imposible— y **se queda corta** en el positivo, donde la fuga vale
cinco veces su propia media. Los dos errores son mayores que el efecto
que se quiere medir. Esa es la respuesta: la serie no puede distinguir
difusión de una reacción parásita, no porque el ajuste sea malo sino
porque las dos cosas producen la misma dependencia con ν sobre una
década.

Lo mismo vale para Dunn. El reparto da 26 % capacitivo a 5 mV/s y 53 % a
50, y con la fuga quitada pasa a 42 % y 63 %: se mueve más de lo que
mide. Además el término en √ν se está tragando la constante, que es la
razón de que la separación no se sostenga en los potenciales marcados en
la figura.

Y tienes razón en lo otro: **tampoco es el experimento para probarlo**.
No hay picos redox definidos —sólo jorobas anchas y poco separadas— así
que no hay un proceso difusivo que separar aunque la medida pudiera
hacerlo. Dunn tiene algo que decir sobre un material con picos; sobre
éste, lo único que separaría sería la fuga.

### La figura del Dunn

Tenías razón en que se veía mal, y era un fallo del paquete, no del
dibujo: los coeficientes se ajustaban sin restricción, salían negativos
en 7 potenciales, y una k₁ negativa dibuja la curva capacitiva al otro
lado del lazo. Ahora el ajuste es no negativo —que es lo que el modelo
dice: ninguna de las dos contribuciones puede restar— y la banda se
recorta al lazo medido, marcando con una cruz los puntos donde la
separación no se sostiene. En esta curva son 6 de 60, casi todos donde la
corriente de la rama cambia de signo entre una velocidad y otra, que es
donde la separación no está definida en absoluto.

## Qué haría falta para una cifra publicable

1. **Bajar el límite positivo.** A −0.2 V vs Ag/AgCl (≈ 0.80 V vs RHE)
   no hay OER posible. La ventana clásica de supercondensador para un
   carbono en álcali es −1.0 a −0.2 V vs Ag/AgCl (0.0 a 0.80 V vs RHE),
   que además es más ancha que la actual.
2. **Purgar con N₂ o Ar** y mantener manta de gas: quita la ORR del
   límite negativo, que es lo que hace que la `C_red` también crezca un
   poco con la duración del ciclo.
3. **Cambiar a Hg/HgO**, que es la referencia que no deriva en KOH.
4. **Una cronoamperometría de unos minutos en el límite positivo**: mide
   la corriente parásita directamente, sin suponer nada sobre `i(ν)`, y
   entonces la corrección del b deja de ser circular.
5. **Ciclar hasta estabilizar** antes de la serie (la `C_lazo` todavía
   cae un 2.5 % por ciclo a 10 mV/s) y **repetir la primera velocidad al
   final**, como ya hiciste. Con todo lo anterior esa repetición debería
   cerrar dentro del 2 %.
6. Con eso, **el b sobre al menos dos décadas** (1 a 100 mV/s).

Mientras tanto, la cifra defendible es la de las medidas menos
contaminadas, las de 50 mV/s, donde la fuga es el 13 % de Q_ox:

> **C = 5.5 a 6.6 F g⁻¹** (1.6 a 1.9 mF cm⁻² de electrodo geométrico),
> de la carga de reducción, con Q_red/Q_ox = 0.90 declarada al lado. El
> intervalo es la degradación entre el primer y el último experimento,
> no error de medida.

## En el programa

- La masa y el área se escriben en la tarjeta **Electrodo** de la barra
  lateral y se aplican al cargar; la barra de estado dice ahora con qué
  masa y qué área ha quedado el archivo, para que una masa en blanco se
  vea al cargar y no tres pestañas después.
- **Ciclo a analizar** es un desplegable nuevo al lado de la velocidad:
  por defecto el último completo, y la lista marca los incompletos. La
  elección se guarda con el proyecto.

## Los archivos

| archivo | qué es |
|---|---|
| `exportar.py` | genera todo lo demás a partir de los `.mpr` |
| `figuras.py` | las dos figuras |
| `tablas.py` | las cinco tablas |
| `resumen.json` | todos los números, condiciones incluidas |
| `ciclos.csv` | ciclo a ciclo: duración, recorrido, Q_ox, Q_red, los tres convenios |
| `CV_0N_ciclo.csv` | el ciclo informado de cada archivo: t, E, i, i/m |
| `CV_01_todos_los_ciclos.csv` | los diez ciclos del primer archivo |
| `dunn_50mVs.csv` | k₁, k₂, R², las corrientes capacitivas de cada rama y dónde no está definida |
| `estudio_de_velocidad.txt` | la salida completa de `analyse_rate_study` y `dunn_analysis` |

```bash
cd packages/ramancarbon
uv run python -I analisis_CV/exportar.py <carpeta con los .mpr>
uv run python    analisis_CV/figuras.py
uv run python    analisis_CV/tablas.py > analisis_CV/tablas.md
```
