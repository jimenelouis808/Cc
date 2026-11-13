# Un electrodo en H₂SO₄ 0.5 M: 21 archivos de EC-Lab

Cuatro experimentos sobre la misma celda, en este orden: seis CV a
distintas velocidades en una ventana ancha, seis CV en una ventana
estrecha para sacar el C_dl, cinco GCPL a distintas corrientes, y cuatro
impedancias intercaladas.

```bash
uv run python analisis_H2SO4/exportar.py <carpeta con los .mpr> [masa_mg] [area_cm2]
```

Por defecto, esta celda: **20 µg** de material activo sobre un **disco
de carbón vítreo de 3 mm** (0.0707 cm², la misma que usa
`analisis_GCD` para otra muestra). Ninguna de las dos está en los
archivos —EC-Lab tiene campo para las dos y se escribieron sin ellas—
así que viven en las constantes del script, que es el único sitio donde
un dato que no está en la medida puede vivir sin que haya que volver a
preguntarlo. Carga: **283 µg/cm²**.

## La celda

La cabecera de los `.mpr` nombra otra celda: arrastra una plantilla. Las
condiciones son las que dio el usuario y mandan sobre la cabecera.

| | |
|---|---|
| Electrolito | H₂SO₄ 0.5 M |
| Referencia | Ag/AgCl, KCl saturado (+0.197 V vs SHE) |
| pH | **0.3**, deducido de la molaridad |

El pH no se dio y no se ha inventado: sale de la tabla de electrolitos
del paquete, que lo calcula como lo calcularía uno a mano. El primer
protón del H₂SO₄ está disociado del todo y el segundo sólo en parte
(Ka₂ = 1.2×10⁻²), así que [H⁺] = 0.51 M y no 1.0 M; con el coeficiente
de actividad a esa fuerza iónica, pH ≈ 0.3. De ahí la escala RHE:

```
E(RHE) = E(Ag/AgCl sat.) + 0.197 + 0.0592 × 0.3 = E + 0.2147 V
```

## Los números, de una vez

Con 20 µg de material activo sobre 0.0707 cm² (283 µg/cm²):

| magnitud | valor | de dónde |
|---|---|---|
| **C (GCD)** | **81.6 F/g** = **23.1 mF/cm²** | descarga a 5 µA, caída IR descontada |
| C (CV) | 64.0 F/g = 18.1 mF/cm² | lazo ÷ 2νΔV a 5 mV/s |
| C (EIS) | 39.4 F/g = 11.1 mF/cm² | C′ a 0.0999 Hz |
| Energía | 11.1 Wh/kg | ∫V dq, no ½CV² |
| Potencia | 103 W/kg | a esa misma corriente |
| C_dl | 330 µF | ventana estrecha, 6 velocidades |
| ECSA | 8.2 cm² | con Cs = 40 µF/cm² (entre 5.5 y 16.5) |
| **Factor de rugosidad** | **117** | ECSA ÷ 0.0707 cm² geométricos (entre 78 y 233) |
| R_s | 85–106 Ω | cruce con el eje real, impedancia |

Los tres métodos difieren un **52 %**, y eso no es ruido: es que miden
en condiciones distintas. El número que entrega un dispositivo es el de
GCD; los otros dos son diagnósticos. Quien publique el mayor sin decir
cuál es está publicando una elección.

Por corriente, del GCD:

| | 5 µA | 10 µA | 20 µA | 40 µA | 60 µA |
|---|---|---|---|---|---|
| j (mA/cm²) | 0.071 | 0.141 | 0.283 | 0.566 | 0.849 |
| C (F/g) | 78.2 | 61.8 | 57.5 | 58.5 | 61.4 |
| C (mF/cm²) | 22.1 | 17.5 | 16.3 | 16.5 | 17.4 |

## Lo primero: dos de las tres ventanas parten el agua

Con esos 0.2147 V, las ventanas medidas caen así frente al RHE, y el
agua es estable sólo entre 0 y 1.229 V:

| experimento | vs Ag/AgCl sat. | vs RHE | margen |
|---|---|---|---|
| CV, serie ancha | −0.600 … +1.200 V | **−0.385 … +1.415 V** | −385 mV abajo, −186 mV arriba |
| CV, ECSA | −0.050 … +0.100 V | +0.165 … +0.315 V | holgada por los dos lados |
| GCD | −0.200 … +1.000 V | **+0.015 … +1.215 V** | +15 mV abajo, +14 mV arriba |

Eso explica casi todo lo demás:

- **La serie ancha de CV desprende hidrógeno en el límite inferior y
  oxígeno en el superior.** 1.8 V de ventana en medio acuoso no son una
  ventana de almacenamiento: son 571 mV de electrólisis repartidos entre
  los dos extremos. La capacitancia que sale de ella lleva dentro esa
  corriente, y por eso no se puede comparar con la del GCD aunque las
  dos estén en mF.
- **La del GCD está elegida justo al borde**, con 15 mV de margen a cada
  lado. Es la decisión correcta —1.2 V es lo máximo que da el agua— pero
  sin holgura: a 5 µA cada semiciclo dura unas tres horas sentado en ese
  límite, y ahí la fuga deja de ser un detalle (ver abajo).
- **La del ECSA está bien.** +0.165 a +0.315 V vs RHE no toca nada.

## La eficiencia culómbica delata la fuga, y escala como una fuga

Un exceso culómbico es carga que sale sin haber entrado. Por archivo:

| I | ciclos | eficiencia media |
|---|---|---|
| 60 µA | 101 | 102.3 % |
| 40 µA | 101 | 100.8 % |
| 20 µA | 101 | 105.8 % |
| 10 µA | 101 | 113.9 % |
| 5 µA | 101 | **142.4 %** |

El exceso crece al bajar la corriente, que es exactamente lo que hace
una corriente parásita CONSTANTE: la misma fuga pesa mucho sobre una
descarga de tres horas y poco sobre una de tres minutos. El módulo la
cifra en ≈ **+1.05 µA**. A 5 µA eso es el 21 % de la corriente aplicada,
y es la razón de que el punto de 5 µA quede fuera de la tendencia.

## Capacidad de velocidad: 78.5 %, con tres descuentos que hacer

`gcd.rate_capability` sobre los cinco archivos en el orden del horario:

| I (mA) | C (µF) | ± entre ciclos | caída IR | % de la excursión |
|---|---|---|---|---|
| 0.005 | 1563 | 13.5 % | 14.1 mV | 1.2 % |
| 0.010 | 1235 | 3.0 % | 27.5 mV | 2.3 % |
| 0.020 | 1150 | 0.7 % | 60.4 mV | 5.1 % |
| 0.040 | 1169 | 7.2 % | 144 mV | 12.1 % |
| 0.060 | 1228 | 0.5 % | 230 mV | 19.4 % |

**Retención 78.5 % a 12× la corriente.** Lo que hay que descontarle:

1. **14.5 puntos son la resistencia, no el electrodo.** Sin descontar la
   caída IR la retención saldría 64.0 %. Aquí está descontada de las
   cinco capacitancias, así que el 78.5 % es del material; el 64 % es lo
   que publicaría un análisis que no corrigiera.
2. **A 60 µA la IR se come el 19 % de la excursión.** Quedan 0.956 V
   frente a 1.19 V a la corriente más baja. El salto está fuera de las
   dos capacitancias, así que no sesga ninguna por sí mismo; lo que no
   son es la misma medida más rápido, porque este electrodo no da lo
   mismo sobre una ventana un 19 % más corta.
3. **El electrodo se está ACTIVANDO, no degradando.** Los cinco archivos
   ganan un 23.3 % compuesto de capacitancia dentro de sus propios
   ciclos (el de 5 µA, un 13.4 % en 101 ciclos). Una retención medida
   mientras el electrodo mejora no es sólo velocidad.

Y la curva **no es monótona**: baja de 1563 a 1150 µF y vuelve a subir a
1228. El ajuste `C = a·I^-m` da m = 0.090 ± 0.048 con **R² = 0.53**, que
es la forma honesta de decir que una ley de potencias no describe esto.
El mínimo en 20 µA es el punto donde la fuga ya no infla el número y la
caída IR todavía no acorta la ventana.

## La impedancia ve la misma activación

Tres espectros legibles, 39 frecuencias de 0.1 Hz a 200 kHz, 10 mV de
amplitud. El potencial de continua lo trae cada archivo y el lector lo
guarda (`potencial_cd_v`), así que cada uno se sitúa solo:

| archivo | E (V vs Ag/AgCl) | qué es | R_s | C′(0.1 Hz) | codo | Kramers-Kronig |
|---|---|---|---|---|---|---|
| `PEIS_incial` | +0.418 | antes de todo | 105.8 Ω | 0.468 mF | 0.183 Hz | consistente |
| `..._mult_01` | +0.275 | tras la CV, a OCV | 87.5 Ω | 0.787 mF | 0.378 Hz | consistente |
| `..._mult_02` | +0.500 | tras la CV, en el pico redox | 85.0 Ω | 0.797 mF | 0.316 Hz | **NO** |

La resistencia en serie baja de 106 a 85 Ω y la capacitancia entregable
sube un 68 %, de 0.47 a 0.79 mF, después de ciclar. Es la **misma
activación** que el GCD mide por dentro de sus archivos, por un camino
independiente.

Que el de +0.500 V no pase Kramers-Kronig no es un fallo de la medida:
está medido EN el pico redox, donde algo reacciona mientras el barrido
baja de 200 kHz a 0.1 Hz, y la celda del final no es la del principio.
Ahí no se debe ajustar un circuito.

Ninguno de los tres cierra el semicírculo —−Z″ sube monótona hasta
0.1 Hz y la fase se queda en −52° a −65°— así que **de estos datos no
sale ninguna R_ct**: la que devuelva un ajuste la pone el modelo. Y la
C′ del extremo es una cota INFERIOR, porque el barrido no llegó al
régimen capacitivo. Hace falta bajar la frecuencia.

## El par redox es lento de verdad, no por la caída óhmica

ΔEp crece de 90 mV a 5 mV/s hasta 572 mV a 200 mV/s, y E_pa se desplaza
de +0.456 a +0.674 V. Con la R_s que mide la impedancia (86 Ω) y el
i_p medido, la contribución óhmica a ΔEp es 2·i_p·R_s = 2.2 mV a
5 mV/s y 55 mV a 200. O sea que de los 482 mV que crece ΔEp, la
resistencia explica 53, un **11 %**. El resto es cinética: el par es
cuasi-reversible a 5–20 mV/s e irreversible a partir de 50.

## El C_dl y el ECSA

La serie estrecha da **C_dl = 330 µF** y, con Cs = 40 µF/cm²,
**ECSA = 8.2 cm²** (entre 5.5 y 16.5 según la Cs que se adopte: ese
factor de tres es irreducible y pasa entero al área).

Dos reservas, las dos del módulo: la recta de C_dl tiene R² = 0.96, y el
archivo de 20 mV/s se sale de la tendencia (0.695 mF donde tocaría
0.55) porque cae un 13.9 % por ciclo dentro de sí mismo. Sin ese punto
la recta mejora; quitarlo es una decisión y aquí no se ha tomado.

## Qué falta

- **La masa de material activo y el área geométrica.** Sin la masa no
  hay F/g, mAh/g ni Wh/kg, que es como se publica casi todo. Tiene que
  ser la masa de material ACTIVO: el aglomerante y el carbón conductor
  suelen ser el 20 %.
- **`GCPL__prueba_ventana_1-0_06_PEIS-FINAL.mpr` está vacío.** Su bloque
  de datos mide 1007 bytes, exactamente la cabecera, con cero registros.
  No es que el lector no lo entienda: no hay nada dentro. Hay que
  reexportarlo desde EC-Lab o volver a medirlo. Sin él no hay el "antes
  y después" completo que la serie de impedancias pedía.
- **Frecuencia más baja en los PEIS.** A 0.1 Hz la fase sigue en −65°,
  así que ninguna de las capacitancias de impedancia está cerrada.

## Lo que hay en la carpeta

### Figuras

| figura | paneles |
|---|---|
| `cv_ventana_amplia` | la serie de velocidades con la escala RHE y los límites del agua, C y eficiencia contra ν, el balance de carga, y la separación de Dunn |
| `cv_leyes` | k_cap, k_diff e i_par con sus errores, y las tres contribuciones apiladas |
| `cv_ecsa` | lo mismo para la ventana estrecha, de donde salen C_dl y el ECSA |
| `gcd` | las curvas, C contra corriente, retención ciclo a ciclo, eficiencia culómbica, la fuga, y la caída IR con la ESR |
| `gcd_articulo` | los tres que imprime la literatura: retención normalizada con sus barras de error, capacidad y eficiencia por ciclo, y el Ragone con su techo |
| `peis` | Nyquist a escala 1:1, el rincón de alta frecuencia, Bode con el codo, y C′(f)/C″(f) |
| `peis_modelo` | Nyquist en log-log, la DRT, el residuo de cada circuito y el de Kramers-Kronig |
| `capacitancia` | CV, GCD y EIS lado a lado, cada uno con su condición |

Cada una en `.png` y `.pdf` a 600 ppp, y los números de cada panel en
`datos_figuras/<figura>/`, un CSV por panel: lo que se exporta es lo que
está DIBUJADO, con las transformaciones ya aplicadas.

### Texto y tablas

| archivo | qué es |
|---|---|
| `informe.txt` | el informe completo: celda, mecanismo, CV, velocidad, Dunn, GCD, capacidad de velocidad, impedancia |
| `informe_ecsa.txt` | la serie estrecha, con C_dl, ECSA, Trasatti y las tres leyes de i(ν) |
| `informe_velocidad.txt` | sólo la capacidad de velocidad del GCD, con sus avisos |
| `informe_peis_*.txt` | un informe por espectro |
| `tablas.md` | las ventanas frente al RHE, la curva de velocidad, la impedancia y el ECSA |
| `cv_velocidades.csv` | capacitancia, ΔEp, picos y reversibilidad por velocidad |
| `gcd_velocidad.csv` | la curva de velocidad con la dispersión de cada punto |
| `gcd_ciclado.csv` | lo que los 101 ciclos de cada archivo le hacen |
| `peis_*.csv` | los tres espectros, f / Z′ / −Z″ |
| `INTERPRETACION.md` | **qué dice cada figura**, y qué no se puede concluir de ellas |
| `resumen.json` | todo lo anterior en una estructura |

Los `.mpr` no se versionan. Los CSV sí, para que las figuras se puedan
reproducir sin ellos.

## El circuito de impedancia

Los tres espectros se ajustan con **`R0-(R1|Q1)-Q2`**, no con lo que
elige `circuit="auto"`. El automático lee la forma de cada espectro y
propone a partir de ella, y sobre el inicial eso falla: su fase sólo
llega a −52° dentro del barrido, así que no detecta cola capacitiva, no
propone ningún circuito poroso, y zanja con un Randles al **17.25 %** de
residuo. El mismo `R0-(R1|Q1)-Q2` ajusta los tres entre **1.5 y 2.0 %
con todos los parámetros determinados**, y usar uno solo es lo que hace
comparables sus parámetros entre los tres estados. `tablas.md` guarda
también lo que elegía `auto`, para que la decisión se pueda discutir.

## Una cosa que las figuras enseñan y el texto no decía

Con el circuito común, los parámetros de los tres estados se pueden
poner en una fila y leer la activación directamente:

| | inicial | tras la CV, OCV | tras la CV, pico |
|---|---|---|---|
| R₀ en serie | 104.4 Ω | 87.0 Ω | 84.5 Ω |
| R₁ interfacial | 637.5 Ω | 297.8 Ω | 286.3 Ω |
| Q₂ almacenamiento | 5.95×10⁻⁴ | 8.44×10⁻⁴ | 8.99×10⁻⁴ |
| codo | 0.183 Hz | 0.378 Hz | 0.316 Hz |

Despues de ciclar: la serie baja un 19 %, la interfacial se parte por la
mitad y el almacenamiento sube la mitad. Es la misma activación que el
GCD mide dentro de sus archivos, por un camino independiente.
