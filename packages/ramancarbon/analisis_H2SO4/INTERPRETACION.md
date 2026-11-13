# Qué dicen estas figuras

Electrodo de 20 µg en H₂SO₄ 0.5 M, referencia Ag/AgCl KCl sat., pH 0.3
deducido de la molaridad. La escala RHE está 0.2147 V por encima de la
medida, y el agua es estable entre 0 y 1.229 V vs RHE. Casi todo lo que
sigue cuelga de ese par de frases.

---

## Primero, tus cuatro objeciones

### 1. «La línea del ciclo de CV se ve quebrada»

Era un defecto, y de los buenos. La curva se dibujaba interpolada sobre
la **rejilla de Dunn** —60 potenciales sobre 1.8 V, un paso de 30 mV—
mientras los datos tienen 13 621 puntos. Un voltamperograma dibujado con
60 puntos es un polígono, y eso era lo que se veía.

Al dibujarlo a resolución completa apareció algo peor, que llevaba ahí
todo el tiempo: `sweeps()` devuelve las ramas anódicas de **los cuatro
ciclos** del archivo y las catódicas de los cuatro. Dibujadas en orden
eso son cuatro lazos unidos por los saltos entre ellos; interpoladas
sobre la rejilla, un promedio de cuatro ciclos de los cuales **el primero
es formación**. Ahora se usa el ciclo informado, el mismo que analiza el
resto del módulo, y la rejilla se subió a 240 puntos.

### 2. «El sombreado sale de los límites de la curva»

Esto es mitad defecto, mitad resultado, y el resultado es interesante.

Lo que quedaba de desbordamiento tras arreglar lo anterior está marcado
ahora en gris: **34 de los 240 potenciales**. Son los potenciales donde
la corriente de la rama **cambia de signo entre una velocidad y otra**.
Ahí `k₁ν` es una fracción de una corriente que no tiene signo, así que
la separación no está definida y la banda puede salirse del lazo por esa
razón y no porque el ajuste sea malo.

Y mira dónde caen: **en los dos extremos de la ventana**, −0.50 V y
+1.12 V. Que son, exactamente, los dos extremos donde el electrodo está
partiendo agua. La separación de Dunn se rompe justo donde deja de haber
sólo almacenamiento.

### 3. «Si la azul es la capacitiva, debería ser plana y no tener picos»

Aquí la intuición es correcta **para un condensador de doble capa** y no
se aplica a este material. La diferencia está en qué significa `k₁`:

```
i(V, ν) = k₁(V)·ν  +  k₂(V)·√ν
          └ superficie ┘ └ difusión ┘
```

`k₁` **se ajusta en cada potencial por separado**. El término `k₁ν` no es
«la doble capa»: es *toda* la corriente que escala linealmente con la
velocidad. Y hay dos cosas que lo hacen:

- la **doble capa**, que sí da una caja plana, y
- la **redox confinada en la superficie** — la pseudocapacitancia — que
  tiene pico, porque es una reacción con su potencial.

Un pico en la banda azul **es la firma de la pseudocapacitancia**, no un
error. Lo que lo confirma, de tres maneras independientes:

| prueba | valor | qué dice |
|---|---|---|
| b en el pico (+0.300 V) | **0.811 ± 0.022**, R² 0.997 | la corriente del pico va con ν, no con √ν |
| ΔEp a 5 mV/s | **90 mV** | pico ancho y poco separado, no de batería |
| mecanismo | **pseudocapacitivo, confianza alta** | hay picos redox Y b ≈ 1 |

Si fuera difusión (intercalación, especie en disolución) el pico iría con
√ν y b saldría 0.5. Da 0.81.

El corolario incómodo, que el propio módulo tiene escrito: **ni b ni Dunn
distinguen doble capa de pseudocapacitivo**, porque los dos dan b ≈ 1 y
los dos caen en `k₁`. Lo que los separa es si hay picos redox. Aquí los
hay.

### 4. «Difusiva y surface, ¿no es lo mismo?»

No: son las dos mitades complementarias. «Superficial» y «capacitivo» sí
son lo mismo aquí, y ése era el problema — la etiqueta decía *surface*
donde tú esperabas *capacitivo*. Ya dice las dos cosas:

- `capacitive, surface-controlled (k₁ν)` — azul
- `diffusion-controlled (k₂√ν)` — rojo

### 5. «El PEIS inicial tiene el ajuste mal»

Correcto, y era el peor de los tres. `circuit="auto"` lee la forma de
cada espectro y propone a partir de ella; el espectro inicial sólo llega
a **−52° de fase** dentro del barrido, así que no se detecta cola
capacitiva, no se propone ningún circuito poroso, y se zanjaba con un
Randles al **17.25 %** de residuo.

Probando el catálogo entero sobre los tres:

| circuito | inicial | OCV | pico | parámetros indeterminados |
|---|---|---|---|---|
| **`R0-(R1\|Q1)-Q2`** | **1.98 %** | **1.86 %** | **1.54 %** | **0, 0, 0** |
| `L0-R0-(R1\|Q1)-T1` | 1.97 % | 1.82 % | 1.49 % | 2, 2, 0 |
| `R0-T1` | 16.87 % | 11.04 % | 10.78 % | 1 |
| `R0-(R1\|Q1)` (el de auto) | 17.25 % | 14.40 % | 13.67 % | 1 |

`R0-(R1|Q1)-Q2` —resistencia en serie, un arco interfacial, y un CPE
bloqueante que es el almacenamiento— ajusta los tres entre el 1.5 y el
2.0 % **con todos los parámetros determinados**. El de ocho parámetros
no mejora el residuo y deja dos indeterminados.

Y hay una segunda razón para usar el mismo en los tres: **sólo así se
comparan sus parámetros**, que es la razón de haber medido tres estados
de la misma celda. La tabla guarda también lo que elegía `auto`, para
que la decisión se pueda discutir.

---

## Figura por figura

### `cv_ventana_amplia`

**(a) La serie.** Lo primero que hay que mirar son las dos rayas
verticales: son los límites del agua pasados a esta escala. La ventana
los cruza por los dos lados. Todo lo que hay a la izquierda de la raya
azul es desprendimiento de hidrógeno y todo lo que hay a la derecha de la
roja es oxígeno, y en los dos sitios la corriente sube bruscamente —
mírala en 200 mV/s, donde el lazo se dispara en los dos extremos. **Eso
no es almacenamiento y está dentro de todas las capacitancias de esta
serie.**

El pico anódico se desplaza de +0.456 a +0.674 V entre 5 y 200 mV/s. De
los 482 mV que crece ΔEp, la resistencia de la celda explica 53 (11 %,
con los 86 Ω que mide la impedancia): el par es **genuinamente lento**,
cuasi-reversible hasta 20 mV/s e irreversible a partir de 50.

**(b) Capacitancia contra velocidad.** 71 → 45 F/g de 5 a 200 mV/s,
retención 63 %. La eficiencia de carga (rojo, eje derecho) está entre
1.02 y 1.04: sale **más carga de la que entra**, en todas las
velocidades. Es la fuga, y sube al bajar la velocidad porque una fuga es
una corriente constante.

**(c) El balance de carga.** Aquí está medida: ⟨i_par⟩ = **−0.35 µA** con
R² = 0.98 sobre seis velocidades. Que el ajuste sea tan bueno es la
prueba de que es una fuga constante y no ruido.

**(d) Dunn.** 61.8 % capacitivo a 5 mV/s. Sube a 91.2 % a 200 porque el
término difusivo va con √ν y pierde peso cuando ν sube — eso es
aritmética, no una mejora del electrodo. El pico de la banda azul en
+0.45 V es la pseudocapacitancia (ver arriba). Las franjas grises son
donde la separación no está definida.

### `cv_leyes`

Las tres contribuciones a la vez, `k_cap·ν + k_diff·√ν + i_par`, que es
el modelo honesto: los de dos términos te obligan a elegir dos de las
tres. Con 6 velocidades el término difusivo queda determinado en 6 de 9
potenciales. Donde los errores son enormes (los potenciales negativos),
la serie no resuelve el reparto y ahí no hay que citar nada.

### `cv_ecsa`

La ventana estrecha, +0.165 a +0.315 V vs RHE, que **sí** está holgada
dentro del agua. De aquí salen **C_dl = 330 µF** y **ECSA = 8.2 cm²**
(entre 5.5 y 16.5 según la Cs que se adopte: ese factor de tres es
irreducible y sólo sirve para comparar entre tus muestras).

Dos reservas: la recta de C_dl tiene R² = 0.96, y el archivo de 20 mV/s
se sale de la tendencia (0.695 mF donde tocaría 0.55) porque cae un
13.9 % por ciclo dentro de sí mismo. Sin ese punto la recta mejora;
quitarlo es una decisión y no la he tomado.

### `gcd`

**(a) Las curvas.** La descarga no es una recta: tiene **meseta**
(R² lineal = 0.967). El módulo lo marca como evidencia *en contra* del
pseudocapacitivo, y tiene razón en marcarlo. Mi lectura es que parte de
esa meseta es la ventana: el límite inferior está a +0.015 V vs RHE y
ahí la evolución de hidrógeno sujeta el potencial mientras sigue saliendo
carga. No es una conclusión, es la hipótesis que explica la meseta y el
142 % de eficiencia a la vez.

**(b) C contra corriente.** 78 → 57 → 61 F/g: **no es monótona**. El
mínimo en 20 µA es donde la fuga ya no infla el número y la caída IR
todavía no acorta la ventana. Por eso `C = a·I^-m` sale con R² = 0.53:
una ley de potencias no describe esto.

**(c) Retención ciclo a ciclo.** Las cinco curvas **suben**. El electrodo
se activa: +13.3 % en 101 ciclos a 5 µA. Mientras eso pasa, ninguna
retención frente a la corriente es sólo velocidad.

**(d) Eficiencia culómbica.** 102 → 142 % al bajar la corriente. La misma
fuga que ve la CV, vista por el otro lado.

**(e) La fuga.** Esta es la comprobación: la corriente parásita sale
**0.44–1.05 µA** en todo el intervalo, contra una media de 0.70. Si fuera
constante del todo, la línea sería plana. Lo es a un factor dos sobre un
factor doce de corriente aplicada, lo que la confirma como fuga y no como
un artefacto de la corriente.

**(f) Caída IR y ESR.** El salto crece de 14 a 230 mV con la corriente
—lo que tiene que hacer— y la ESR que implica sube de 1.4 a 1.9 kΩ. Que
**no** sea plana es la señal de que ese salto no es sólo óhmico: lleva
dentro algo que depende de la corriente. Y los 1.4 kΩ no cuadran con los
85 Ω de la impedancia: la diferencia es resistencia que a alta frecuencia
se cortocircuita (contacto, el propio material).

### `gcd_articulo`

**(a)** La retención normalizada con **barras de error**, que son lo que
hacen los propios ciclos de cada archivo. La barra del punto de 5 µA es
±13.5 %: ese punto no es un punto. **(b)** Capacidad y eficiencia por
ciclo en la misma caja. **(c)** El Ragone: 11.1 Wh/kg a 103 W/kg,
cayendo a ~7 Wh/kg a 1.4 kW/kg.

### `peis` y `peis_modelo`

**Ninguno de los tres cierra el semicírculo**: −Z″ sube monótona hasta
0.1 Hz. De estos datos **no sale ninguna R_ct medida** — la que devuelve
un ajuste la pone el modelo. Y la C′ del extremo es una **cota inferior**,
porque el barrido no llegó al régimen capacitivo.

Lo que sí se mide, y es el resultado principal:

| | inicial | tras la CV, OCV | tras la CV, pico |
|---|---|---|---|
| E (V vs Ag/AgCl) | +0.418 | +0.275 | +0.500 |
| R₀ (serie) | 104.4 Ω | 87.0 Ω | **84.5 Ω** |
| R₁ (interfacial) | 637.5 Ω | 297.8 Ω | **286.3 Ω** |
| Q₂ (almacenamiento) | 5.95×10⁻⁴ | 8.44×10⁻⁴ | **8.99×10⁻⁴** |
| n₂ | 0.713 | 0.809 | 0.780 |
| C′(0.1 Hz) | 0.468 mF | 0.787 mF | 0.797 mF |
| codo | 0.183 Hz | 0.378 Hz | 0.316 Hz |

Después de ciclar: la resistencia en serie baja un **19 %**, la
interfacial se **parte por la mitad** (−55 %), el elemento de
almacenamiento sube un **51 %** y el codo se desplaza a frecuencia más
alta, o sea que el electrodo responde **el doble de rápido**. Eso es la
misma activación que el GCD mide por dentro de sus archivos, por un
camino completamente independiente y con un modelo que ajusta al 2 %.

**El de +0.500 V no pasa Kramers-Kronig** y eso no es un fallo de la
medida: está medido *en* el pico redox, donde algo reacciona mientras el
barrido tarda minutos en bajar de 200 kHz a 0.1 Hz, así que la celda del
final no es la del principio. De ése no se deben citar parámetros de
circuito aunque sea el que mejor ajusta.

La DRT (b) enseña varios picos por espectro, y **no hay que leerlos como
procesos**: el aviso del propio módulo dice que la fase pasa 0.5–0.7
décadas cerca de −45°, y un elemento distribuido no es una suma de arcos.

### `capacitancia`

La misma magnitud por tres métodos: **81.6 (GCD), 64.0 (CV), 39.4 (EIS)
F/g**, con un **52 %** de dispersión. No es ruido y el orden tiene
explicación: el de EIS se mide con 10 mV alrededor de un punto fijo y no
ha llegado al régimen capacitivo, así que es cota inferior; el de CV
lleva dentro la electrólisis de los dos extremos; el de GCD es el que
entrega el dispositivo. **Ése es el que se publica, y hay que decir a qué
corriente.**

---

## Lo que yo concluiría

1. **Es un pseudocondensador**, con confianza alta: picos redox anchos
   (ΔEp 90 mV), b = 0.81 en el pico, 62 % capacitivo por Dunn.
2. **81.6 F/g a 5 µA** (= 0.25 A/g), 11.1 Wh/kg, 103 W/kg. Con las
   reservas de abajo.
3. **El electrodo se activa durante toda la medida**, y eso lo ven dos
   técnicas independientes: +23 % compuesto dentro de los archivos de
   GCD, y −55 % de resistencia interfacial con +51 % de almacenamiento
   en la impedancia.
4. **Hay una fuga constante de ~0.5–1 µA** medida por tres caminos
   (balance de carga de la CV, exceso culómbico del GCD, y su escala con
   la corriente), y a 5 µA es el 21 % de la corriente aplicada.

## Lo que no se puede concluir, y qué haría falta

- **Nada cuantitativo de la serie ancha de CV.** Esa ventana parte el
  agua por los dos extremos. Repítela entre −0.20 y +1.00 V, que es la
  del GCD, y entonces sus F/g se podrán comparar con los del GCD.
- **Ninguna R_ct.** Ningún espectro cierra el arco. Harían falta
  frecuencias más bajas (hasta 10 mHz) para cerrar la cola capacitiva y
  que C′ deje de ser una cota.
- **La retención de velocidad, mientras el electrodo se activa.** Mide
  una serie de ida y vuelta: 5 → 60 → 5 µA. Si el 5 µA final devuelve el
  valor del inicial, lo medido fue velocidad; si no, fue activación.
- **El ECSA, contra la literatura.** El factor tres de la Cs es
  irreducible. Sirve para comparar entre tus muestras y para nada más.
- **La meseta del GCD.** Que sea la ventana o que sea el material se
  decide con la misma repetición en ventana estrecha.
- **El PEIS final** sigue vacío y hace falta para cerrar el «antes y
  después».
