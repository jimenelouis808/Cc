# Voltametría cíclica — cinco archivos, un electrodo

Cinco `.mpr` de EC-Lab sobre carbono vítreo de 3 mm con 20 µg de material
depositado, a 5, 10, 20 y 50 mV/s, en la ventana de −0.600 a +0.400 V.

El encargo era verificar que la capacitancia específica, el análisis de
Dunn y la clasificación de mecanismo estuvieran bien calculados. No lo
estaban: había cuatro fallos en el camino de la integral, y al quitarlos
quedó a la vista algo que los tapaba. Este documento va en ese orden —
primero lo que estaba mal, luego lo que la medida dice.

## Lo primero, antes que los números

**Los archivos no describen este experimento.** El bloque de comentarios
de los cinco `.mpr` dice `Fe(CN)63-/Fe(CN)64- 5.10-3 M`, `Au/Pd`,
`NaCl (0.2M)`, `SCE Saturated Calomel Electrode`. De ahí sólo se ha
tomado la referencia (ECS) para rotular el eje. Lo demás contradice lo
que me dijiste, y no es poca cosa: el material del electrodo de trabajo y
el electrolito cambian lo que significa cada número de aquí.

Los voltamperogramas no tienen el par de picos de un ferricianuro 5 mM
sobre carbono vítreo —que sería inconfundible: dos picos separados unos
70 mV en torno a +0.2 V— así que lo más probable es que EC-Lab arrastrara
el bloque de comentarios de un experimento anterior y nadie lo
actualizara. **Conviene que lo confirmes**, porque todos los F/g y los
mF/cm² salen de la masa (20 µg) y del área (0.0707 cm²) que me diste tú,
no del archivo.

## Los cuatro fallos

Están corregidos en `ramancarbon/echem/`, con prueba cada uno.

**1. El eje de tiempo se inventaba.** `Voltammogram` no guardaba el
tiempo, así que la carga se integraba contra `∫|dE|/ν`. Esa identidad
vale para un barrido ininterrumpido y para nada más: cualquier
subconjunto —una rama, un ciclo cuyo contador arrancó a mitad de
barrido— son dos tramos disjuntos, y el salto entre ellos es un potencial
que el electrodo nunca barrió. Medido en una rama anódica real: una
discontinuidad de 0.9998 V que salía como 39.99 s de barrido en vez de
20.00, y con ella la carga. Ahora se usa el reloj del instrumento, que
estos archivos traen, y cualquiera de los dos ejes se aplana en sus
empalmes.

**2. Un ciclo podía estar incompleto llegando a los dos extremos.** El
ciclo 10 de CV_01 abarca 1.0003 V de una ventana de 1.0003 V —pasa
cualquier prueba de vano— y recorre 1.65 V de los 2.00 que son una vuelta
entera, porque la adquisición se cortó a tres cuartos de la bajada. Se
informaba como si fuera un ciclo: 5.87 F/g contra los 7.17 del anterior.
Ahora se exige también el recorrido.

**3. El recorte de los vértices era un sesgo, no una corrección.** Se
recortaba un 2 % de la ventana en cada extremo, presuponiendo el
transitorio de conmutación del potenciostato. Estos archivos no lo
tienen: con 2000 puntos por ciclo y corriente promediada, la corriente
pasa lisa por el vértice y **baja** justo después. El recorte se llevaba
el 7.5 % de la capacitancia a 5 mV/s y el 2.9 % a 50 — y cuánto se lleva
depende de cómo reparta cada muestra su corriente en la ventana, o sea
que es un sesgo por muestra. Ahora se mide: se descarta lo que destaca
más de tres veces sobre su propio entorno junto a un vértice, que en
estos archivos es nada, y el informe lo dice.

**4. Los dos convenios de capacitancia discrepaban y nada lo explicaba.**
Discrepan porque el lazo no está equilibrado. Ver abajo.

## Lo que la medida dice

Las tablas están en [`tablas.md`](tablas.md); las figuras en
`cv_capacitancia.{png,pdf}` y `cv_cinetica.{png,pdf}` (600 dpi y vector).

### La eficiencia coulómbica es 0.48 a 5 mV/s

Un condensador devuelve lo que toma. Estas curvas devuelven, de la carga
de oxidación, el 48 % a 5 mV/s, el 69 % a 10, el 82 % a 20 y el 90 % a
50. **A la velocidad más lenta, más de la mitad de lo que la integral
llama capacitancia no se almacena: se gasta.**

### Lo que se gasta es 0.41 µA, constante

La carga neta de un ciclo, `ΔQ = Q_ox − Q_red = ∫i dt`, es por definición
lo que pasó y no volvió. Contra la duración del ciclo da una recta:

    ΔQ = 0.41 µA · t,  ordenada −8.9 µC,  R² = 0.996

sobre cargas de 10 a 150 µC. Una corriente parásita constante —corrosión,
oxidación del carbono, oxidación del electrolito cerca del límite
positivo— aporta más carga cuanto más dura el ciclo, y eso **imita una
dependencia con la velocidad de barrido**. Es el 54 % de Q_ox a 5 mV/s y
el 13 % a 50.

Con eso, la «retención del 51 % entre 5 y 50 mV/s» que da el convenio del
lazo no es una limitación de transporte. La capacitancia de la carga de
reducción, que la fuga anódica no contamina, va de 7.39 a 5.52 F/g, y de
esa diferencia la mayor parte es degradación (abajo), no cinética.

### El electrodo perdió un 16 % durante la sesión

CV_01 y CV_05 son los dos a 50 mV/s, el primero y el último experimento.
Dan 6.55 y 5.52 F/g. Entre medias hay 22 ciclos.

Esa repetición es lo único de la serie que separa cinética de
degradación, porque las velocidades se midieron en orden y por tanto el
tiempo y ν van juntos: sin ella, la pérdida entra en la dependencia con ν
y sale como un b, como un reparto de Dunn y como una retención, sin que
nada lo diga. **Hiciste bien en medirla.**

Dentro del propio CV_01, en cambio, la `C_red` apenas baja (6.61 → 6.55
F/g en diez ciclos) mientras la `C_lazo` cae de 7.82 a 7.17: en esos diez
ciclos lo que se va es la fuga, no lo almacenado.

### El b no prueba difusión, y la clasificación de mecanismo cambia

El b baja de 0.96 en −0.30 V a 0.46 en +0.30 V, y un b de 0.46 se lee
como «controlado por difusión, tipo batería, informar en mAh/g». Con esta
serie esa lectura no se sostiene.

Ajustando a las mismas cuatro velocidades tres leyes de dos parámetros
cada una —`a·ν^b`, la de Dunn `k₁ν + k₂√ν`, y `k_cap·ν + i_par`— las tres
dan R² entre 0.975 y 0.998 en todos los potenciales. **Ninguna gana**:
cuatro puntos y dos parámetros no distinguen una constante de un término
en √ν sobre una década de ν. Pero la constante no es una hipótesis — está
medida aparte, en el balance de carga, y la ley de fuga pide en cada
potencial una capacitancia que apenas varía en la ventana (128 µF en
−0.40 V, 95 µF en −0.20 V) y una corriente que sube monótona hacia el
límite positivo (0.30 → 2.30 µA): la magnitud correcta y la forma
correcta para una oxidación irreversible.

Con el b retirado como prueba, la clasificación pasa de **«tipo batería,
informar en mAh/g»** a **«pseudocapacitivo, F/g con reservas»**, que es
lo que las curvas parecen: jorobas anchas y poco separadas, sin picos de
difusión.

### El reparto de Dunn, aquí, no significa nada

Da 28 % capacitivo a 5 mV/s y 58 % a 50, pero el ajuste de dos términos
no describe los datos en 46 de 60 potenciales (R² < 0.95) y en 7 saca
coeficientes negativos, que ninguna de las dos contribuciones puede ser.
Está absorbiendo la corriente constante en el término de √ν. No lo cites
sin arreglar antes la fuga.

## Qué haría falta para una cifra publicable

1. **Quitar la fuga.** Estrechar la ventana por el lado positivo hasta
   donde la eficiencia coulómbica suba por encima de 0.95, y desoxigenar
   el electrolito. Una cronoamperometría de unos minutos en el límite
   positivo mide la corriente parásita directamente y dice cuánto se ha
   ganado.
2. **Ciclar hasta estabilizar** antes de empezar la serie: la `C_lazo`
   todavía cae un 2.5 % por ciclo a 10 mV/s y un 1.3 % a 20.
3. **Repetir la primera velocidad al final**, como ya hiciste. Con la
   fuga quitada y el electrodo estabilizado, esa repetición debería cerrar
   dentro del 2 %, y sólo entonces la dependencia con ν es cinética.
4. Con eso, **medir el b sobre al menos dos décadas** (1 a 100 mV/s) y
   volver a comparar las tres leyes. Si la de fuga deja de ganar, el b
   querrá decir lo que se supone que quiere decir.

Mientras tanto, la cifra defendible de estos datos es
**C_red = 6.8 ± 0.8 F g⁻¹** (media de las cuatro velocidades; la
dispersión es la degradación, no error de ajuste), equivalente a
1.9 ± 0.2 mF cm⁻² de electrodo geométrico, con la eficiencia coulómbica
declarada al lado.

## Los archivos

| archivo | qué es |
|---|---|
| `exportar.py` | genera todo lo demás a partir de los `.mpr` |
| `figuras.py` | las dos figuras, a partir de lo que exporta el anterior |
| `resumen.json` | todos los números, incluidos los metadatos de los `.mpr` |
| `ciclos.csv` | ciclo a ciclo: duración, recorrido, Q_ox, Q_red, los tres convenios |
| `CV_0N_ciclo.csv` | el ciclo informado de cada archivo: t, E, i, i/m |
| `CV_01_todos_los_ciclos.csv` | los diez ciclos del primer archivo |
| `dunn_50mVs.csv` | k₁, k₂, R² y las corrientes capacitivas de cada rama |
| `estudio_de_velocidad.txt` | la salida completa de `analyse_rate_study` y `dunn_analysis` |
| `tablas.md` | las cuatro tablas |

Para rehacerlo:

```bash
cd packages/ramancarbon
uv run python -I analisis_CV/exportar.py <carpeta con los .mpr>
uv run python analisis_CV/figuras.py
```
