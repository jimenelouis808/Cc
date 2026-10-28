# Voltametría cíclica — tablas

Electrodo: carbono vítreo de 3 mm (A = 0.07069 cm²), 20 µg de material activo.
Ventana 1.000 V, de −0.600 a +0.400 V frente a la referencia que declara el
archivo (ECS). Se informa, de cada archivo, el ÚLTIMO ciclo completo.

## Tabla 1 — Capacitancia por archivo

| archivo | orden | ν / mV s⁻¹ | ciclos | ciclo informado | Q_red/Q_ox | C_lazo / F g⁻¹ | C_rama / F g⁻¹ | **C_red / F g⁻¹** | C_red / mF cm⁻² |
|---|---|---|---|---|---|---|---|---|---|
| CV_multpit_01 | 1.º | 50.0 | 10 (9 completos) | ciclo 9.0 | 0.841 | 7.17 | 7.40 | **6.55** | 1.85 |
| CV_multpit_02 | 2.º | 5.0 | 4 (4 completos) | ciclo 4.0 | 0.484 | 11.33 | 12.49 | **7.39** | 2.09 |
| CV_multpit_03 | 3.º | 10.0 | 4 (4 completos) | ciclo 4.0 | 0.687 | 9.23 | 9.74 | **7.52** | 2.13 |
| CV_multpit_04 | 4.º | 20.0 | 4 (4 completos) | ciclo 4.0 | 0.817 | 7.53 | 7.74 | **6.77** | 1.92 |
| CV_multpit_05 | 5.º | 50.0 | 4 (4 completos) | ciclo 4.0 | 0.904 | 5.81 | 5.87 | **5.52** | 1.56 |

`C_lazo` = ∮|i|dt ÷ 2νΔV, `C_rama` = ∫i dt de la rama anódica ÷ νΔV,
`C_red` = carga de reducción ÷ ΔV. Los tres coinciden en un condensador
ideal; aquí no coinciden, y la columna `Q_red/Q_ox` dice por qué. **La
columna a citar es `C_red`**: es la única que no lleva dentro la corriente
parásita.

## Tabla 2 — Balance de carga y corriente parásita

| ν / mV s⁻¹ | duración del ciclo / s | Q_ox / µC | Q_red / µC | ΔQ / µC | ΔQ/t / µA |
|---|---|---|---|---|---|
| 50 | 40.0 | 122.1 | 110.4 | 11.7 | 0.292 |
| 20 | 100.0 | 165.8 | 135.5 | 30.3 | 0.303 |
| 10 | 200.0 | 219.0 | 150.4 | 68.6 | 0.343 |
| 5 | 400.0 | 305.5 | 147.8 | 157.8 | 0.394 |

Ajuste ΔQ = ⟨i_par⟩·t:  **⟨i_par⟩ = 0.41 µA**, ordenada -8.9 µC, R² = 0.9962.
La ordenada tiene que ser cero y lo es, dentro de cargas de 10 a 150 µC.

| ν / mV s⁻¹ | fracción de Q_ox que es fuga |
|---|---|
| 5 | 54 % |
| 10 | 38 % |
| 20 | 25 % |
| 50 | 13 % |

## Tabla 3 — Las tres leyes para i(ν), potencial a potencial

Tres modelos de dos parámetros cada uno, ajustados a las mismas cuatro
velocidades, con los residuos por tanto comparables.

| E / V | b de a·ν^b | R²(potencia) | R²(Dunn) | R²(fuga) | k_cap / µF | i_par / µA |
|---|---|---|---|---|---|---|
| -0.50 | 0.867 | 0.97493 | 0.98988 | 0.98029 | 111 | +0.59 |
| -0.40 | 0.930 | 0.98280 | 0.99408 | 0.98964 | 128 | +0.42 |
| -0.30 | 0.956 | 0.98687 | 0.99575 | 0.99297 | 122 | +0.30 |
| -0.20 | 0.805 | 0.99550 | 0.99702 | 0.99137 | 95 | +0.54 |
| -0.10 | 0.783 | 0.99553 | 0.99675 | 0.99063 | 97 | +0.61 |
| -0.00 | 0.725 | 0.99687 | 0.99705 | 0.99001 | 94 | +0.75 |
| +0.10 | 0.679 | 0.99750 | 0.99746 | 0.98959 | 93 | +0.89 |
| +0.20 | 0.584 | 0.99801 | 0.99844 | 0.98889 | 96 | +1.32 |
| +0.30 | 0.458 | 0.99326 | 0.99432 | 0.99232 | 106 | +2.30 |

Ninguna de las tres gana. Es lo que se espera de cuatro puntos y dos
parámetros, y es el motivo por el que un b de 0.46 en el extremo positivo
NO es prueba de difusión: una corriente constante de 2.3 µA sobre una
capacitancia de 106 µF produce exactamente ese b, y esa corriente está
medida aparte en la tabla 2.

## Tabla 4 — Degradación durante la sesión

| | 1.ª medida a 50 mV/s (CV_01) | última (CV_05) | cambio |
|---|---|---|---|
| C_lazo / F g⁻¹ | 7.17 | 5.81 | -18.9 % |
| **C_red / F g⁻¹** | 6.55 | 5.52 | -15.7 % |
| Q_red/Q_ox | 0.84 | 0.90 | +7.5 % |

Entre las dos medidas hay 22 ciclos. Esa pérdida está dentro de la serie
de velocidades, porque las velocidades se midieron en orden: parte de lo
que la retención llama «dependencia con ν» es esto.

Dentro del propio CV_01 (10 ciclos seguidos a 50 mV/s) la `C_red` apenas
baja, de 6.61 a 6.55 F/g, mientras la `C_lazo` cae de 7.82 a 7.17: lo que
se va en esos diez ciclos es la fuga, no lo almacenado.

