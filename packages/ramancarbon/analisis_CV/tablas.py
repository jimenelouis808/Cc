import json
import numpy as np

OUT = "/home/user/Cc/packages/ramancarbon/analisis_CV"
s = json.load(open(f"{OUT}/resumen.json"))
rows = {r["archivo"]: r for r in s["por_archivo"]}
cycles = np.genfromtxt(f"{OUT}/ciclos.csv", delimiter=",", names=True,
                       dtype=None, encoding="utf-8")
c = s["condiciones"]

print("# Voltametría cíclica — tablas\n")
print("""Muestra **S1**, el FeSe/C tratado a 900 °C ya analizado aquí por Raman y
por difracción. Electrodo de trabajo de carbono vítreo de 3 mm
(A = %.5f cm²) con 20 µg de material activo, en **KOH 0.5 M**, frente a
**Ag/AgCl en KCl saturado**, con alambre de Pt como contraelectrodo.

Ventana **%+.3f a %+.3f V vs Ag/AgCl**, que con %s
son **%+.3f a %+.3f V vs RHE**. El límite positivo queda **%.0f mV POR
ENCIMA** del equilibrio del oxígeno (1.23 V vs RHE).

De cada archivo se informa el ÚLTIMO ciclo completo.
""" % (c["area_geometrica_cm2"], c["ventana_v_vs_referencia"][0],
       c["ventana_v_vs_referencia"][1], c["conversion_a_rhe"],
       c["ventana_v_vs_rhe"][0], c["ventana_v_vs_rhe"][1],
       1e3*c["sobrepotencial_oer_en_el_limite_positivo_v"]))

print("## Tabla 1 — Capacitancia por archivo\n")
print("| archivo | orden | ν / mV s⁻¹ | ciclos | ciclo informado | "
      "Q_red/Q_ox | C_lazo / F g⁻¹ | C_rama / F g⁻¹ | **C_red / F g⁻¹** | "
      "C_red / mF cm⁻² |")
print("|---|---|---|---|---|---|---|---|---|---|")
for n, tag in enumerate(s["orden_de_medida"], start=1):
    r = rows[f"CV_{tag}"]
    print("| CV_multpit_%s | %d.º | %.1f | %d (%d completos) | %s | %.3f | "
          "%.2f | %.2f | **%.2f** | %.2f |"
          % (tag, n, r["mv_s"], r["ciclos"], r["ciclos_completos"],
             r["ciclo_informado"].split("[")[-1].rstrip("]"),
             r["eficiencia_coulombica"], r["c_lazo_f_g"], r["c_rama_f_g"],
             r["c_reduccion_f_g"], r["c_reduccion_mf_cm2"]))

print("""
`C_lazo` = ∮|i|dt ÷ 2νΔV, `C_rama` = ∫i dt de la rama anódica ÷ νΔV,
`C_red` = carga de reducción ÷ ΔV. Los tres coinciden en un condensador
ideal; aquí no, y la columna `Q_red/Q_ox` dice por qué. **La columna a
citar es `C_red`**: es la única que no lleva dentro la corriente de la
reacción que corre en el límite positivo.
""")

print("## Tabla 2 — Balance de carga y corriente parásita\n")
print("| ν / mV s⁻¹ | duración del ciclo / s | Q_ox / µC | Q_red / µC | "
      "ΔQ / µC | ΔQ/t / µA | j / µA cm⁻² |")
print("|---|---|---|---|---|---|---|")
for tag in ("05", "04", "03", "02"):
    m = (cycles["archivo"] == f"CV_{tag}") & cycles["completo"]
    row = cycles[m][-1]
    net = row["q_ox_c"] - row["q_red_c"]
    print("| %.0f | %.1f | %.1f | %.1f | %.1f | %.3f | %.1f |"
          % (row["mv_s"], row["duracion_s"], 1e6*row["q_ox_c"],
             1e6*row["q_red_c"], 1e6*net, 1e6*net/row["duracion_s"],
             1e6*net/row["duracion_s"]/c["area_geometrica_cm2"]))
f = s["fuga"]
print("""
Ajuste ΔQ = ⟨i_par⟩·t:  **⟨i_par⟩ = %.2f µA**, ordenada %.1f µC, R² = %.4f.
La ordenada tiene que ser cero y lo es, dentro de cargas de 10 a 150 µC.
Son **%.1f µA cm⁻²** sobre el área geométrica, promediados sobre el ciclo;
en el límite positivo la ley de fuga pide 2.3 µA, o sea 33 µA cm⁻² a
η = %.0f mV. Es evolución de oxígeno sobre un material que lleva α-Fe,
Fe₃C y FeSe en álcali.
""" % (1e6*f["i_par_a"], 1e6*f["ordenada_c"], f["r2"],
       1e6*f["i_par_a"]/c["area_geometrica_cm2"],
       1e3*c["sobrepotencial_oer_en_el_limite_positivo_v"]))
print("| ν / mV s⁻¹ | fracción de Q_ox que es fuga |")
print("|---|---|")
for k, v in sorted(f["fraccion_de_q_ox"].items(), key=lambda kv: float(kv[0])):
    print("| %s | %.0f %% |" % (k, 100*v))

print("\n## Tabla 3 — Las tres leyes para i(ν), potencial a potencial\n")
print("Tres modelos de dos parámetros cada uno, ajustados a las mismas "
      "cuatro\nvelocidades, con los residuos por tanto comparables.\n")
print("| E / V vs Ag/AgCl | E / V vs RHE | b de a·ν^b | R²(potencia) | "
      "R²(Dunn) | R²(fuga) | k_cap / µF | i_par / µA |")
print("|---|---|---|---|---|---|---|---|")
shift = c["ventana_v_vs_rhe"][0] - c["ventana_v_vs_referencia"][0]
for law in s["leyes_de_i_nu"]:
    print("| %+.2f | %.2f | %.3f | %.5f | %.5f | %.5f | %.0f | %+.2f |"
          % (law["e_v"], law["e_v"]+shift, law["b"], law["r2"]["potencia"],
             law["r2"]["dunn"], law["r2"]["fuga"],
             1e6*law["c_fuga_f"], 1e6*law["i_par_a"]))

print("""
Ninguna gana, y ninguna puede: con cuatro velocidades y dos parámetros,
una constante y un término en √ν son casi la misma función sobre una
década. El b de 0.46 del extremo positivo no es prueba de difusión.
""")

print("## Tabla 4 — Qué pasa con el b si se quita la fuga medida\n")
sf = s["sin_la_fuga"]
print("Restando una corriente anódica constante de %.2f µA, la que da el\n"
      "BALANCE DE CARGA —que no mira cómo escala i con ν, así que la\n"
      "corrección no es circular—:\n" % (1e6*f["i_par_a"]))
print("| E / V vs Ag/AgCl | b medido | b sin la fuga |")
print("|---|---|---|")
for a, b in zip(s["b"], sf["b"]):
    print("| %+.3f | %.3f | %.3f |" % (a["e_v"], a["b"], b["b"]))
print("""
Y ahí está la respuesta a si el b se puede medir con estos datos: **no**.
Una sola constante sobrecorrige en el extremo negativo —el b pasa de 1,
que es imposible— y se queda corta en el positivo, donde la fuga vale
cinco veces su propia media. Los dos errores son mayores que el efecto
que se quiere medir.
""")

print("## Tabla 5 — Degradación durante la sesión\n")
print("| | 1.ª medida a 50 mV/s (CV_01) | última (CV_05) | cambio |")
print("|---|---|---|---|")
early, late = rows["CV_01"], rows["CV_05"]
for label, key in (("C_lazo / F g⁻¹", "c_lazo_f_g"),
                   ("**C_red / F g⁻¹**", "c_reduccion_f_g"),
                   ("Q_red/Q_ox", "eficiencia_coulombica")):
    print("| %s | %.2f | %.2f | %+.1f %% |"
          % (label, early[key], late[key], 100 * (late[key] / early[key] - 1.0)))
print("""
Entre las dos medidas hay 22 ciclos. Como las velocidades se midieron en
orden, esa pérdida está dentro de la serie: parte de lo que la retención
llama «dependencia con ν» es esto.

Dentro del propio CV_01 (10 ciclos seguidos a 50 mV/s) la `C_red` apenas
baja, de 6.61 a 6.55 F/g, mientras la `C_lazo` cae de 7.82 a 7.17: en
esos diez ciclos lo que se va es la fuga, no lo almacenado.
""")
