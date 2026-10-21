"""Article tables from the saved summaries."""
import json
import sys

OUT = "/home/user/Cc/packages/ramancarbon/analisis_FeSe"


def load(sample):
    with open(f"{OUT}/{sample}_resumen.json") as handle:
        return json.load(handle)


def fraction(row):
    weight, sigma = row["weight"], row["weight_sigma"]
    if weight is None:
        return "n/d"
    if sigma:
        return f"{100 * weight:.1f} ± {100 * sigma:.1f}"
    return f"{100 * weight:.1f}"


def table_phases(summaries):
    lines = [
        "### Tabla 1. Fracciones en peso de la parte cristalina identificada",
        "",
        "| Muestra | Fase | Grupo esp. | % en peso | % de la intensidad "
        "difractada | Tamaño de dominio (nm) | a (Å) | c (Å) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for sample, summary in summaries.items():
        label = {"S1_900": "S1 (900 °C)", "S2_1020": "S2 (1020 °C)"}[sample]
        for index, row in enumerate(summary["phases"]):
            size = row["size_nm"]
            lines.append(
                "| %s | %s | %s | %s | %.1f | %s | %.4f | %.4f |"
                % (label if index == 0 else "",
                   row["pretty"].split(" (")[0],
                   row["pretty"].split("(")[-1].rstrip(")") if "(" in row["pretty"] else "—",
                   fraction(row),
                   100 * row["share"],
                   "—" if size is None else f"{size:.1f}",
                   row["a"], row["c"])
            )
    return "\n".join(lines)


def table_quality(summaries):
    lines = [
        "### Tabla 2. Calidad del ajuste",
        "",
        "| | " + " | ".join(
            {"S1_900": "S1 (900 °C)", "S2_1020": "S2 (1020 °C)"}[s]
            for s in summaries) + " |",
        "|---|" + "---:|" * len(summaries),
    ]
    def row(label, key, fmt, scale=1.0):
        values = " | ".join(fmt % (scale * summaries[s][key]) for s in summaries)
        lines.append(f"| {label} | {values} |")
    row("R_wp (%)", "r_wp", "%.2f", 100)
    row("R_p (%)", "r_p", "%.2f", 100)
    row("R_exp (%)", "r_expected", "%.2f", 100)
    row("GOF (= R_wp/R_exp)", "gof", "%.3f")
    row("χ² reducido", "chi2", "%.3f")
    row("Parámetros libres", "free_parameters", "%d")
    row("Puntos del patrón", "n_points", "%d")
    row("Fondo / patrón calculado (%)", "background_share", "%.1f", 100)
    row("Alturas calc./obs., envolvente (%)", "height_envelope", "%.0f", 100)
    row("Desplazamiento de cero (°)", "zero", "%+.4f")
    return "\n".join(lines)


def table_coordinates(summaries):
    lines = [
        "### Tabla 3. Coordenadas atómicas refinadas",
        "",
        "Solo las que la simetría deja libres y el átomo tiene peso "
        "dispersor para fijar. Las demás quedan en el valor publicado.",
        "",
        "| Muestra | Fase | Sitio | Publicada | Refinada | Δ |",
        "|---|---|---|---:|---:|---:|",
    ]
    any_row = False
    for sample, summary in summaries.items():
        label = {"S1_900": "S1 (900 °C)", "S2_1020": "S2 (1020 °C)"}[sample]
        for row in summary["phases"]:
            if not row["site_shifts"] or not any(row["site_shifts"]):
                continue
            for site in row["sites"]:
                published, refined = site["published"], site["fract"]
                for axis, (was, now) in zip("xyz", zip(published, refined)):
                    if abs(now - was) < 1e-5:
                        continue
                    any_row = True
                    lines.append(
                        "| %s | %s | %s %s | %.4f | %.4f | %+.4f |"
                        % (label, row["pretty"].split(" (")[0],
                           site["label"], axis, was, now, now - was))
    if not any_row:
        lines.append("| — | — | — | — | — | — |")
    return "\n".join(lines)


def table_background(summaries):
    lines = [
        "### Tabla 4. Sensibilidad al orden del polinomio de fondo",
        "",
        "El número que más mueve las fracciones, y es una elección de "
        "modelado y no una medida.",
        "",
        "| Muestra | Fase | Dispersión de la fracción sobre los órdenes "
        "probados (puntos %) |",
        "|---|---|---:|",
    ]
    for sample, summary in summaries.items():
        label = {"S1_900": "S1 (900 °C)", "S2_1020": "S2 (1020 °C)"}[sample]
        spread = summary.get("background_spread") or {}
        for index, (name, value) in enumerate(sorted(
                spread.items(), key=lambda kv: -kv[1])):
            lines.append("| %s | %s | %.1f |"
                         % (label if index == 0 else "", name, 100 * value))
    return "\n".join(lines)


if __name__ == "__main__":
    samples = sys.argv[1:] or ["S1_900", "S2_1020"]
    summaries = {s: load(s) for s in samples}
    parts = [
        table_phases(summaries), "", table_quality(summaries), "",
        table_coordinates(summaries), "", table_background(summaries), "",
        "### Avisos del refinamiento", "",
    ]
    for sample, summary in summaries.items():
        parts.append(f"**{sample}**")
        parts.append("")
        for warning in summary["warnings"]:
            parts.append(f"- {warning}")
        parts.append("")
    text = "\n".join(parts)
    with open(f"{OUT}/tablas.md", "w") as handle:
        handle.write(text + "\n")
    print(text[:3000])
