"""``python -m carbonforge.adsorption.run <carpeta>``: run an isotherm folder.

The job queue starts calculations as subprocesses, so a GCMC campaign
needs an entry point it can spawn. It is deliberately thin — every
decision is in :mod:`carbonforge.adsorption.campaign`, because a
subprocess's behaviour is the hardest kind to test and logic that lives
here is logic no unit test reaches.

Run by hand as well, on a machine with hours to spare: it resumes, so it
can be stopped and started, and `--pressure` re-runs a single point whose
warnings said it had not equilibrated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .campaign import read_isotherm_request, run_isotherm, status


def main(argv: list[str] | None = None) -> int:
    """Run the folder named on the command line. Returns an exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m carbonforge.adsorption.run",
        description="Corre una isoterma GCMC preparada en una carpeta. "
                    "Reanuda: los puntos que ya están en resultados.json no "
                    "se vuelven a calcular.")
    parser.add_argument("folder", help="Carpeta con isoterma.json.")
    parser.add_argument(
        "--pressure", type=float, action="append", default=None,
        dest="pressures", metavar="MPa",
        help="Repetible. Corre sólo estas presiones, de las que pide la "
             "carpeta. Para repetir un punto que avisó de que no había "
             "equilibrado.")
    args = parser.parse_args(argv)

    folder = Path(args.folder)
    try:
        request = read_isotherm_request(folder)
    except (FileNotFoundError, ValueError) as exc:
        print(f"No se puede leer la petición: {exc}", file=sys.stderr)
        return 1

    print(f"{request['nombre']}: {request['atomos']} átomos, "
          f"{request['temperatura_k']:g} K, "
          f"{len(request['presiones_mpa'])} presiones", flush=True)
    try:
        points = run_isotherm(folder, pressures=args.pressures)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Rechazado: {exc}", file=sys.stderr)
        return 1

    print(f"\n{'P (MPa)':>8} {'rho_o':>10} {'exceso':>10} {'wt% tot':>8} "
          f"{'wt% /C':>8} {'N':>10}", flush=True)
    for point in points:
        print(f"{point['presion_mpa']:8.2f} "
              f"{point['densidad_absoluta_g_cm3']:10.5f} "
              f"{point['densidad_exceso_g_cm3']:+10.5f} "
              f"{point['wt_pct_total']:8.2f} "
              f"{point['wt_pct_por_carbono']:8.2f} "
              f"{point['n_medio']:10.1f}", flush=True)
        for warning in point["avisos"]:
            print(f"         aviso: {warning}", flush=True)

    final = status(folder)
    print(f"\n{final['line']} — {final['state']}", flush=True)
    return 0 if final["state"] == "terminado" else 1


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
