"""The command line, end to end on small structures."""

from __future__ import annotations

import numpy as np
from ase.build import graphene, graphene_nanoribbon, molecule
from ase.io import write

from tbkit.cli import main


def test_levels_and_json(tmp_path, capsys):
    path = tmp_path / "benceno.xyz"
    write(path, molecule("C6H6"))
    out = tmp_path / "r.json"
    assert main(["levels", str(path), "--bonds", "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "gap 5.4000" in text and "enlace" in text and out.exists()


def test_scc_and_sp3(tmp_path, capsys):
    path = tmp_path / "piridina.xyz"
    write(path, molecule("C5H5N"))
    assert main(["levels", str(path), "--scc"]) == 0
    assert "SCC (convergido" in capsys.readouterr().out


def test_bands_and_dos(tmp_path):
    sheet = graphene(a=2.46, vacuum=5.0)
    sheet.pbc = (True, True, False)
    path = tmp_path / "grafeno.extxyz"
    write(path, sheet)
    assert main(["bands", str(path), "--path", "GKMG", "--npoints", "60",
                 "-o", str(tmp_path / "b.csv")]) == 0
    data = np.loadtxt(tmp_path / "b.csv", delimiter=",", skiprows=1)
    assert data.shape == (60, 3) and np.abs(data[:, 1:]).min() < 0.1
    assert main(["dos", str(path), "--kmesh", "12", "--pdos", "element",
                 "-o", str(tmp_path / "d.csv")]) == 0
    assert "C" in (tmp_path / "d.csv").read_text().splitlines()[0]


def test_hubbard_and_m_of_e(tmp_path, capsys):
    ribbon = graphene_nanoribbon(3, 1, type="zigzag", saturated=False, vacuum=6.0)
    path = tmp_path / "zgnr.extxyz"
    write(path, ribbon)
    assert main(["hubbard", str(path), "--U", "2.7", "--kmesh", "24",
                 "--m-energy", str(tmp_path / "m.csv")]) == 0
    assert "Momentos locales" in capsys.readouterr().out
    assert (tmp_path / "m.csv").exists()


def test_orbital_cube_and_errors(tmp_path, capsys):
    path = tmp_path / "benceno.xyz"
    write(path, molecule("C6H6"))
    assert main(["orbital", str(path), "--band", "lumo", "--spacing", "0.4",
                 "-o", str(tmp_path / "l.cube")]) == 0
    assert (tmp_path / "l.cube").exists()
    assert main(["bands", str(path)]) == 1          # finite: no bands


def test_imports_no_other_package_of_the_workspace():
    """tbkit stays independent (root CLAUDE.md): structures come in as files."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    siblings = {"carbonforge", "ramancarbon", "nanocarbon_lab"}
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            assert not any(n.split(".")[0] in siblings for n in names), path
