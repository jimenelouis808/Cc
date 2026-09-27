"""Quantum ESPRESSO Γ phonons in the TB Raman (fase E, paso 3).

The mode files are written here in the documented dynmat.x layout (as
displacements and as eigenvectors, with complex phases and complex
degenerate combinations), so the answer is known: importing the model's own
modes must reproduce the model's own Raman.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from ase.build import molecule

from tbkit.params import load_parameters
from tbkit.qe import modes_for_raman, read_qe_modes, write_qe_modes
from tbkit.raman import _model_phonons, raman, raman_from_qe


@pytest.fixture(scope="module")
def chn():
    return load_parameters("xu_chn")


@pytest.fixture(scope="module")
def methylamine(chn):
    from ase.optimize import BFGS

    from tbkit.calculator import TBCalculator

    atoms = molecule("H3CNH2")
    atoms.calc = TBCalculator(chn)
    BFGS(atoms, logfile=None).run(fmax=0.002, steps=300)
    atoms = atoms.copy()
    frequencies, modes = _model_phonons(atoms.copy(), chn, 12, 0.01, 0.005, [])
    return atoms, frequencies, modes


def _same_mode(a, b):
    a, b = a.ravel(), b.ravel()
    return min(np.abs(a - b).max(), np.abs(a + b).max())


class TestReader:
    @pytest.mark.parametrize("kind", ["displacements", "eigenvectors"])
    def test_round_trip_recovers_the_modes(self, methylamine, tmp_path, kind):
        atoms, frequencies, modes = methylamine
        path = write_qe_modes(tmp_path / "dynmat.out", frequencies, modes, kind,
                              atoms.get_masses())
        read = read_qe_modes(path)
        assert read.frequencies == pytest.approx(frequencies, abs=1e-5)
        vectors = modes_for_raman(read, atoms.get_masses())          # kind="auto"
        for k in range(6, len(modes)):                               # internal modes
            assert _same_mode(vectors[k], modes[k]) < 1e-5

    def test_complex_phase_and_degenerate_combinations(self, tmp_path):
        """(x + iy)/√2 in a degenerate pair and a mode times i are both real modes."""
        masses = np.array([12.0, 1.0])
        x = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
        y = np.array([[0, 1.0, 0], [0, -1.0, 0]])
        z = np.array([[0, 0, 1.0], [0, 0, 5.0]])
        lines = [" q = 0.0 0.0 0.0"]
        for k, (f, vec) in enumerate([(700.0, (x + 1j * y) / np.sqrt(2)),
                                      (700.0, (x - 1j * y) / np.sqrt(2)),
                                      (3000.0, 1j * z)], start=1):
            lines.append(f"     freq ({k:5d}) = {f / 33.356:12.6f} [THz] = {f:12.6f} [cm-1]")
            for row in vec:
                parts = " ".join(f"{c.real:10.6f} {c.imag:10.6f}" for c in row)
                lines.append(f" ( {parts} )")
        path = tmp_path / "modes.eig"
        path.write_text("\n".join(lines) + "\n")
        vectors = modes_for_raman(read_qe_modes(path), masses, "displacements")
        span = vectors[:2].reshape(2, -1)
        target = np.vstack([x.ravel(), y.ravel()])
        # the pair spans the same real plane as x and y
        rank = np.linalg.matrix_rank(np.vstack([span, target]), tol=1e-6)
        assert rank == 2
        assert np.abs(vectors[2].ravel() / np.abs(vectors[2]).max()
                      - np.sign(vectors[2].ravel()[5]) * z.ravel() / 5.0).max() < 1e-6

    def test_refusals(self, tmp_path, methylamine):
        atoms, frequencies, modes = methylamine
        path = write_qe_modes(tmp_path / "d.out", frequencies, modes)
        with pytest.raises(ValueError, match="átomos"):
            modes_for_raman(read_qe_modes(path), atoms.get_masses()[:-1])
        other_q = tmp_path / "q.out"
        other_q.write_text(path.read_text().replace("0.0000      0.0000      0.0000",
                                                    "0.5000      0.0000      0.0000"))
        with pytest.raises(ValueError, match="q = 0"):
            read_qe_modes(other_q)


class TestRamanWithImportedModes:
    def test_equals_the_native_raman(self, chn, methylamine, tmp_path):
        atoms, frequencies, modes = methylamine
        native = raman(atoms, chn)
        path = write_qe_modes(tmp_path / "dynmat.out", frequencies, modes)
        imported = raman_from_qe(atoms, chn, path)
        assert imported.frequencies == pytest.approx(native.frequencies, abs=1e-4)
        # the file carries six decimals per component
        assert imported.activities == pytest.approx(native.activities, rel=1e-4, abs=1e-6)
        assert imported.depolarization == pytest.approx(native.depolarization, abs=1e-5)
        assert any("importados" in w for w in imported.warnings)

    def test_needs_no_repulsive_term(self, chn, methylamine, tmp_path):
        atoms, frequencies, modes = methylamine
        electronic_only = dataclasses.replace(chn, repulsive=None)
        path = write_qe_modes(tmp_path / "dynmat.out", frequencies, modes)
        result = raman_from_qe(atoms, electronic_only, path)
        assert result.activities.max() > 0
        with pytest.raises(ValueError, match="repulsiva"):
            raman(atoms, electronic_only)
