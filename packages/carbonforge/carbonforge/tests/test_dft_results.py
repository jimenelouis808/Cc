"""Leer lo que dejan las recetas, en los objetos que el paquete ya dibuja."""

from __future__ import annotations

import json

import numpy as np
import pytest

from carbonforge.dft.results import (
    READERS,
    available_results,
    read_bands,
    read_dos,
    read_ldos,
    read_phonons,
)
from carbonforge.results.bands import BandStructure
from carbonforge.results.dos import DensityOfStates


@pytest.fixture
def bands_file(tmp_path):
    """Como lo escribe ase.spectrum.band_structure.BandStructure.write."""
    path = tmp_path / "bands.json"
    path.write_text(json.dumps({
        "energies": [[[-5.0, 1.0], [-4.5, 1.5], [-4.0, 2.0]]],
        "reference": -3.0,
        "path": {
            "kpts": [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]],
            "cell": [[2.46, 0, 0], [0, 2.46, 0], [0, 0, 15.0]],
            "special_points": {"G": [0.0, 0.0, 0.0], "X": [0.5, 0.0, 0.0]},
        },
    }), encoding="utf-8")
    return path


class TestLasBandas:
    def test_devuelve_el_objeto_del_paquete(self, bands_file):
        bands = read_bands(bands_file)
        assert isinstance(bands, BandStructure)
        assert bands.n_kpoints == 3 and bands.n_bands == 2

    def test_el_espin_se_reduce_al_primer_canal(self, bands_file):
        """ASE guarda (espín, k, banda); el dibujo espera (k, banda)."""
        assert read_bands(bands_file).energies.shape == (3, 2)

    def test_el_eje_x_se_recalcula_y_crece(self, bands_file):
        distances = read_bands(bands_file).distances
        assert distances[0] == 0.0
        assert np.all(np.diff(distances) > 0)

    def test_los_puntos_de_alta_simetria_se_colocan(self, bands_file):
        special = read_bands(bands_file).special_points
        assert [name for _d, name in special] == ["G", "X"]
        assert special[0][0] == 0.0

    def test_el_nivel_de_fermi_viaja(self, bands_file):
        assert read_bands(bands_file).fermi_energy == pytest.approx(-3.0)


class TestElDOS:
    def _write(self, tmp_path, **extra):
        path = tmp_path / "dos.json"
        path.write_text(json.dumps({
            "fermi": -2.5,
            "energies": [-1.0, 0.0, 1.0],
            "total": [0.1, 0.5, 0.2],
            **extra,
        }), encoding="utf-8")
        return path

    def test_total_en_el_objeto_del_paquete(self, tmp_path):
        total, projections = read_dos(self._write(tmp_path))
        assert isinstance(total, DensityOfStates)
        assert total.fermi_energy == pytest.approx(-2.5)
        assert projections == {}

    def test_las_proyecciones_vienen_aparte(self, tmp_path):
        path = self._write(tmp_path, por_orbital={"s": [0.0, 0.1, 0.0],
                                                  "p": [0.1, 0.4, 0.2]})
        _total, projections = read_dos(path)
        assert set(projections) == {"s", "p"}

    def test_las_proyecciones_no_pretenden_sumar_el_total(self, tmp_path):
        """La proyección sobre orbitales atómicos es incompleta: el lector
        las devuelve aparte justamente para no insinuar lo contrario."""
        path = self._write(tmp_path, por_orbital={"s": [0.0, 0.1, 0.0]})
        total, projections = read_dos(path)
        assert not np.allclose(projections["s"], total.dos)


class TestElLDOS:
    def test_un_sitio_por_atomo_elegido(self, tmp_path):
        path = tmp_path / "ldos.json"
        path.write_text(json.dumps({
            "fermi": -1.0,
            "energies": [-1.0, 0.0, 1.0],
            "sitios": {"7": {"simbolo": "N",
                             "orbitales": {"s": [0.0, 0.2, 0.0],
                                           "p": [0.1, 0.5, 0.1]}},
                       "3": {"simbolo": "C",
                             "orbitales": {"p": [0.2, 0.3, 0.1]}}},
        }), encoding="utf-8")
        sites, fermi = read_ldos(path)
        assert [s.index for s in sites] == [3, 7]      # ordenados
        assert sites[1].symbol == "N"
        assert fermi == pytest.approx(-1.0)

    def test_el_total_del_sitio_suma_sus_orbitales(self, tmp_path):
        path = tmp_path / "ldos.json"
        path.write_text(json.dumps({
            "energies": [0.0, 1.0],
            "sitios": {"0": {"simbolo": "C",
                             "orbitales": {"s": [1.0, 2.0], "p": [3.0, 4.0]}}},
        }), encoding="utf-8")
        sites, _ = read_ldos(path)
        assert sites[0].total.tolist() == [4.0, 6.0]


class TestLosFonones:
    def test_las_imaginarias_se_cuentan(self, tmp_path):
        path = tmp_path / "phonons.json"
        path.write_text(json.dumps({"gamma_cm1": [-120.0, -0.4, 0.2,
                                                  800.0, 1580.0]}),
                        encoding="utf-8")
        phonons = read_phonons(path)
        assert len(phonons.imaginary) == 1
        assert not phonons.is_a_minimum

    def test_los_modos_rigidos_no_cuentan_como_imaginarias(self, tmp_path):
        """Las traslaciones salen como ceros numéricos con signo cualquiera;
        tomarlas por modos imaginarios condenaría cualquier cálculo bueno."""
        path = tmp_path / "phonons.json"
        path.write_text(json.dumps({"gamma_cm1": [-0.6, -0.1, 0.3, 1580.0]}),
                        encoding="utf-8")
        assert read_phonons(path).is_a_minimum


class TestLoQueFaltaSeDiceClaro:
    def test_un_archivo_que_no_esta(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no ha corrido"):
            read_dos(tmp_path / "dos.json")

    def test_un_json_a_medias(self, tmp_path):
        path = tmp_path / "dos.json"
        path.write_text('{"energies": [1.0, 2.0', encoding="utf-8")
        with pytest.raises(ValueError, match="interrumpido"):
            read_dos(path)

    def test_solo_se_ofrece_lo_que_existe(self, tmp_path):
        (tmp_path / "dos.json").write_text("{}", encoding="utf-8")
        found = available_results(tmp_path)
        assert set(found) == {"dos"}

    def test_cada_tipo_con_resultado_tiene_su_lector(self):
        from carbonforge.dft.kinds import KINDS

        for kind, (name, reader) in READERS.items():
            assert kind in KINDS, kind
            assert name in KINDS[kind].produces, kind
            assert callable(reader)
