

class TestDescendingAxis:
    """Many Raman instruments export high-to-low.

    A real 532 nm file arrived running 3000.8 down to 99.9 cm-1 and came
    back "desconocido", because every shape test in the detector is
    written for an axis that grows and the whole block sat behind
    `_ascending`. Sorting first costs nothing and is what any reader does
    with the file anyway.
    """

    def _write(self, tmp_path, ascending: bool):
        import numpy as np
        shift = np.linspace(100.0, 3000.0, 400)
        signal = (80.0 + 400.0 * np.exp(-0.5 * ((shift - 1345.0) / 60.0) ** 2)
                  + 380.0 * np.exp(-0.5 * ((shift - 1585.0) / 30.0) ** 2))
        if not ascending:
            shift, signal = shift[::-1], signal[::-1]
        path = tmp_path / ("sube.txt" if ascending else "baja.txt")
        with path.open("w") as handle:
            handle.write("#Wave\t#Intensity\n")
            for a, b in zip(shift, signal, strict=True):
                handle.write(f"{a:.6f}\t{b:.6f}\n")
        return path

    def test_a_descending_spectrum_is_recognised_as_raman(self, tmp_path):
        from ramancarbon.dataio.detect import detect

        assert detect(self._write(tmp_path, False)).kind == "raman"

    def test_it_reads_the_same_either_way(self, tmp_path):
        """The order is a convention of the instrument, not a difference
        in what was measured, so both files must give the same answer."""
        from ramancarbon.dataio.detect import detect

        up = detect(self._write(tmp_path, True))
        down = detect(self._write(tmp_path, False))
        assert up.kind == down.kind == "raman"

    def test_a_descending_file_loads_and_comes_back_ascending(self, tmp_path):
        import numpy as np

        from ramancarbon.dataio import load

        spectrum = load(self._write(tmp_path, False)).data
        shift = np.asarray(spectrum.shift)
        assert np.all(np.diff(shift) > 0), "el lector debe entregar el eje ordenado"


class TestTheExtensionAscIsNotAnInstrument:
    """`.asc` was mapped to diffraction and it should not have been.

    Adding Rigaku's ASCII export to the suite put `.asc` in the table of
    extensions that name their instrument, and from then on any Raman
    spectrum saved as `.asc` -- which is most of them, in a building where
    every machine writes `.asc` -- came back "xrd" with the correct shape
    reading demoted to an alternative. Rigaku's own export carries `*ASC`
    and `*SCAN_AXIS` in its header, and that is the check that should
    decide, because it is evidence from the file.
    """

    def _raman(self, tmp_path, suffix):
        import numpy as np

        shift = np.linspace(120.0, 3200.0, 500)
        signal = (90.0
                  + 420.0 * np.exp(-0.5 * ((shift - 1345.0) / 55.0) ** 2)
                  + 400.0 * np.exp(-0.5 * ((shift - 1585.0) / 26.0) ** 2))
        path = tmp_path / f"espectro{suffix}"
        with path.open("w", encoding="utf-8") as handle:
            handle.write("#Wave\t#Intensity\n")
            for a, b in zip(shift, signal, strict=True):
                handle.write(f"{a:.6f}\t{b:.6f}\n")
        return path

    def test_a_raman_spectrum_saved_as_asc_is_still_raman(self, tmp_path):
        from ramancarbon.dataio.detect import detect

        found = detect(self._raman(tmp_path, ".asc"))
        assert found.kind == "raman"
        assert "xrd" not in found.alternatives

    def test_the_same_numbers_under_four_extensions_agree(self, tmp_path):
        from ramancarbon.dataio.detect import detect

        kinds = {detect(self._raman(tmp_path, suffix)).kind
                 for suffix in (".txt", ".dat", ".csv", ".asc")}
        assert kinds == {"raman"}

    def test_a_rigaku_asc_is_still_recognised_by_its_header(self, tmp_path):
        from ramancarbon.dataio.detect import detect

        path = tmp_path / "patron.asc"
        path.write_text(
            "*TYPE = RAW\n*ASC\n*SCAN_AXIS = Theta/2-Theta\n"
            "*START = 10.0000\n*STOP = 90.0000\n*STEP = 0.0100\n"
            "*COUNT = 8\n*RAS_INT_START\n"
            "120, 118, 131, 125\n129, 122, 117, 126\n*RAS_INT_END\n",
            encoding="utf-8",
        )
        found = detect(path)
        assert found.kind == "xrd"
        assert found.fmt == "asc"
