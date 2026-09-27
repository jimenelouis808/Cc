

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
