"""The 3D view: atoms, bonds, per-atom colours, isosurfaces, arrows and animated modes."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyvista as pv
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.data.colors import jmol_colors

from .actions import bonds_of


def element_color(symbol: str):
    return tuple(float(c) for c in jmol_colors[atomic_numbers[symbol]])


class StructureView:
    """Draws into a pyvista plotter (a ``QtInteractor`` in the window, off-screen in tests)."""

    def __init__(self, plotter):
        self.plotter = plotter
        self.atoms: Optional[Atoms] = None
        self._timer = None
        self._mode = None
        self._phase = 0.0
        self.plotter.set_background("white")

    # --- structure ------------------------------------------------------------

    def show(self, atoms: Atoms, scalars: Optional[np.ndarray] = None, label: str = "",
             cmap: str = "coolwarm", keep_camera: bool = False):
        """Atoms (element colours, or ``scalars`` per atom on a diverging map) and bonds."""
        self.stop_animation()
        camera = self.plotter.camera_position if keep_camera and self.atoms is not None else None
        self.atoms = atoms
        self.plotter.clear()
        self._draw(atoms.get_positions(), scalars, label, cmap)
        cell = atoms.cell
        if atoms.get_pbc().any() and cell.rank == 3:
            self._draw_cell(np.array(cell))
        if camera is not None:
            self.plotter.camera_position = camera
        else:
            self.face(atoms.get_positions())
        self.plotter.render()

    def face(self, positions):
        """Look along the normal of the best-fit plane (a flake or molecule seen face-on)."""
        centred = positions - positions.mean(axis=0)
        if len(positions) >= 3:
            _, _, vt = np.linalg.svd(centred, full_matrices=False)
            self.plotter.view_vector(vt[-1], viewup=vt[0])
        self.plotter.reset_camera()

    def _draw(self, positions, scalars=None, label="", cmap="coolwarm", name="atoms"):
        atoms = self.atoms
        symbols = atoms.get_chemical_symbols()
        radii = 0.35 * covalent_radii[atoms.numbers] + 0.15
        for element in sorted(set(symbols)) if scalars is None else [None]:
            index = [i for i, s in enumerate(symbols) if element is None or s == element]
            cloud = pv.PolyData(positions[index])
            cloud["radius"] = radii[index]
            sphere = pv.Sphere(radius=1.0, theta_resolution=20, phi_resolution=20)
            glyphs = cloud.glyph(scale="radius", geom=sphere, orient=False)
            if scalars is None:
                self.plotter.add_mesh(glyphs, color=element_color(element), smooth_shading=True,
                                      name=f"{name}-{element}")
            else:
                values = np.asarray(scalars, dtype=float)[index]
                glyphs[label or "valor"] = np.repeat(values, sphere.n_points)
                limit = float(np.abs(values).max()) or 1.0
                self.plotter.add_mesh(glyphs, scalars=label or "valor", cmap=cmap,
                                      clim=(-limit, limit), smooth_shading=True, name=name,
                                      scalar_bar_args={"title": label or "valor"})
        bonds = bonds_of(atoms)
        if bonds:
            lines = np.hstack([[2, i, j] for i, j in bonds])
            mesh = pv.PolyData(positions, lines=lines)
            self.plotter.add_mesh(mesh.tube(radius=0.09), color=(0.55, 0.55, 0.55),
                                  smooth_shading=True, name=f"{name}-bonds")

    def _draw_cell(self, cell):
        corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) @ cell
        edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5),
                 (4, 6), (5, 7), (6, 7)]
        lines = np.hstack([[2, a, b] for a, b in edges])
        self.plotter.add_mesh(pv.PolyData(corners, lines=lines), color="black", line_width=1,
                              name="cell")

    # --- fields -----------------------------------------------------------------

    def isosurface(self, origin, spacing, values, level: Optional[float] = None):
        """± isosurfaces of a real field on a grid (lobes orange and teal, apart from
        the red-blue map of the per-atom colours)."""
        grid = pv.ImageData(dimensions=values.shape, spacing=spacing, origin=origin)
        grid["psi"] = values.ravel(order="F")
        level = level if level is not None else 0.25 * float(np.abs(values).max())
        for sign, color, name in ((1, (0.95, 0.6, 0.1), "iso+"), (-1, (0.1, 0.6, 0.6), "iso-")):
            surface = grid.contour([sign * level], scalars="psi")
            if surface.n_points:
                self.plotter.add_mesh(surface, color=color, opacity=0.55, smooth_shading=True,
                                      name=name)
            else:
                self.plotter.remove_actor(name)
        self.plotter.render()
        return level

    def clear_isosurface(self):
        for name in ("iso+", "iso-"):
            self.plotter.remove_actor(name)
        self.plotter.render()

    def arrows(self, vectors: np.ndarray, scale: float = 1.0, name: str = "arrows"):
        """One arrow per atom (a mode's displacement, a force, a dipole)."""
        cloud = pv.PolyData(self.atoms.get_positions())
        cloud["v"] = np.asarray(vectors) * scale
        self.plotter.add_mesh(cloud.glyph(orient="v", scale="v", factor=1.0,
                                          geom=pv.Arrow(shaft_radius=0.04, tip_radius=0.1)),
                              color=(0.1, 0.6, 0.1), name=name)
        self.plotter.render()

    # --- animated mode ---------------------------------------------------------

    def animate_mode(self, displacement: np.ndarray, amplitude: float = 0.4, timer=None):
        """Oscillate the structure along ``displacement`` (N, 3), normalised to ``amplitude`` Å."""
        self.stop_animation()
        largest = float(np.linalg.norm(displacement, axis=1).max()) or 1.0
        self._mode = np.asarray(displacement) * amplitude / largest
        self._phase = 0.0
        if timer is not None:
            self._timer = timer
            timer.timeout.connect(self._step)
            timer.start(40)

    def _step(self):
        if self._mode is None or self.atoms is None:
            return
        self._phase += 0.25
        positions = self.atoms.get_positions() + np.sin(self._phase) * self._mode
        for actor in [n for n in self.plotter.actors if n.startswith("atoms")]:
            self.plotter.remove_actor(actor)
        self._draw(positions)
        self.plotter.render()

    def stop_animation(self):
        if self._timer is not None:
            self._timer.stop()
            try:
                self._timer.timeout.disconnect(self._step)
            except (RuntimeError, TypeError):
                pass
            self._timer = None
        self._mode = None

    def screenshot(self, path: str):
        self.plotter.screenshot(path)
