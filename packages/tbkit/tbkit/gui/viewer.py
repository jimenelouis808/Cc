"""The 3D view: atoms, bonds, per-atom colours, isosurfaces, arrows and animated modes."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyvista as pv
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.data.colors import jmol_colors

from .actions import bonds_with_images


def element_color(symbol: str):
    """Jmol colours, except H: pure white vanishes on the white background."""
    if symbol == "H":
        return (0.82, 0.82, 0.86)
    return tuple(float(c) for c in jmol_colors[atomic_numbers[symbol]])


class StructureView:
    """Draws into a pyvista plotter (a ``QtInteractor`` in the window, off-screen in tests)."""

    def __init__(self, plotter):
        self.plotter = plotter
        self.atoms: Optional[Atoms] = None
        #: Copies of the cell drawn along each periodic axis (the view only: the
        #: calculation keeps the structure as it is).
        self.repeat = (1, 1, 1)
        self.show_cell = True
        self._scalars = None
        self._label = ""
        self._cmap = "coolwarm"
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
        self._scalars, self._label, self._cmap = scalars, label, cmap
        self.plotter.clear()
        self._draw(atoms.get_positions(), scalars, label, cmap)
        cell = atoms.cell
        if self.show_cell and atoms.get_pbc().any() and cell.rank == 3:
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

    def set_repeat(self, repeat):
        """Draw ``(n1, n2, n3)`` copies of the cell (periodic axes only) and redraw."""
        self.repeat = tuple(max(1, int(n)) for n in repeat)
        if self.atoms is not None:
            self.show(self.atoms, self._scalars, self._label, self._cmap, keep_camera=False)

    def _images(self, atoms):
        """Lattice offsets of the copies drawn (only along periodic axes)."""
        ranges = [range(n if atoms.pbc[k] else 1) for k, n in enumerate(self.repeat)]
        return [(a, b, c) for a in ranges[0] for b in ranges[1] for c in ranges[2]]

    def update_positions(self, positions):
        """Move the atoms (a relaxation step) keeping the camera and the colours."""
        if self.atoms is None or len(positions) != len(self.atoms) or self._mode is not None:
            return
        self.atoms = self.atoms.copy()
        self.atoms.positions = np.asarray(positions)
        for actor in [n for n in self.plotter.actors if n.startswith("atoms")]:
            self.plotter.remove_actor(actor)
        self._draw(self.atoms.get_positions(), self._scalars, self._label, self._cmap)
        self.plotter.render()

    def _draw(self, positions, scalars=None, label="", cmap="coolwarm", name="atoms"):
        base = self.atoms
        cell = np.array(base.cell)
        images = self._images(base) if base.pbc.any() else [(0, 0, 0)]
        offsets = [np.array(img) @ cell for img in images]
        # the atoms of every copy, in copy order
        positions = np.concatenate([positions + o for o in offsets])
        atoms = Atoms(list(base.numbers) * len(images), positions=positions)
        if scalars is not None:
            scalars = np.tile(np.asarray(scalars, dtype=float), len(images))
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
                                      specular=0.4, name=f"{name}-{element}")
            else:
                values = np.asarray(scalars, dtype=float)[index]
                glyphs[label or "valor"] = np.repeat(values, sphere.n_points)
                limit = float(np.abs(values).max()) or 1.0
                self.plotter.add_mesh(glyphs, scalars=label or "valor", cmap=cmap,
                                      clim=(-limit, limit), smooth_shading=True, name=name,
                                      scalar_bar_args={"title": label or "valor"})
        self._drawn = positions
        # Bonds of the original cell, with the lattice offset of the far atom. A bond to an
        # atom in a copy that is drawn joins the two atoms; a bond that leaves the drawn
        # copies stops halfway at the boundary (a stub), never across the cell.
        probe = base.copy()
        probe.positions = positions[:len(base)]
        n = len(base)
        index = {img: k for k, img in enumerate(images)}
        full, stubs = set(), []
        for i, j, shift in bonds_with_images(probe):
            for a_atom, b_atom, step in ((i, j, shift), (j, i, tuple(-x for x in shift))):
                vector = probe.positions[b_atom] + np.array(step) @ cell - probe.positions[a_atom]
                for k, img in enumerate(images):
                    target = tuple(np.add(img, step))
                    a = k * n + a_atom
                    if target in index:
                        b = index[target] * n + b_atom
                        full.add((min(a, b), max(a, b)))
                    else:
                        stubs.append((a, positions[a] + 0.5 * vector))
        if full or stubs:
            points = list(positions) + [mid for _, mid in stubs]
            lines = []
            for a, b in sorted(full):
                lines += [2, a, b]
            for k, (a, _) in enumerate(stubs):
                lines += [2, a, len(positions) + k]
            mesh = pv.PolyData(np.array(points), lines=np.array(lines))
            self.plotter.add_mesh(mesh.tube(radius=0.09), color=(0.55, 0.55, 0.55),
                                  smooth_shading=True, name=f"{name}-bonds")

    def atom_at(self, point) -> int | None:
        """The atom (index in the structure) drawn nearest to a picked point, within 1.5 Å."""
        drawn = getattr(self, "_drawn", None)
        if drawn is None or self.atoms is None or point is None:
            return None
        distance = np.linalg.norm(drawn - np.asarray(point), axis=1)
        k = int(np.argmin(distance))
        return k % len(self.atoms) if distance[k] < 1.5 else None

    def enable_picking(self, callback) -> bool:
        """Call ``callback(atom_index)`` when an atom is clicked (interactive window only)."""
        def picked(point, *_):
            index = self.atom_at(point)
            if index is not None:
                callback(index)
        try:
            self.plotter.enable_point_picking(callback=picked, show_message=False,
                                              show_point=False, use_picker=True,
                                              left_clicking=True)
            return True
        except (TypeError, AttributeError):
            try:
                self.plotter.enable_point_picking(callback=picked, show_message=False)
                return True
            except Exception:  # noqa: BLE001  (an old pyvista, or no interactor)
                return False

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
