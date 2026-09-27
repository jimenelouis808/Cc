"""The "Calcular → Trabajos" page: every calculation the window runs, in one queue.

GPAW (vibspec), QE, SIESTA and LAMMPS directories share the window's
:class:`carbonforge.jobs.JobQueue`; this page lists them with their engine,
state and progress, shows the selected job's log, cancels, and opens a
finished job's results on the right page. The decisions (what runs a
directory, whether it finished, what it produced) are in
:mod:`carbonforge.jobs`.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Optional

from ...jobs.queue import DONE, Job, adapter_for, job_log, result_of

_POLL_MS = 1000


class JobsTab:
    """The shared job list.

    Uses from the host: ``tk``, ``ttk``, ``root``, ``jobs`` (the queue),
    ``_show_error``, ``select_page``, ``_ensure_vibspec``, and the analysis
    page's ``_on_open_bands`` / ``_on_open_spectrum`` / ``_on_open_dos`` /
    ``_on_open_pdos``.
    """

    def _build_jobs_tab(self, parent) -> None:
        tk, ttk = self.tk, self.ttk
        outer = ttk.Frame(parent, padding=8)
        outer.pack(fill="both", expand=True)

        bar = ttk.Frame(outer)
        bar.pack(fill="x")
        ttk.Button(bar, text="Añadir directorio…", command=self._on_add_job_dir).pack(side="left")
        ttk.Label(bar, text="Procesos MPI").pack(side="left", padx=(12, 2))
        self.jobs_nprocs_var = tk.StringVar(value="1")
        ttk.Entry(bar, textvariable=self.jobs_nprocs_var, width=4).pack(side="left")
        ttk.Button(bar, text="Cancelar", command=self._on_cancel_job).pack(side="left", padx=4)
        ttk.Button(bar, text="Abrir resultados", command=self._on_job_results).pack(side="left")
        ttk.Button(bar, text="Quitar terminados", command=self._on_clear_jobs
                   ).pack(side="left", padx=4)

        columns = ("motor", "estado", "progreso", "carpeta")
        self.jobs_tree = ttk.Treeview(outer, columns=columns, height=10)
        for key, text, width in (("#0", "cálculo", 200), ("motor", "motor", 70),
                                 ("estado", "estado", 90), ("progreso", "progreso", 260),
                                 ("carpeta", "carpeta", 360)):
            self.jobs_tree.heading(key, text=text)
            self.jobs_tree.column(key, width=width)
        self.jobs_tree.pack(fill="x", pady=(6, 0))
        self.jobs_tree.bind("<<TreeviewSelect>>", lambda _e: self._refresh_job_log())

        self.jobs_status_var = tk.StringVar(
            value="Encola desde Preparar (QE, SIESTA, LAMMPS) o desde IR y Raman (GPAW); o añade "
                  "un directorio ya preparado.")
        ttk.Label(outer, textvariable=self.jobs_status_var, wraplength=900,
                  foreground="#667").pack(anchor="w", pady=4)
        ttk.Label(outer, text="Log").pack(anchor="w")
        self.jobs_log = tk.Text(outer, wrap="word", height=18, font=("TkFixedFont", 9))
        self.jobs_log.pack(fill="both", expand=True)
        self.jobs_log.configure(state="disabled")
        self.root.after(_POLL_MS, self._poll_jobs)

    # -- queue ---------------------------------------------------------

    def submit_job(self, directory: Path, nprocs: Optional[int] = None) -> Optional[Job]:
        """Queue ``directory`` if this machine can run it; say why not otherwise."""
        directory = Path(directory)
        try:
            ok, why = adapter_for(directory).available(directory)
            if not ok:
                self.jobs_status_var.set(why)
                return None
            job = self.jobs.submit(directory, nprocs=nprocs or self._jobs_nprocs())
        except Exception as exc:
            self._show_error(exc, traceback.format_exc())
            return None
        self.jobs.poll()
        self._on_job_submitted(job)
        return job

    def _on_job_submitted(self, job: Optional[Job]) -> None:
        """Show the list (and the new job, if any)."""
        self._refresh_jobs()
        if job is not None and self.jobs_tree.exists(str(job.directory)):
            self.jobs_tree.selection_set(str(job.directory))
            self.jobs_status_var.set(f"{job.name} encolado ({job.engine}).")
        self.select_page("Trabajos")

    def _jobs_nprocs(self) -> int:
        try:
            return max(1, int(self.jobs_nprocs_var.get()))
        except ValueError:
            return 1

    def _poll_jobs(self) -> None:
        try:
            if self.jobs.poll() or self.jobs.active():
                self._refresh_jobs()
                self._refresh_job_log()
        finally:
            self.root.after(_POLL_MS, self._poll_jobs)

    def _refresh_jobs(self) -> None:
        present = set()
        for job in self.jobs.jobs:
            iid = str(job.directory)
            present.add(iid)
            values = (job.engine, job.state, job.progress(), str(job.directory.parent))
            if self.jobs_tree.exists(iid):
                self.jobs_tree.item(iid, values=values)
            else:
                # An export's engine folders are all called qe/, siesta/...:
                # name them with the export they belong to.
                label = job.name if job.engine == "gpaw" else \
                    f"{job.directory.parent.name}/{job.name}"
                self.jobs_tree.insert("", "end", iid=iid, text=label, values=values)
        for iid in self.jobs_tree.get_children():
            if iid not in present:
                self.jobs_tree.delete(iid)

    def _selected_shared_job(self) -> Optional[Job]:
        selection = self.jobs_tree.selection()
        if not selection:
            return None
        return next((j for j in self.jobs.jobs if str(j.directory) == selection[0]), None)

    def _refresh_job_log(self) -> None:
        job = self._selected_shared_job()
        if job is None:
            return
        self.jobs_log.configure(state="normal")
        self.jobs_log.delete("1.0", "end")
        self.jobs_log.insert("1.0", job_log(job))
        self.jobs_log.configure(state="disabled")
        self.jobs_log.see("end")

    # -- buttons -------------------------------------------------------

    def _on_add_job_dir(self) -> None:
        from tkinter import filedialog

        path = filedialog.askdirectory(title="Directorio de cálculo (record.json o job.json)")
        if path:
            self.submit_job(Path(path))

    def _on_cancel_job(self) -> None:
        job = self._selected_shared_job()
        if job is not None:
            self.jobs.cancel(job)
            self._refresh_jobs()

    def _on_clear_jobs(self) -> None:
        self.jobs.remove_finished()
        self._refresh_jobs()

    def _on_job_results(self) -> None:
        """Open what a finished job produced on the page that shows it."""
        job = self._selected_shared_job()
        if job is None:
            return
        if job.state != DONE:
            self.jobs_status_var.set(f"{job.name} no ha terminado ({job.state}).")
            return
        found = result_of(job.directory)
        if found is None:
            self.jobs_status_var.set(f"{job.name} no produjo nada que esta ventana grafique; "
                                     f"mira la carpeta {job.directory}.")
            return
        kind, path = found
        if kind == "ir_gpaw":
            self._ensure_vibspec().show_results(path)
        elif kind == "bands":
            self.select_page("Bandas y espectros")
            self._on_open_bands(path)
        elif kind == "spectrum":
            self.select_page("Bandas y espectros")
            self._on_open_spectrum(path)
        elif kind == "pdos":
            self.select_page("Bandas y espectros")
            self._on_open_pdos(str(path.parent))
        else:
            self.select_page("Bandas y espectros")
            self._on_open_dos(path)
