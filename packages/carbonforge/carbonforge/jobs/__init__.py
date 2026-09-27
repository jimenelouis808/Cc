"""Running calculations: one queue for GPAW, QE, SIESTA and LAMMPS directories.

* :mod:`~carbonforge.jobs.manifest` -- ``job.json``: the ordered steps of an
  export directory and their history.
* :mod:`~carbonforge.jobs.run` -- ``python -m carbonforge.jobs.run DIR``
  runs those steps, resumably.
* :mod:`~carbonforge.jobs.queue` -- subprocess queue with an adapter per kind
  of directory (vibspec ``record.json`` or ``job.json``).

No Tk here: the window only draws what this package decides. The runner is
not imported here, so ``python -m carbonforge.jobs.run`` starts clean.
"""

from .manifest import JobManifest, Step, manifest_for_directory, write_manifest
from .queue import (
    CANCELLED,
    DONE,
    ERROR,
    QUEUED,
    RUNNING,
    Job,
    JobQueue,
    adapter_for,
    engine_of,
    job_log,
)

__all__ = [
    "CANCELLED",
    "DONE",
    "ERROR",
    "QUEUED",
    "RUNNING",
    "Job",
    "JobManifest",
    "JobQueue",
    "Step",
    "adapter_for",
    "engine_of",
    "job_log",
    "manifest_for_directory",
    "write_manifest",
]
