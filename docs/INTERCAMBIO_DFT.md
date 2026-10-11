# Intercambio DFT entre paquetes: `nanocarbon-dft/1`

Cómo pide tbkit (u otro paquete) un cálculo DFT, y cómo lo prepara y corre carbonforge, sin que
ningún paquete importe al otro (regla 1 del `CLAUDE.md` de la raíz). Todo pasa por **una carpeta**
con archivos de texto y `.npy`. Cada paquete tiene su propio lector y escritor de este formato y
sus propias pruebas.

```
mi_solicitud/
├── solicitud.json      ← lo escribe quien pide (tbkit: «Validar con DFT…», dft_exchange.write_request)
├── estructura.extxyz   ← la geometría (celda y ejes periódicos incluidos)
├── vectores.npy        ← opcional: vectores (n, N, 3) para la tarea «desplazamientos»
├── run_dft.py          ← lo copia carbonforge (carbonforge dft preparar): solo usa ASE y el motor
├── trabajo.slurm       ← opcional: carbonforge dft preparar --slurm
├── tareas/…            ← resultados parciales (uno por cálculo): se reanuda donde quedó
├── progreso.json       ← avance (mismo formato que tbkit.progress: tbkit estado lo ve)
└── resultado.json      ← el resultado, con "terminado": true al final
```

## `solicitud.json`

```json
{
  "esquema": "nanocarbon-dft/1",
  "nombre": "coil_N_modos",
  "origen": {"programa": "tbkit", "version": "0.x", "fecha": "2026-10-11 12:00", "motivo": "…"},
  "estructura": "estructura.extxyz",
  "codigo": "gpaw",
  "ajustes": {"xc": "PBE", "modo": "lcao", "base": "dzp", "h": 0.2, "kpts": [1, 1, 2],
              "ensanchamiento_ev": 0.05, "espin": false, "carga": 0},
  "tareas": [
    {"id": "sp", "tipo": "punto_simple"},
    {"id": "relax", "tipo": "relajar", "fmax": 0.05, "pasos": 300},
    {"id": "modos", "tipo": "desplazamientos", "vectores": "vectores.npy", "amplitud": 0.01,
     "sobre": "relax"},
    {"id": "fc", "tipo": "hessiana", "delta": 0.01, "atomos": null},
    {"id": "dos", "tipo": "dos", "sigma_ev": 0.1}
  ],
  "recursos": {"mpi": 4, "hilos": 1, "memoria_gb": 24}
}
```

| Campo | Qué es |
|---|---|
| `codigo` | `gpaw`, `qe` (Quantum ESPRESSO, `pw.x`), `dftb+`, o `emt`, que es solo para pruebas (EMT de ASE, sin DFT) |
| `ajustes` | Comunes: `xc`, `kpts`, `ensanchamiento_ev`, `espin`, `carga`. GPAW: `modo` (`lcao` o `pw`), `base`, `h`, `ecut_ev` (pw). QE: `ecut_ry`, `ecutrho_ry`, `pseudos` ({elemento: archivo}), `pseudo_dir`. DFTB+: `skf_dir`, `momento_angular` ({elemento: "p"}), `scc` |
| `tareas` | En orden; cada una tiene `id` y `tipo`. `sobre` dice de qué geometría parte: `"entrada"` (la de por omisión) o el `id` de una relajación anterior |
| `recursos` | Lo que pide quien solicita; carbonforge lo usa para `mpiexec -n` y el script de SLURM |

Tipos de tarea:

| Tipo | Hace | Devuelve |
|---|---|---|
| `punto_simple` | Energía y fuerzas | `energia`, `fuerzas`, `dipolo` (si el motor lo da), `fermi`, `autovalores` |
| `relajar` | BFGS de posiciones, celda fija; se reanuda desde su trayectoria | Lo mismo de la geometría final, más `estructura` (archivo), `convergida` y `pasos` |
| `desplazamientos` | Geometría ± s·Lᵢ para cada vector, con s tal que el átomo que más se mueve lo haga `amplitud` Å | Por vector y signo: `energia`, `fuerzas`, `dipolo`. Con ellos, quien pidió calcula la frecuencia de cada patrón (cociente de Rayleigh) y dμ/dQ |
| `hessiana` | Diferencias finitas ±`delta` Å por átomo y eje (`atomos`: solo esos; `null`: todos) | `matriz` (`.npy`, eV/Å², filas de los átomos pedidos) |
| `dos` | De los autovalores del cálculo autoconsistente | `energia_menos_fermi` y `total` (estados/eV), `gap` |

## `resultado.json`

```json
{
  "esquema": "nanocarbon-dft/1",
  "codigo": "gpaw", "version_codigo": "25.1.0", "ase": "3.29.0",
  "unidades": {"energia": "eV", "fuerza": "eV/Å", "distancia": "Å", "dipolo": "e·Å"},
  "terminado": true,
  "tareas": {"sp": {…}, "relax": {…}, "modos": {"amplitud": 0.01, "escalas": […],
             "desplazamientos": [{"vector": 0, "signo": 1, "energia": …, "fuerzas": […]}]}}
}
```

Las unidades son siempre eV, Å y e·Å, las de ASE: quien escribe convierte.
`"terminado": false` significa que faltan tareas: se pueden leer las que ya están.

## `progreso.json`

Las mismas claves que `tbkit.progress`: `label`, `done`, `total`, `started`, `updated`, `pid`,
`seconds_per_step`, `remaining_seconds` y `note`. Así, `tbkit estado` y la pestaña Procesos de
tbkit muestran el avance de un cálculo de carbonforge sin conocer carbonforge.

## Reglas

- **Nunca se sobrescribe un resultado.** Cada cálculo deja `tareas/<id>/<n>.json` y se salta si
  ya existe. Una carpeta interrumpida se relanza con el mismo comando.
- **Cambiar la solicitud con resultados ya escritos no tiene efecto.** El conductor guarda en
  `tareas/solicitud_usada.json` lo que corrió y se niega a seguir si `solicitud.json` cambió:
  hay que crear otra carpeta.
- **El conductor (`run_dft.py`) solo importa ASE, numpy y el motor.** Corre donde esté el motor,
  sin tbkit ni carbonforge instalados. Esto incluye una supercomputadora: se copia la carpeta y se
  lanza `trabajo.slurm`.
- **Bajo MPI solo escribe el rango 0.**
