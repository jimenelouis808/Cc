"""GPAW reference data for boron, sulfur and phosphorus: ``xu_chnob``, ``xu_chnos``, ``xu_chnop``.

Run (needs GPAW; about an hour per element on four cores)::

    python -m tbkit.recipes.bsp_references B gpaw_b.json --workers 4

Same protocol as :mod:`tbkit.recipes.chn_references` (PBE, LCAO dzp; each
training molecule relaxed, then four random displacements and two uniform
scalings; test molecules relaxed only). Each element gets molecules for every
bond it makes with H, C, N, O and itself -- the pairs its parameter set must
define -- chosen for the groups found in doped and functionalised carbon:

* B: borane, trimethylborane, boric acid and trimethyl borate (B-O),
  methylboronic acid, ammonia-borane and borazine (B-N), vinylborane (B-C π),
  tetrahydroxydiboron (B-B). Test: phenylboronic acid, triethylborane and a
  B-N pair in coronene's central ring (BN co-doped graphene; a lone B would be
  open-shell).
* S: H₂S, methanethiol, dimethyl sulfide and disulfide (S-S), thiophene,
  thioformaldehyde, CS₂, OCS, DMSO, dimethyl sulfone, SO₂, methanesulfonic
  acid and methanesulfonamide (S-N). Test: ethanethiol, thiophenol,
  benzenesulfonic acid, thiol on coronene's edge.
* P: PH₃, methyl- and trimethylphosphine, trimethylphosphine oxide,
  phosphoric and methylphosphonic acid, trimethyl phosphate, phosphinine
  (aromatic P), diphosphane (P-P), aminophosphine (P-N). Test: phenylphosphine,
  triethylphosphine, phosphonic acid on coronene's edge.

Starting geometries are in ``data/bsp_start.extxyz`` (RDKit ETKDG, seed 1,
then UFF; regenerate with :func:`write_start_geometries`, which needs RDKit,
not a dependency of tbkit). GPAW relaxes every one of them.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from ase import Atoms

DATA = Path(__file__).with_name("data") / "bsp_start.extxyz"

#: name -> SMILES, per element: (training, test). Coronene motifs are built
#: from :func:`tbkit.recipes.chno_references.coronene` where SMILES cannot
#: express them (the in-plane B-N pair).
CORONENE = "c1cc2ccc3ccc4ccc5ccc6ccc1c7c2c3c4c5c67"
MOLECULES = {
    "B": ({"BH3": "B", "BMe3": "CB(C)C", "B(OH)3": "OB(O)O", "B(OMe)3": "COB(OC)OC",
           "MeB(OH)2": "CB(O)O", "H3BNH3": "[BH3-][NH3+]", "borazine": "B1NBNBN1",
           "vinylborane": "C=CB", "B2(OH)4": "OB(O)B(O)O"},
          {"PhB(OH)2": "OB(O)c1ccccc1", "BEt3": "CCB(CC)CC", "coronene_BN": None}),
    "S": ({"H2S": "S", "CH3SH": "CS", "CH3SCH3": "CSC", "CH3SSCH3": "CSSC",
           "thiophene": "c1ccsc1", "H2CS": "C=S", "CS2": "S=C=S", "OCS": "O=C=S",
           "DMSO": "CS(C)=O", "Me2SO2": "CS(C)(=O)=O", "SO2": "O=S=O",
           "CH3SO3H": "CS(=O)(=O)O", "CH3SO2NH2": "CS(N)(=O)=O"},
          {"CH3CH2SH": "CCS", "PhSH": "Sc1ccccc1", "PhSO3H": "OS(=O)(=O)c1ccccc1",
           "coronene_SH": "S" + CORONENE}),
    "P": ({"PH3": "P", "CH3PH2": "CP", "PMe3": "CP(C)C", "OPMe3": "CP(C)(C)=O",
           "H3PO4": "OP(O)(O)=O", "MePO3H2": "CP(=O)(O)O", "PO(OMe)3": "COP(=O)(OC)OC",
           "phosphinine": "c1ccpcc1", "P2H4": "PP", "H2PNH2": "NP"},
          {"PhPH2": "Pc1ccccc1", "PEt3": "CCP(CC)CC",
           "coronene_PO3H2": "OP(O)(=O)" + CORONENE}),
    "Se": ({"H2Se": "[SeH2]", "CH3SeH": "C[SeH]", "CH3SeCH3": "C[Se]C",
            "CH3SeSeCH3": "C[Se][Se]C", "selenophene": "c1cc[se]c1", "CSe2": "[Se]=C=[Se]",
            "OCSe": "O=C=[Se]", "H2CSe": "C=[Se]", "SeO2": "O=[Se]=O", "DMSeO": "C[Se](C)=O",
            "CH3SeO2H": "C[Se](=O)O", "CH3SeNH2": "C[Se]N"},
           {"CH3CH2SeH": "CC[SeH]", "PhSeH": "[SeH]c1ccccc1",
            "coronene_SeH": "[SeH]" + CORONENE}),
}


def coronene_bn() -> Atoms:
    """Coronene with two bonded carbons of the central ring replaced by B and N."""
    from .chno_references import _central_ring, coronene

    atoms = coronene()
    ring = _central_ring(atoms)
    a = ring[0]
    b = min(ring[1:], key=lambda j: np.linalg.norm(atoms.positions[j] - atoms.positions[a]))
    symbols = atoms.get_chemical_symbols()
    symbols[a], symbols[b] = "B", "N"
    atoms.set_chemical_symbols(symbols)
    return atoms


def write_start_geometries(path: Path = DATA) -> None:
    """Embed every SMILES with RDKit (ETKDG, seed 1) and relax with UFF."""
    from ase.io import write
    from rdkit import Chem
    from rdkit.Chem import AllChem

    frames = []
    for element, (train, test) in MOLECULES.items():
        for name, smiles in {**train, **test}.items():
            if smiles is None:
                continue
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            if AllChem.EmbedMolecule(mol, randomSeed=1) != 0:
                raise RuntimeError(f"{name}: RDKit no generó geometría")
            AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
            atoms = Atoms([a.GetSymbol() for a in mol.GetAtoms()],
                          positions=mol.GetConformer().GetPositions())
            atoms.info["name"] = name
            atoms.info["smiles"] = smiles
            frames.append(atoms)
    path.parent.mkdir(exist_ok=True)
    write(path, frames, format="extxyz")


def structure(name: str) -> Atoms:
    from ase.io import read

    if name == "coronene_BN":
        return coronene_bn()
    for atoms in read(DATA, index=":", format="extxyz"):
        if atoms.info["name"] == name:
            atoms = Atoms(atoms.get_chemical_symbols(), positions=atoms.get_positions())
            return atoms
    raise KeyError(name)


def _one(args):
    name, role, index, settings = args
    from tbkit.references import generate_gpaw

    kwargs = {} if role == "train" else {"n_random": 0, "scales": ()}
    items = generate_gpaw({name: structure(name)}, settings, role=role, seed=2000 + 100 * index,
                          **kwargs)
    return name, [item.to_dict() for item in items]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("element", choices=sorted(MOLECULES))
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    from tbkit.references import GPAW_DEFAULTS, gpaw_settings_record

    settings = dict(GPAW_DEFAULTS)
    train, test = MOLECULES[args.element]
    jobs = [(n, "train", k, settings) for k, n in enumerate(train)]
    jobs += [(n, "test", 50 + k, settings) for k, n in enumerate(test)]
    jobs.sort(key=lambda job: -len(structure(job[0])))
    parts = args.out.with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)
    todo = [job for job in jobs if not (parts / f"{job[0]}.json").exists()]
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(_one, job) for job in todo]
        for future in as_completed(futures):
            name, result = future.result()
            (parts / f"{name}.json").write_text(json.dumps(result))
            print(f"{name}: {len(result)} estructuras", flush=True)
    structures = []
    for job in jobs:
        structures += json.loads((parts / f"{job[0]}.json").read_text())
    data = {"settings": gpaw_settings_record(settings), "structures": structures}
    args.out.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(structures)} estructuras en {args.out}")


if __name__ == "__main__":
    main()
