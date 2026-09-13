"""Several heteroatoms at once, each with its own fraction and a say in
whether the species seek each other out or keep apart.

This is not the same operation as doping twice. Two things separate it, and
both were wrong in the sequential version this replaces.

**Every fraction is against the same denominator.** Doping sequentially,
each species' concentration is taken against the carbons *remaining* after
the previous one, so asking a 400-atom sheet for 5% N and 5% B gave 5.00%
and 4.75%. The error compounds with the concentration and with the number
of species, and it is silent: the numbers in ``info`` were the ones asked
for, not the ones placed. Here the counts are worked out together against
the original carbon count, by largest remainder, so they sum exactly and
no site is claimed twice.

**Placement is correlated, because the chemistry is.** Independent random
placement is one physical case -- a dilute, kinetically frozen sample -- and
not the interesting one. In B,N co-doped graphene the two species strongly
prefer to sit next to each other, because a B-N pair is isoelectronic with
a C-C pair and the lattice pays almost nothing for it; that is why real
co-doped samples grow BN domains rather than a solid solution. The opposite
case is just as real: like heteroatoms on adjacent sites are usually
unfavourable, so a sample near equilibrium disperses them.

``affinity`` therefore takes three values, and what each one does is
**measured and recorded** rather than asserted:

``"random"``
    Independent placement. The baseline, and what a sequential double
    doping approximates.
``"seek"``
    Dopants placed as bonded pairs, and where two or more species are
    asked for, the two members of a pair are **different** species. This
    is the BN-domain case.
``"avoid"``
    Dopants placed so that no two are bonded, spread by a greedy
    furthest-first rule over the bond graph. This is the dispersed case.

What this is not: an energy calculation. No term here knows that B-N is
favourable; the affinity is a *placement rule* chosen by the caller to
match the sample they mean. ``info["codoping"]`` records the achieved
heteroatom-heteroatom bond counts so the rule's effect is a number the
caller can check, not a claim they have to take on trust.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from ase import Atoms

from ..utils.rng import make_rng
from .chemistry import get_chemistry
from .substitutional import _carbon_indices, _validate_element, substitute_atoms

#: How the dopants are distributed with respect to one another.
AFFINITIES: tuple[str, ...] = ("random", "seek", "avoid")


def _exact_counts(fractions: Sequence[float], n_sites: int) -> list[int]:
    """Split ``n_sites`` by ``fractions`` so the parts sum as they should.

    Largest remainder, not independent rounding. Rounding each fraction on
    its own loses or gains a site whenever the remainders happen to fall
    the same way -- four species at 0.1 on 95 carbons round to 10 each,
    which is 40 sites for a requested 38.
    """
    raw = [f * n_sites for f in fractions]
    counts = [int(np.floor(value)) for value in raw]
    target = int(round(sum(raw)))
    short = target - sum(counts)
    if short > 0:
        order = np.argsort([-(value - np.floor(value)) for value in raw])
        for k in range(short):
            counts[int(order[k % len(order)])] += 1
    return counts


def _neighbour_table(atoms: Atoms) -> list[list[int]]:
    """Bonded neighbours per atom, from the recorded graph when there is one."""
    from ..functionalize.attach import bond_pairs

    pairs = bond_pairs(atoms)
    table: list[list[int]] = [[] for _ in range(len(atoms))]
    for first, second in pairs:
        table[int(first)].append(int(second))
        table[int(second)].append(int(first))
    return table


def _sites_random(pool: np.ndarray, total: int, rng) -> list[int]:
    return [int(k) for k in rng.choice(pool, size=total, replace=False)]


def _sites_avoiding(pool: np.ndarray, total: int, table: list[list[int]],
                    rng) -> tuple[list[int], int]:
    """Furthest-first: never take a site bonded to one already taken.

    Greedy rather than optimal, and it can run out of room -- at high
    concentration an independent set of the requested size may not exist on
    a trivalent lattice at all. The shortfall is returned rather than
    papered over, because quietly placing the remainder adjacent would
    defeat the only thing the caller asked for.
    """
    order = rng.permutation(pool)
    chosen: list[int] = []
    blocked: set[int] = set()
    for candidate in order:
        site = int(candidate)
        if site in blocked:
            continue
        chosen.append(site)
        blocked.add(site)
        blocked.update(table[site])
        if len(chosen) == total:
            break
    return chosen, total - len(chosen)


def _sites_seeking(pool: np.ndarray, total: int, table: list[list[int]],
                   rng) -> tuple[list[int], list[tuple[int, int]], int]:
    """Bonded pairs first, so unlike species can be put side by side.

    Returns the chosen sites, the pairs among them, and how many sites
    could not be paired. An odd ``total`` leaves one unpaired by
    arithmetic; a crowded lattice can leave more.
    """
    available = set(int(k) for k in pool)
    order = rng.permutation(pool)
    chosen: list[int] = []
    pairs: list[tuple[int, int]] = []
    for candidate in order:
        first = int(candidate)
        if len(chosen) + 2 > total:
            break
        if first not in available:
            continue
        partners = [n for n in table[first] if n in available and n != first]
        if not partners:
            continue
        second = int(partners[int(rng.integers(len(partners)))])
        available.discard(first)
        available.discard(second)
        chosen += [first, second]
        pairs.append((first, second))
    # Whatever is left over goes down singly.
    leftover = total - len(chosen)
    if leftover > 0:
        rest = np.array(sorted(available), dtype=int)
        if len(rest):
            extra = rng.choice(rest, size=min(leftover, len(rest)),
                               replace=False)
            chosen += [int(k) for k in extra]
    return chosen, pairs, total - len(chosen)


def _assign_paired(sites: list[int], pairs: list[tuple[int, int]],
                   elements: list[str], counts: list[int], rng
                   ) -> dict[int, str]:
    """Give each pair two *different* species where the budget allows.

    With one species asked for there is nothing to alternate, and the pairs
    simply become like-like neighbours -- which is the honest answer to
    "seek" for a single element.
    """
    remaining = {element: count for element, count in zip(elements, counts, strict=True)}
    assignment: dict[int, str] = {}

    def take(exclude: str | None = None) -> str | None:
        live = [e for e, c in remaining.items() if c > 0 and e != exclude]
        if not live:
            live = [e for e, c in remaining.items() if c > 0]
        if not live:
            return None
        # Prefer whichever species has most left to place, so one does not
        # run dry early and leave the last pairs like-like by accident.
        live.sort(key=lambda e: -remaining[e])
        top = [e for e in live if remaining[e] == remaining[live[0]]]
        pick = top[int(rng.integers(len(top)))]
        remaining[pick] -= 1
        return pick

    for first, second in pairs:
        one = take()
        if one is None:
            break
        assignment[first] = one
        other = take(exclude=one)
        if other is None:
            break
        assignment[second] = other
    for site in sites:
        if site in assignment:
            continue
        element = take()
        if element is None:
            break
        assignment[site] = element
    return assignment


def _assign_spread(sites: list[int], elements: list[str],
                   counts: list[int], rng) -> dict[int, str]:
    labels: list[str] = []
    for element, count in zip(elements, counts, strict=True):
        labels += [element] * count
    labels = labels[:len(sites)]
    rng.shuffle(labels)
    return dict(zip(sites, labels, strict=True))


def codope(
    atoms: Atoms,
    spec: Sequence[tuple[str, float]],
    affinity: str = "random",
    seed: int | None = None,
) -> Atoms:
    """Substitute several heteroatoms at once, with a correlation rule.

    Parameters
    ----------
    atoms
        Host structure. Only carbons are eligible, so this runs on any
        carbon family -- sheet, tube, cage, schwarzite, haeckelite.
    spec
        ``(element, fraction)`` pairs. Every fraction is of the **original**
        carbon count, so two species at 0.05 give 5% each and not 5% then
        4.75%. Each element is warned against its own ceiling in
        :mod:`.chemistry`, and the fractions must sum to at most 1.
    affinity
        ``"random"``, ``"seek"`` or ``"avoid"`` -- see the module
        docstring. ``"seek"`` pairs unlike species on bonded sites, the
        B-N case; ``"avoid"`` leaves no two dopants bonded.
    seed
        RNG seed. One generator drives the whole placement, so the result
        is reproducible as a whole rather than per species.

    Returns
    -------
    ase.Atoms
        Co-doped copy. ``info["codoping"]`` holds the requested and
        **achieved** fraction per element, the affinity, and the measured
        heteroatom-heteroatom bond counts -- ``dopant_bonds`` and
        ``unlike_bonds`` -- which are what show the affinity did what it
        says. A ``"seek"`` run on graphene at 5% N + 5% B gives tens of
        unlike bonds; ``"avoid"`` gives zero of either.

    Raises
    ------
    ValueError
        For an unknown element or affinity, a non-positive fraction, or
        fractions summing above 1 -- there are only so many carbons.

    Warns
    -----
    When a species exceeds its own ceiling in :mod:`.chemistry`, and when
    ``"avoid"`` cannot place everything: an independent set of the
    requested size need not exist on a trivalent lattice, and the shortfall
    is reported rather than filled in with adjacent sites.
    """
    if affinity not in AFFINITIES:
        raise ValueError(
            f"Unknown affinity {affinity!r}; expected one of {list(AFFINITIES)}."
        )
    if not spec:
        return atoms.copy()

    elements = [element for element, _ in spec]
    fractions = [float(fraction) for _, fraction in spec]
    if len(set(elements)) != len(elements):
        raise ValueError(f"Each element may appear once; got {elements}.")
    for element, fraction in zip(elements, fractions, strict=True):
        _validate_element(element)
        if not 0.0 < fraction <= 1.0:
            raise ValueError(
                f"fraction for {element} must be in (0, 1], got {fraction}."
            )
    total_fraction = sum(fractions)
    if total_fraction > 1.0:
        raise ValueError(
            f"The fractions sum to {total_fraction:.3f}, so more than every "
            "carbon would have to be replaced. They are each of the original "
            "carbon count, not of what the previous species left."
        )
    for element, fraction in zip(elements, fractions, strict=True):
        ceiling = get_chemistry(element).max_fraction
        if fraction > ceiling:
            warnings.warn(
                f"{fraction:.1%} {element} is above the {ceiling:.0%} this "
                f"framework treats as realistic for {element}; building it "
                "anyway.",
                stacklevel=2,
            )

    rng = make_rng(seed)
    carbons = _carbon_indices(atoms)
    counts = _exact_counts(fractions, len(carbons))
    total = sum(counts)
    if total == 0:
        return atoms.copy()

    table = _neighbour_table(atoms)
    shortfall = 0
    pairs: list[tuple[int, int]] = []
    if affinity == "random":
        sites = _sites_random(carbons, total, rng)
    elif affinity == "avoid":
        sites, shortfall = _sites_avoiding(carbons, total, table, rng)
    else:
        sites, pairs, shortfall = _sites_seeking(carbons, total, table, rng)

    if shortfall:
        warnings.warn(
            f"affinity={affinity!r} placed {len(sites)} of {total} dopants: "
            f"{shortfall} had no site left that satisfies the rule. At this "
            "concentration the lattice cannot hold them all apart, and "
            "placing the rest anyway would quietly give up the one property "
            "that was asked for. Lower the fractions or use a larger cell.",
            stacklevel=2,
        )

    if affinity == "seek":
        assignment = _assign_paired(sites, pairs, elements, counts, rng)
    else:
        assignment = _assign_spread(sites, elements, counts, rng)

    out = atoms
    for element in elements:
        chosen = [site for site, value in assignment.items() if value == element]
        if chosen:
            out = substitute_atoms(out, chosen, element)

    symbols = out.get_chemical_symbols()
    dopant_set = set(elements)
    dopant_bonds = 0
    unlike_bonds = 0
    for first, second_list in enumerate(table):
        for second in second_list:
            if second <= first:
                continue
            if symbols[first] in dopant_set and symbols[second] in dopant_set:
                dopant_bonds += 1
                if symbols[first] != symbols[second]:
                    unlike_bonds += 1

    achieved = {element: symbols.count(element) for element in elements}
    out.info["doping_mode"] = "codope"
    out.info["codoping"] = {
        "affinity": affinity,
        "seed": seed,
        "host_carbons": int(len(carbons)),
        "requested": {element: round(fraction, 6)
                      for element, fraction in zip(elements, fractions, strict=True)},
        "achieved": {element: round(achieved[element] / len(carbons), 6)
                     for element in elements},
        "counts": achieved,
        "unplaced": int(shortfall),
        # The affinity's own evidence. `avoid` must give 0 dopant bonds;
        # `seek` gives many, and nearly all of them unlike when two or more
        # species were asked for.
        "dopant_bonds": int(dopant_bonds),
        "unlike_bonds": int(unlike_bonds),
    }
    # Kept for the older single-species readers, which look here.
    out.info["codoping_spec"] = [(element, fraction)
                                 for element, fraction in zip(elements, fractions, strict=True)]
    return out


def describe_codoping(atoms: Atoms) -> str:
    """One line on what was actually placed, not what was asked for."""
    record = atoms.info.get("codoping")
    if not record:
        return "not co-doped."
    parts = ", ".join(
        f"{element} {record['achieved'][element]:.2%}"
        for element in record["counts"])
    line = (f"{record['affinity']} co-doping: {parts} of "
            f"{record['host_carbons']} carbons; "
            f"{record['dopant_bonds']} dopant-dopant bond(s), "
            f"{record['unlike_bonds']} between unlike species")
    if record["unplaced"]:
        line += f"; {record['unplaced']} could not be placed"
    return line + "."


__all__ = ["AFFINITIES", "codope", "describe_codoping"]
