# Diccionario del clasificador

> **Generado automáticamente** desde `nanocarbon_biblio/lexicons.py` con
> `python -m nanocarbon_biblio.cli vocab`. No lo edites a mano: cambia el
> léxico y vuelve a generarlo. Última generación: 2026-11-22.

8 facetas · 66 etiquetas · 514 términos.

## Para qué sirve

1. **Al escribir o revisar la consulta** (`queries/scopus.md`): si un
   dopante o un defecto está aquí pero no en la consulta, el clasificador
   nunca podrá etiquetarlo, porque esos registros no se recuperarán.
2. **En Métodos**: es la descripción del esquema de clasificación que un
   revisor va a pedir. Publícalo como material suplementario.

Esta lista **no es la consulta**. La consulta recupera; el léxico describe
lo recuperado. Los términos de aplicación y de método están aquí a
propósito y **no** deben añadirse a los bloques núcleo de la consulta:
arruinarían su precisión.

Los términos son una lectura legible de los patrones, no los patrones
mismos. La clasificación usa siempre las expresiones compiladas del
léxico, con sus guardas de desambiguación.

Las variantes ortográficas se expanden mecánicamente, así que alguna
combinación imposible se cuela (`sul(?:f|p)h?ur` da *sulfur* y *sulphur*,
pero también *sulpur*). No es un error: el patrón acepta las cuatro y la
tabla las lista todas.

## Dopantes — `dopant`

*Faceta de recuperación: sus términos tienen que estar en el bloque B (dopaje) de la consulta.*

| Etiqueta | Términos que reconoce |
|---|---|
| **nitrogen** | `nitrogen doped`, `nitrogen doping`, `nitrogen dopant`, `doped with nitrogen`, `n doped`, `n doping`, `n dopant`, `ncnts`, `n cnts`, `n mwcnts`, `n mwcnt`, `n cnt`, `ncnt`, `pyridinic`, `pyrrolic`, `graphitic nitrogen`, `quaternary nitrogen`, `graphitic n`, `nitrogen incorporation`, `nitrogen substitution`, `nitrogen functionali` |
| **boron** | `boron doped`, `boron doping`, `boron dopant`, `doped with boron`, `b doped`, `boron incorporation`, `boron substitution` |
| **phosphorus** | `phosphorus doped`, `phosphorus doping`, `phosphorus dopant`, `phosphorous doped`, `phosphorous doping`, `phosphorous dopant`, `doped with phosphorus`, `doped with phosphorous`, `p doped` |
| **sulfur** | `sulfhur doped`, `sulfhur doping`, `sulfhur dopant`, `sulfur doped`, `sulfur doping`, `sulfur dopant`, `sulphur doped`, `sulphur doping`, `sulphur dopant`, `sulpur doped`, `sulpur doping`, `sulpur dopant`, `doped with sulfhur`, `doped with sulfur`, `doped with sulphur`, `doped with sulpur`, `s doped` |
| **fluorine** | `fluorine doped`, `fluorine doping`, `fluorine dopant`, `fluorinated doped`, `fluorinated doping`, `fluorinated dopant`, `f doped` |
| **halogen_other** | `chlorine doped`, `chlorine doping`, `chlorine dopant`, `bromine doped`, `bromine doping`, `bromine dopant`, `iodine doped`, `iodine doping`, `iodine dopant`, `cl doped`, `br doped`, `i doped` |
| **silicon** | `silicon doped`, `silicon doping`, `silicon dopant`, `si doped` |
| **oxygen** | `oxygen doped`, `oxygen doping`, `oxygen dopant`, `o doped` |
| **selenium** | `selenium doped`, `selenium doping`, `selenium dopant`, `se doped` |
| **transition_metal** | `fe doped`, `co doped`, `ni doped`, `mn doped`, `cu doped`, `zn doped`, `mo doped`, `w doped`, `pt doped`, `pd doped`, `ru doped`, `ir doped`, `ag doped`, `au doped`, `transition metal dop`, `single atom catalyst`, `single atom site`, `fe n`, `co n`, `ni n` |
| **codoped** | `co doped`, `co doping`, `co dopant`, `dual doped`, `dual doping`, `ternary doped`, `ternary doping`, `tri doped`, `tri doping` |

## Tipos de defecto — `defect`

*Faceta de recuperación: sus términos tienen que estar en el bloque B (defectos) de la consulta.*

| Etiqueta | Términos que reconoce |
|---|---|
| **vacancy** | `vacancy`, `vacancies`, `mono vacanc`, `di vacanc`, `single vacanc`, `double vacanc`, `multi vacanc` |
| **stone_wales** | `stone wales`, `sw defect`, `pentagon heptagon`, `5-7 pair`, `5-7 defect` |
| **topological** | `topological defect`, `pentagonal`, `heptagonal`, `octagonal`, `octagon`, `heptagon`, `pentagon`, `non hexagonal`, `ring defect`, `ring statistic` |
| **grain_boundary** | `grain boundar` |
| **edge** | `edge state`, `edge site`, `edge defect`, `edge atom`, `edge carbon`, `armchair edge`, `zigzag edge`, `open ended`, `open end`, `open tiped`, `open tip` |
| **sp3** | `sp3 hybrid`, `sp3 defect`, `sp3 carbon site`, `sp3` |
| **adatom_interstitial** | `adatom`, `interstitial` |
| **dangling_bond** | `dangling bond` |
| **disorder_amorphous** | `amorphous carbon`, `structural disorder`, `lattice disorder`, `disordered carbon` |
| **irradiation_induced** | `ion irradiat`, `electron irradiat`, `proton irradiat`, `neutron irradiat`, `gamma irradiat`, `ion implantation` |
| **defect_engineering** | `defect engineer` |
| **defect_free** | `defect free`, `pristine` |

## Modo de dopaje — `doping_mode`

| Etiqueta | Términos que reconoce |
|---|---|
| **substitutional** | `substitutional`, `in situ dop`, `incorporatw into the carbon lattice`, `incorporatw into the lattice`, `incorporatw in the carbon lattice`, `incorporatw in the lattice`, `substitutw into the carbon lattice`, `substitutw into the lattice`, `substitutw in the carbon lattice`, `substitutw in the lattice`, `graphitic nitrogen`, `pyridinic`, `pyrrolic`, `lattice substitution`, `lattice incorporation` |
| **charge_transfer** | `charge transfer dop`, `n type dop`, `p type dop`, `hno3 dop`, `nitric acid dop`, `socl2 dop`, `aucl3 dop`, `fecl3 dop`, `h2so4 dop`, `molecular dop`, `intercalat` |
| **functionalization** | `covalent functionali`, `non covalent functionali`, `surface functionali`, `side wall functionali`, `grafted`, `grafting`, `acid treatment`, `carboxylated`, `carboxylation` |

## Métodos teóricos y computacionales — `method_theory`

| Etiqueta | Términos que reconoce |
|---|---|
| **dft** | `density functional theor`, `dft`, `first principle`, `ab initio`, `vasp`, `siesta`, `quantum espresso`, `castep`, `cp2k`, `gpaw`, `abinit`, `formation energ`, `pbe`, `gga`, `lda`, `hse06`, `b3lyp` |
| **molecular_dynamics** | `molecular dynamic`, `reaxff`, `airebo`, `lammps`, `tersoff`, `brenner potential`, `classical simulation` |
| **tight_binding** | `tight binding` |
| **transport_theory** | `non equilibrium green`, `negf`, `landauer`, `boltzmann transport` |
| **many_body** | `many body perturbation`, `bethe salpeter`, `gw approximation` |
| **monte_carlo** | `kinetic monte carlo`, `monte carlo` |
| **machine_learning** | `machine learning potential`, `machine learning interatomic`, `machine learned potential`, `machine learned interatomic`, `neural network potential`, `machine learning`, `high throughput screening` |

## Métodos experimentales y de caracterización — `method_experiment`

| Etiqueta | Términos que reconoce |
|---|---|
| **synthesis_cvd** | `chemical vap ur deposition`, `chemical vap u deposition`, `cvd`, `floating catalyst`, `aerosol assisted`, `pecvd` |
| **synthesis_other** | `arc discharge`, `laser ablation`, `electrospinning`, `electrospining`, `pyrolysis`, `carboniz`, `were synthesi`, `were prepar`, `were fabricat`, `were grown`, `were produced`, `were obtained`, `was synthesi`, `was prepar`, `was fabricat`, `was grown`, `was produced`, `was obtained`, `is synthesi`, `is prepar`, `is fabricat`, `is grown`, `is produced`, `is obtained`, `are synthesi`, `are prepar`, `are fabricat`, `are grown`, `are produced`, `are obtained`, `we synthesi`, `we prepar`, `we fabricat`, `we report the synthesis` |
| **raman** | `raman spectroscop`, `raman spectra`, `raman scattering`, `raman analysis`, `id/ig`, `d band`, `g band`, `2d band` |
| **xps** | `x ray photoelectron`, `xps` |
| **electron_microscopy** | `transmission electron microscop`, `hrtem`, `tem`, `stem`, `haadf`, `sem`, `electron energy loss`, `eels` |
| **scanning_probe** | `scanning tunnelling microscop`, `scanning tunnelling spectroscop`, `scanning tunneling microscop`, `scanning tunneling spectroscop`, `scanning tunelling microscop`, `scanning tunelling spectroscop`, `scanning tuneling microscop`, `scanning tuneling spectroscop`, `stm`, `sts`, `afm` |
| **xray_absorption** | `xanes`, `nexafs`, `exafs`, `xas`, `near edge x ray absorption`, `synchrotron` |
| **spin_resonance** | `electron paramagnetic resonance`, `electron spin resonance`, `epr`, `esr` |
| **electrochemistry** | `cyclic voltammetr`, `rotating disk electrode`, `rde`, `rrde`, `rotating ring disk electrode`, `electrochemical impedance`, `eis`, `tafel slope` |

## Morfología y escala — `morphology`

| Etiqueta | Términos que reconoce |
|---|---|
| **forest_vacnt** | `nanotube forest`, `cnt forest`, `vertically aligned`, `vacnts`, `va cnts`, `va cnt`, `vacnt` |
| **fiber_yarn** | `nanotube fiber`, `nanotube yarn`, `nanotube thread`, `nanotube filament`, `nanotube fibre`, `cnt fiber`, `cnt yarn`, `cnt thread`, `cnt filament`, `cnt fibre`, `wet spun`, `dry spun`, `spinning of`, `macroscopic fiber`, `macroscopic fibre` |
| **film_buckypaper** | `bucky paper`, `free standing film`, `free standing membrane`, `free standing paper`, `nanotube film`, `nanotube membrane`, `nanotube mat`, `cnt film`, `cnt membrane`, `cnt mat` |
| **sponge_aerogel_foam** | `nanotube sponge`, `nanotube aerogel`, `nanotube foam`, `nanotube monolith`, `cnt sponge`, `cnt aerogel`, `cnt foam`, `cnt monolith`, `carbon sponge`, `carbon aerogel`, `carbon foam`, `carbon monolith`, `aerogel`, `sponge`, `xerogel`, `3d network`, `3d architecture`, `3d framework`, `3d scaffold`, `three dimensional network`, `three dimensional architecture`, `three dimensional framework`, `three dimensional scaffold` |
| **array_bundle** | `nanotube array`, `nanotube bundle`, `nanotube rope`, `cnt array`, `cnt bundle`, `cnt rope`, `aligned nanotube`, `aligned cnt` |
| **network_junction** | `nanotube network`, `nanotube junction`, `cnt network`, `cnt junction`, `y junction`, `branched nanotube`, `percolat` |
| **composite** | `polymer composite`, `polymer matrix`, `polymer nanocomposite`, `epoxy composite`, `epoxy matrix`, `epoxy nanocomposite`, `ceramic composite`, `ceramic matrix`, `ceramic nanocomposite`, `cement composite`, `cement matrix`, `cement nanocomposite`, `metal matrix composite`, `metal matrix matrix`, `metal matrix nanocomposite`, `nanocomposite` |

## Aplicaciones — `application`

| Etiqueta | Términos que reconoce |
|---|---|
| **orr_fuelcell** | `oxygen reduction reaction`, `orr`, `fuel cell`, `metal free electrocatalys`, `metal free catalys` |
| **her_oer_water** | `hydrogen evolution`, `oxygen evolution`, `her`, `oer`, `water splitting`, `water electrolysis` |
| **co2_n2_reduction** | `co2 reduction`, `co2rr`, `co2 electroreduction`, `nitrogen reduction reaction`, `nrr` |
| **supercapacitor** | `supercapacitor`, `electrochemical double layer capacitor`, `electrochemical capacitor`, `pseudocapacit` |
| **battery** | `lithium ion`, `sodium ion`, `potassium ion`, `lithium sulfur`, `li s batter`, `metal air batter`, `zinc air batter`, `anode material`, `cathode material` |
| **sensor** | `gas sens`, `chemical sens`, `bio sens`, `electrochemical sens`, `biosensor`, `chemiresistor`, `detection of` |
| **field_emission** | `field emission`, `emitter` |
| **electronic_device** | `field effect transistor`, `fet`, `transistor`, `interconnect`, `logic device` |
| **thermal_thermoelectric** | `thermoelectric`, `thermal conductivity`, `phonon transport`, `seebeck` |
| **mechanical_composite** | `mechanical propert`, `mechanical strength`, `mechanical reinforcement`, `tensile propert`, `tensile strength`, `tensile reinforcement`, `young's modulus`, `fracture`, `stiffness` |
| **adsorption_separation** | `adsorption`, `adsorbent`, `co2 capture`, `gas separation`, `water treatment`, `water purification`, `water desalination`, `capacitive deionization` |
| **hydrogen_storage** | `hydrogen storage`, `hydrogen uptake` |
| **photocatalysis** | `photocataly` |
| **emi_microwave** | `emi shielding`, `electromagnetic interference`, `microwave absor` |
| **biomedical** | `drug delivery`, `bioimaging`, `tissue engineering`, `tissue scaffold`, `cytotoxic`, `biocompatib` |

## Material huésped — `host`

*Faceta de recuperación: sus términos tienen que estar en el bloque A (nanoestructura) de la consulta.*

| Etiqueta | Términos que reconoce |
|---|---|
| **carbon_1d** | `carbon nanotube`, `carbon nanofiber`, `carbon nanocoil`, `carbon nanofilament`, `carbon nanofibre`, `swcnts`, `swcnt`, `swnts`, `swnt`, `mwcnts`, `mwcnt`, `mwnts`, `mwnt`, `dwcnts`, `dwcnt`, `dwnts`, `dwnt` |
| **graphene_2d** | `graphene`, `graphene oxide`, `rgo`, `graphene nanoribbon` |

## Cobertura de la consulta

¿Puede la consulta recuperar lo que el clasificador sabe etiquetar?

| Faceta | Etiqueta | ¿Cubierta? | Términos encontrados en la consulta |
|---|---|---|---|
| host | carbon_1d | sí | `carbon nanotube`, `carbon nanofiber`, `carbon nanocoil`, `carbon nanofilament`, `carbon nanofibre` |
| host | graphene_2d | por diseño | `graphene`, `graphene oxide`, `rgo`, `graphene nanoribbon` |
| dopant | nitrogen | sí | `nitrogen doped`, `nitrogen doping`, `nitrogen dopant`, `doped with nitrogen`, `n doped` |
| dopant | boron | sí | `boron doped`, `boron doping`, `boron dopant`, `doped with boron`, `b doped` |
| dopant | phosphorus | sí | `phosphorus doped`, `phosphorus doping`, `phosphorus dopant`, `phosphorous doped`, `phosphorous doping` |
| dopant | sulfur | sí | `sulfur doped`, `sulfur doping`, `sulfur dopant`, `sulphur doped`, `sulphur doping` |
| dopant | fluorine | sí | `fluorine doped`, `fluorine doping`, `fluorine dopant`, `f doped` |
| dopant | halogen_other | sí | `chlorine doped`, `chlorine doping`, `chlorine dopant`, `bromine doped`, `bromine doping` |
| dopant | silicon | sí | `silicon doped`, `silicon doping`, `silicon dopant`, `si doped` |
| dopant | oxygen | sí | `oxygen doped`, `oxygen doping`, `oxygen dopant`, `o doped` |
| dopant | selenium | sí | `selenium doped`, `selenium doping`, `selenium dopant`, `se doped` |
| dopant | transition_metal | sí | `fe doped`, `co doped`, `ni doped`, `mn doped`, `cu doped` |
| dopant | codoped | sí | `co doped`, `co doping`, `co dopant`, `dual doped`, `dual doping` |
| defect | vacancy | sí | `vacancy`, `vacancies`, `di vacanc` |
| defect | stone_wales | sí | `stone wales`, `sw defect`, `pentagon heptagon`, `5-7 defect` |
| defect | topological | sí | `topological defect`, `heptagon`, `pentagon` |
| defect | grain_boundary | sí | `grain boundar` |
| defect | edge | sí | `edge site`, `edge defect`, `edge carbon` |
| defect | sp3 | sí | `sp3 defect`, `sp3 carbon site`, `sp3` |
| defect | adatom_interstitial | sí | `adatom`, `interstitial` |
| defect | dangling_bond | sí | `dangling bond` |
| defect | disorder_amorphous | sí | `disordered carbon` |
| defect | irradiation_induced | sí | `ion irradiat`, `electron irradiat`, `ion implantation` |
| defect | defect_engineering | sí | `defect engineer` |
| defect | defect_free | sí | `defect free` |

Todas las etiquetas de recuperación están cubiertas, o declaradas ausentes por diseño.

> **host/graphene_2d** — El grafeno queda fuera del corpus núcleo por decisión de alcance (docs/PROTOCOL.md §2.1). La etiqueta existe para marcar híbridos DENTRO de registros de CNT ya recuperados, no para recuperarlos.

