# Montaje de cada celda

Orden de armado y por dónde entra cada electrodo. `checks/walls.py → assembly_access_checks`
verifica lo esencial: lo que entra por un puerto pasa por su agujero, y los tornillos que se
aprietan dentro de la celda quedan al alcance con la celda abierta.

Comunes a todas:
- **RE Gamry:** entra desde afuera, al final. La punta Ø4 pasa por su agujero (Ø4.25) y por el
  O-ring radial interior; el hombro Ø9 apoya en el resalte de la cara: la punta queda 1.5 mm
  dentro de la cavidad. Poner primero el O-ring en su ranura interior (está a 4 mm de la cara).
- **CE de Pt (Ø6):** enrollar antes los 35 mm de alambre en una espiral de Ø5 (sobre una broca
  de 4.5 mm); la espiral pasa por el agujero de Ø6.25 y el cuerpo sella en su O-ring.
- **Conectores 1/4-28:** roscados en agujeros machuelados después de imprimir/maquinar.
- **O-rings en ranuras que miran hacia abajo** (anillo, tapas, marcos): fijarlos con una gota de
  grasa de silicona antes de voltear la pieza.

## Raman, muestra sellada (`base_flat` + `chamber_ring` + `lid_raman`)
1. Tuercas M3 en sus alojamientos de la base (desde abajo); pines guía a presión.
2. Lámina de contacto en su ranura (sale por −x) y la muestra encima, cara arriba.
3. O-ring de la muestra en el escalón del anillo (grasa); bajar el anillo por los pines; 4 × M3×16.
4. RE por −y y CE por +y; conectores de entrada y salida.
5. Llenar con la celda inclinada (salida arriba); lainas en los insertos; tapa con su O-ring; 4 × M3×6.
6. **WE:** la pinza (CS945, double knob o cocodrilo) toma la lámina de contacto **afuera, en seco**.

## Raman con pinza (`chamber_clamp` + tapón + `lid_raman`)
La CS945 no puede apretarse dentro de la celda: con el tornillo abajo no entra el destornillador
(queda 1 mm sobre el piso) y con el tornillo arriba la cabeza choca con la nariz de la tapa.
Por eso el puerto WE es un **portaelectrodo desmontable**:
1. **Afuera:** poner la tira en la ranura de la CS945 y apretar el tornillo (hacia abajo).
2. Pasar el tapón de PEEK por la varilla desde atrás (por el conector dorado), con su O-ring
   interior (Ø6) y el exterior puestos.
3. Meter la cabeza con la tira por el agujero de Ø16.6 de la cara −x hasta que la tira apoye en
   el pedestal; asentar el tapón y fijarlo con 2 × M3 a los insertos de la cara.
4. RE y CE como arriba; conectores; llenar; lainas y tapa Raman.

## XRD (`chamber_xrd` + `lid_xrd`)
1. Cuerpo sin marco ni película. Muestra en su alojamiento, cara arriba; la lengüeta (ver abajo)
   en su ranura bajo el borde +y de la muestra, doblada hacia el bolsillo de la pinza.
2. Quitar el tornillo de la CS945 y pasar la varilla desde afuera por el puerto +y (O-ring Ø6)
   hasta que la ranura reciba la punta de la lengüeta.
3. **Desde arriba, con el bolsillo abierto:** poner el tornillo (cabeza hacia arriba) y apretarlo.
4. Tapa del bolsillo, al ras.
5. RE, CE y conectores por −y. Llenar por la entrada con la salida abierta.
6. Cordón de EPDM en su ranura, película encima, marco y 10 × M3 a los insertos.

### Materiales de la lengüeta y de la "placa"
- **Lengüeta (conector muestra ↔ pinza):** está **mojada** (el bolsillo está lleno), así que debe
  ser inerte en 0.5 M H2SO4 y en 3 M KOH: **lámina de Pt de 25–50 µm**, el mismo metal del
  contacto de la pinza (sin par galvánico). Alternativas: **oro** (bien en ambos; se oxida por
  encima de ~1.3 V vs RHE en ácido) o tu **alambre de Pt de 0.406 mm** con la punta aplanada (la
  ranura bajo la muestra tendría que ser de 0.45 mm). No usar Ti (su óxido corta el contacto y
  absorbe H en HER), Ni (se disuelve en ácido), Cu ni acero. La lengüeta solo toca los últimos
  3 mm del borde +y de la muestra, fuera del haz: no aparece en el difractograma.
- **La "placa" es tu muestra (el WE):** el diseño supone un sustrato plano y conductor de
  18 × 18 × ≤1 mm (papel de carbón, lámina de Ti/Ni, grafito, GC, FTO/ITO). La lengüeta la toca
  por **abajo**: si la muestra solo conduce por arriba (película sobre vidrio), doblar la lengüeta
  sobre el borde, por encima, fuera del corredor del haz.
- **El marco gris** que tensa la película es de **aluminio** (no toca el electrolito: la película
  queda debajo). Cuerpo y tapa del bolsillo: PEEK o PTFE maquinado (PP impreso para pruebas).

## Chladni (`chladni`)
1. Pegar el piezo bajo la placa (Kapton/epoxi entre ambos) y dejar curar.
2. Tuercas en el marco inferior; placa en su alojamiento (la lengüeta sale por +x).
3. Cordón en la ranura de la pared (grasa); bajar la pared; 12 × M3×16, en cruz y con torque parejo.
4. CE: el alambre de Pt (15 cm) en espiral plana sobre el escalón; su extremo sale por el
   conector −x con una funda de PTFE (se pasa de adentro hacia afuera).
5. Llenar; tapa; RE por su resalte desde arriba.
6. **WE:** la pinza (double knob o CS945) toma la lengüeta **afuera, en seco**.

## EPD (`epd`)
1. Chip en el alojamiento de la base (los pads sobresalen 11.5 mm en +x).
2. Junta, vidrio superior y marco (O-rings de los conectores con grasa); 6 × M3 hasta el tope.
3. Conectores de entrada y salida.
4. **Electrodos:** una pinza en cada pad, afuera, en seco.
