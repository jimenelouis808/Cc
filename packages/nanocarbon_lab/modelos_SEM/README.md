# Modelos estructurales para las morfologías SEM de FeSe@C

Construidos con `nanocarbon_lab` a partir de los tres CIF de `cif/`
(procedencia en `cif/PROCEDENCIA.md`).

```bash
uv run python modelos_SEM/construir.py
```

Todo sale en `xyz/` (coordenadas) e `img/` (dos vistas por modelo: una
oblicua y otra por el eje, con sufijo `_eje`).

## Lo primero: qué es una hipótesis aquí y qué no

**Cuál fase es cuál sigue sin decidirse, y estos modelos no lo deciden.**
Lo que hacen es ponerlas una al lado de la otra con el hábito que cada
grupo espacial da de forma natural, para que la comparación con el SEM
sea una comparación y no un recuerdo:

- la **placa hexagonal** se corta del FeSe hexagonal (P6₃/mmc, tipo
  NiAs). Un prisma hexagonal es el hábito que esa simetría regala, y es
  exactamente tu razonamiento — queda escrito como lo que es, una
  sospecha hecha geometría.
- la **barra** se corta del FeSe tetragonal (P4/nmm, tipo PbO).

Nada en el modelo prueba la asignación. **Lo que la zanja es la
difracción**: la 101 del tetragonal está a 28.6° y la del hexagonal a
32.4°, catorce veces la anchura de un pico. Si tienes DRX de la muestra
de 900 °C, eso lo cierra en una tarde; el SEM solo lo sugiere.

## Los modelos

| modelo | átomos | fórmula |
|---|---|---|
| `placa_H_nucleo_desnudo` | 1626 | Fe813Se813 |
| `placa_H_C` | 4976 | C3350Fe813Se813 |
| `placa_H_NSe` | 4976 | C3065Fe813N201Se897 |
| `placa_H_NSeO` | 5014 | C3068H15Fe813N204O13Se901 |
| `placa_H_delgada_desnuda` | 1086 | Fe546Se540 |
| `barra_T_nucleo_desnudo` | 192 | Fe96Se96 |
| `barra_T_C` | 982 | C790Fe96Se96 |
| `barra_T_NSe` | 982 | C723Fe96N47Se116 |
| `barra_T_NSeO` | 992 | C724H4Fe96N48O4Se116 |
| `particula_Fe3C_nucleo_desnudo` | 1100 | C292Fe808 |
| `particula_Fe3C_C` | 4250 | C3442Fe808 |
| `particula_Fe3C_NSe` | 4250 | C3174Fe808N189Se79 |
| `particula_Fe3C_NSeO` | 4280 | C3176H12Fe808N191O10Se83 |
| `particula_Fe3C_interior` | 126 | C36Fe90 |
| `nanotubo_C` | 2160 | C2160 |
| `nanotubo_NSe` | 2160 | C1976N130Se54 |
| `nanotubo_NSeO` | 2185 | C1978H10N132O9Se56 |
| `brazo_particula_dentro` | 2286 | C2196Fe90 |
| `escena_anemona` | **22 860** | C21960Fe900 |

### Las tres versiones

Cada objeto de carbono viene en tres, y **la geometría de abajo es la
misma en las tres**: el núcleo se corta una vez y la coraza se malla una
vez, así que lo único que cambia entre dos versiones es la química. Si se
construyeran por separado, el mallado diferiría y la comparación no
mediría nada.

| sufijo | qué lleva |
|---|---|
| `_C` | carbono puro. El control. |
| `_NSe` | coraza codopada N + Se, sustitucional, **en una sola pasada** |
| `_NSeO` | lo anterior más grupos de superficie O/N/Se en la cara externa |

**Por qué una sola pasada**: las dos fracciones son del mismo
denominador. Sustituir N y luego Se haría que la segunda fracción fuera
una fracción de lo que dejó la primera.

**Las fracciones**: N al 6 %, Se al 2.5 %. No son libres — el catálogo de
`dopants/chemistry.py` pone el techo del Se en el 3 % («más mismatch que
el S y correspondientemente más raro») y el del N en el 20 %. El Se es el
que manda y 2.5 % está justo debajo de su techo. Correlación `avoid`,
así que no quedan dos heteroátomos enlazados: N y Se no son una pareja
dador-aceptor como B–N, no hay razón para que se junten, y juntarlos
concentraría la tensión que el Se ya trae (es un 48 % más grande que el
carbono y saca su sitio del plano).

**Los grupos**, en el orden en que se injertan y con su cobertura:
hidroxilo 5 %, carboxilo 3 %, carbonilo 3 %, amina 3 %, **selenol** 2 %,
**selenona** 2 %. Cada pasada ve los sitios que dejó la anterior, así que
la cobertura cae sola — que es lo que hace un carbono oxidado de verdad,
unos pocos por ciento y no una monocapa.

**La selenona y el selenol no están en el catálogo**, y no hacía falta
añadirlos: `functionalize.groups.substitute` cambia el elemento y
**reconstruye las longitudes de enlace**, porque la biblioteca solo
guarda ángulos. El selenol es el hidroxilo con O→Se y la selenona es el
tiocarbonilo con S→Se, así que el C=Se sale con la longitud del selenio y
no con la del azufre.

## Lo que el programa avisó, y que hay que leer antes de usar esto

**1. La interfase Fe–C no está decidida por nadie.** El aviso sale en los
tres núcleos con hierro:

> el núcleo contiene Fe, que forma carburos estables: el carbono sobre
> ellos quimisorbe a unos 2.1 Å en vez de quedarse en el contacto de van
> der Waals de 3.45 Å. Las dos estructuras existen —el contacto es lo
> que una coraza grafítica tiene en la mayor parte de su área— pero son
> interfases distintas y nada aquí calcula cuál prefiere tu sistema.

Estos modelos usan el contacto de van der Waals. **Para tu pregunta de
si la partícula está dentro o fuera del tubo eso importa**: una partícula
que nucleó el tubo y quedó en su punta estuvo en contacto quimisorbido
mientras crecía.

**2. El Fe₃C sale 6.2 % fuera de estequiometría** y el programa lo llama
`strained`. No es un fallo del corte: el carbono de la cementita es
intersticial y está más adentro, así que cualquier superficie que cortes
termina en hierro. Baja al 5 % si agrandas la partícula y nunca llega a
cero. El aviso habla de «cluster cargado», que es el caso iónico; la
cementita es metálica, así que ahí la advertencia no aplica tal cual —
pero el desbalance sí es real y hay que declararlo si publicas una
composición.

**3. La estequiometría del FeSe se forzó quitando átomos.** 92 en la
placa. «Los menos coordinados» es la receta habitual y es una heurística:
menos enlazado por número de vecinos no es menos enlazado por energía.

**4. En la placa, unos pocos carbonos quedan a 1.68 Å del núcleo.** El
percentil cinco está en 2.33 Å, así que son unos pocos átomos y no la
pared: es el borde afilado del prisma hexagonal. Un borde real se
reconstruye ahí y este modelo no. Ya subí el `wall_anchor` de 0.7 a 1.0
por eso; si te molesta, lo honrado es redondear el canto del núcleo, no
dejar que la pared se desplome.

**5. El hexagonal es deficiente en hierro en la vida real** (Fe₁₋ₓSe, y a
composiciones concretas ordena como Fe₇Se₈ o Fe₃Se₄ con superestructura).
Estos modelos lo cortan 1:1, que es la celda ideal.

## La anémona, y lo que cambió en la segunda versión

| | v1 | v2 |
|---|---|---|
| átomos | 32 730 | **22 860** |
| brazos | 5 | **10** |
| brazos | rectos | **curvados** |
| partícula | fuera, pegada a la punta | **dentro de la punta** |
| radio del tubo | 7.85 Å | **11.76 Å** |

**La partícula dentro obligó a ensanchar el tubo, no a encoger la
partícula**, y esa dirección es un resultado: por debajo de unos 126
átomos una partícula de cementita deja de ser cementita — 70 átomos
vuelve `broken` y 36 también — así que la pieza pequeña es la que no se
puede encoger. Un tubo de 7.85 Å de radio deja 4.5 Å de hueco y la
partícula más pequeña que sigue siendo cementita llega a 6.57 Å del eje.
Con `freq = 6` el tubo tiene 11.76 Å de radio y 8.41 Å de hueco, y la
holgura medida es **1.84 Å**. El ensamblaje la mide y se niega si no
cabe, en vez de meterla igual.

**Los brazos se curvan con `shape="arc"`, no con `bend_angle`.** El
segundo impone el arco después de relajar y lo paga en tensión
(1.372–1.529 Å de enlace a 0.6 rad); el primero pasa por el presupuesto
de deformación de `centerline.py`, que recorta la amplitud hasta que la
pared aguanta, y vuelve en **1.393–1.447 Å**. La curva es gratis si se
pide por el camino correcto.

**Y arrancan a cuatro radios del centro.** A uno, cuatro pares de brazos
quedaban a 2.4–3.1 Å, que en un render se lee como fundidos. Van
repartidos sobre la esfera por el método de Fibonacci y no en un plano:
una anémona no es una flor.

Sigue siendo una **ilustración**, y el programa lo dice en cada una:

> esto es una ILUSTRACIÓN, no una estructura: los objetos se colocaron a
> mano, no hay ninguna interfase relajada entre dos de ellos, y nada en
> los huecos es física. No lo cuantifiques, no lo relajes y no se lo des
> a un código DFT.

## La placa: 8.9 Å, y el techo lo pone la coraza

| grosor | ancho | relación | qué pasa |
|---|---|---|---|
| 3.0 Å (media celda) | 45 Å | 15:1 | **no es FeSe**: termina Se–Se, sale Fe110Se200 |
| 6.0 Å (una celda) | 58 Å | 9.8:1 | el núcleo está bien; **la coraza se rompe** |
| **8.9 Å** | **58 Å** | **6.5:1** | **lo que se entrega** |
| 11.9 Å (dos celdas) | 58 Å | 4.9:1 | la primera versión |

**Lo que impide bajar de 8.9 Å no es el cristal, es el recubrimiento.**
A 6.0 Å la coraza grafítica vuelve rota con todos los huecos y rigideces
probados —enlaces hasta 1.73–1.76 Å contra el techo sp² de 1.55— y el
mensaje del propio constructor dice por qué: *«la superficie tiene
detalles más finos que un anillo de carbono»*. Un canto de 6 Å es más
fino que la pared que tendría que envolverlo. En una muestra real ese
borde es donde el recubrimiento está defectuoso o simplemente abierto.

Por eso va también `placa_H_delgada_desnuda`: 6.0 Å de grosor y 58 Å de
ancho, **9.8:1**, sin cubrir. Es la más delgada que el cristal permite y
la que hay que comparar con la micrografía.

## El render de portada

`modelos_SEM/renderizar.sh` produce las imágenes de `render/`. Tres
banderas hacen el aspecto que pediste:

```
--model auto            metales en relleno de van der Waals,
                        carbono en bolas y palos
--spacefill-element Se  ...y el calcógeno también
--atom-scale 0.55 --bond-scale 0.5    bolas y palos finos
```

**`--spacefill-element` es nuevo en el paquete y hacía falta.** El modelo
`auto` pregunta por el elemento y su tabla son **metales**, porque un
metal no tiene enlaces direccionales que dibujar. Un núcleo de FeSe
también es un sólido y el selenio no es un metal, así que salía abierto y
con palos hacia el hierro. Ahora se puede declarar por render.

**No se declara en las versiones dopadas**, y esa asimetría es el punto:
ahí el selenio está en dos sitios a la vez —en el núcleo de FeSe y
sustituido dentro del carbono— y **ninguna regla por elemento puede
distinguirlos**. En `placa_H_NSeO` se queda abierto, que es lo correcto
para el selenio del grafito y lo incorrecto para el del núcleo. Si hace
falta la portada con el núcleo sólido *y* el dopaje visible, el camino es
colorear por estructura y no por elemento.

El grafito sale más liso que en la primera versión porque la coraza se
malla y se relaja mucho más fino: `grid_resolution` 80 → **128**,
`remesh_iterations` 25 → **60**, `relax_iterations` 3000 → **8000**, y
`roughness` explícitamente a 0. Cuesta tiempo, no fidelidad.

**Blender no está en este contenedor como aplicación**; el render corre
con la rueda `bpy` en un entorno aparte, porque `bpy` fija numpy 1.26 y
el workspace necesita numpy 2 —el `pyproject.toml` raíz declara ese
conflicto justo por esto—. El script lo explica arriba del todo y vale
igual con un Blender instalado.

## Lo que falta y es fácil de añadir cuando decidas

- **Tubos de pared múltiple**, con `builders/assemblies.py`, si lo
  recubierto de grafito es el tubo entero y no sólo la punta.
- **N piridínico y pirrólico de verdad.** El dopaje sustitucional de aquí
  da el N **grafítico**; los otros dos motivos son N en el borde de una
  vacante y hay que pasar por `defects/`. Si tu XPS dice que el
  piridínico domina, este modelo todavía no lo representa.
- **La partícula fuera de la punta**, que era la v1 y sigue siendo la
  otra lectura de tu micrografía: cambiar el signo del desplazamiento en
  `arm_with_particle_inside`.
