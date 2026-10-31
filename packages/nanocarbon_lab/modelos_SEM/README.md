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

| modelo | átomos | fórmula | núcleo | capas C |
|---|---|---|---|---|
| `placa_H_nucleo_desnudo` | 772 | Fe388Se384 | Fe388Se384 | 0 |
| `placa_H_C` | 2398 | C1626Fe388Se384 | Fe388Se384 | 1 |
| `placa_H_NSe` | 2398 | C1488Fe388N97Se425 | Fe388Se384 | 1 |
| `placa_H_NSeO` | 2413 | C1489H6Fe388N98O5Se427 | Fe388Se384 | 1 |
| `barra_T_nucleo_desnudo` | 192 | Fe96Se96 | Fe96Se96 | 0 |
| `barra_T_C` | 996 | C804Fe96Se96 | Fe96Se96 | 1 |
| `barra_T_NSe` | 996 | C736Fe96N48Se116 | Fe96Se96 | 1 |
| `barra_T_NSeO` | 1006 | C737H4Fe96N49O4Se116 | Fe96Se96 | 1 |
| `particula_Fe3C_nucleo_desnudo` | 1100 | C292Fe808 | C292Fe808 | 0 |
| `particula_Fe3C_C` | 4306 | C3498Fe808 | C292Fe808 | 2 |
| `particula_Fe3C_NSe` | 4306 | C3226Fe808N192Se80 | C292Fe808 | 2 |
| `particula_Fe3C_NSeO` | 4336 | C3228H12Fe808N194O10Se84 | C292Fe808 | 2 |
| `nanotubo_C` | 2240 | C2240 | — | 0 |
| `nanotubo_NSe` | 2240 | C2050N134Se56 | — | 0 |
| `nanotubo_NSeO` | 2265 | C2052H10N136O9Se58 | — | 0 |
| `escena_punta` | 6546 | C5738Fe808 | Fe3C | 0 |
| `escena_anemona` | 32730 | C28690Fe4040 | Fe3C | 0 |

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

## Las dos escenas NO son estructuras

`escena_punta` y `escena_anemona` salen de `scene.compose`, que coloca
estructuras terminadas una al lado de otra y **no las enlaza**. El propio
programa lo dice en cada una:

> esto es una ILUSTRACIÓN, no una estructura: los objetos se colocaron a
> mano, no hay ninguna interfase relajada entre dos de ellos, y nada en
> los huecos es física. No lo cuantifiques, no lo relajes y no se lo des
> a un código DFT.

Sirven para la figura y para enseñar la morfología. En `escena_punta` el
tubo y la partícula quedan a 3.38 Å, que es el contacto grafítico, y hay
que dejarlo explícito porque `compose` coloca y no separa. En la anémona
los tubos arrancan a un radio de tubo del origen: cinco convergiendo en
un punto quedaban a 0.05 Å, que son cinco tubos ocupando el mismo sitio.

## Lo que falta y es fácil de añadir cuando decidas

- **La partícula DENTRO de la punta.** Está a una línea: en
  `compositions()`, mover la partícula a `-half_tube` y encoger su radio
  por debajo del radio del tubo (7.85 Å aquí). Lo dejé fuera porque es la
  otra lectura de tu micrografía y no sé cuál es.
- **Tubos de pared múltiple**, con `builders/assemblies.py`, si lo que se
  ve recubierto de grafito es el tubo entero y no solo la punta.
- **Tubo curvado o en bambú**, con `shape=` y los preajustes de
  `centerline.py`, si la anémona tiene los brazos doblados.
- **N piridínico y pirrólico de verdad.** El dopaje sustitucional de aquí
  da el N **grafítico**; los otros dos motivos son N en el borde de una
  vacante y hay que pasar por `defects/`. Si tu XPS dice que el
  piridínico domina, este modelo todavía no lo representa.
