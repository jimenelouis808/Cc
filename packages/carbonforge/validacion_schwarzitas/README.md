# Schwarzitas P, D y G: una comprobación de clase, no una validación

Nueve puntos de GCMC a 77 K sobre las tres schwarzitas de hexágonos y
heptágonos que construye `nanocarbon_lab` (ruta de rodilla), contra
Kowalczyk, Holyst, Terrones y Terrones, *Phys. Chem. Chem. Phys.* **9**
(2007) 1786.

**Esto no es lo que fue Levesque, y la diferencia importa.** Allí había
tablas con cinco decimales y un adsorbente definido hasta el último
átomo, y el resultado fue ±6 % en dieciséis puntos. Aquí:

- Los números de Kowalczyk **están en figuras**. Lo único en su texto es
  «~3 wt% a 3.8 MPa y 77 K» para las tres, y 6 wt% para la icosaédrica.
- Una «schwarzita primitive» es **una familia, no un fichero**. El área
  por gramo depende del tamaño de celda y del radio de tubo. Las
  nuestras no tienen por qué ser las suyas, y no lo son.

Así que lo que se puede comprobar es si el prototipo cae donde debe para
esta clase de material, y si lo hace por la razón correcta.

## Las estructuras, medidas antes de simular

| | átomos | celda | vacío | helio | área (m²/g) | cavidad | canal |
|---|---|---|---|---|---|---|---|
| primitive | 968 | 31.9 Å | 0.537 | 0.854 | 2543 | 25.3 Å | 14.6 Å |
| diamond | 1440 | 30.0 Å | 0.266 | 0.721 | 1717 | 11.4 Å | 6.7 Å |
| gyroid | 1744 | 39.0 Å | 0.548 | 0.867 | 2534 | 16.7 Å | 15.8 Å |

Todas cruzan la celda con holgura para H₂ (sonda de 2.9 Å de diámetro),
así que el GCMC tenía sentido. El área es geométrica, no BET.

## Lo que salió

| | 1 MPa | 3.8 MPa | 10 MPa |
|---|---|---|---|
| primitive | 4.54 | 6.07 | 6.98 |
| diamond | 3.28 | 3.68 | 3.86 |
| gyroid | 4.76 | 6.37 | 7.29 |

wt% absoluto, masa total. `tabla.txt` lleva también ρ_o, N y la base
por carbono; los tres `.json` son la salida completa.

**Las formas son Langmuir**, como describe el artículo: casi todo el
salto ocurre antes de 3.8 MPa. El diamond es el más plano de los tres y
es el de poros más estrechos — canal de 6.7 Å contra 14–16 Å. Un poro
estrecho satura antes, y eso es coherencia interna, no ajuste.

## ¿Es la diferencia con Kowalczyk de método o de material?

De material, y la figura lo muestra: **el diamond, cuya área (1717 m²/g)
es la que su ~3 wt% implica por la regla de Chahine, da 3.68 absoluto y
2.82 en exceso.** Las otras dos son el doble de abiertas y dan el doble.
No estamos discrepando con él: estamos describiendo materiales distintos
que comparten nombre.

## Y el área no lo explica todo

Esto hay que decirlo porque la figura invita a lo contrario. A 3.8 MPa,
frente a la regla de Chahine (~1 wt% por 500 m²/g a 77 K):

| | área | wt% abs | wt% exceso | abs/Chahine | exc/Chahine |
|---|---|---|---|---|---|
| haz SWNT | 2590 | 4.49 | 3.49 | 0.87 | 0.67 |
| primitive | 2543 | 6.07 | 4.32 | 1.19 | 0.85 |
| diamond | 1717 | 3.68 | 2.82 | 1.07 | 0.82 |
| gyroid | 2534 | 6.37 | 4.57 | 1.26 | 0.90 |

El wt% en exceso es `100 m_exc/(m_exc + m_C)`, la misma fórmula que el
absoluto con la masa que queda al descontar el gas libre — no el
absoluto reescalado por `rho_exc/rho_abs`, que es lo que decía antes
esta columna y da entre 1 y 2 % menos.

Los cocientes van de 0.87 a 1.26, un factor 1.45 entre el menor y el
mayor, que es lo que una regla empírica entrega: suficiente para
descartar un error de factor dos y no para verificar nada. Y el caso
que lo deja claro: **el haz y el gyroid tienen casi la misma área (2590
contra 2534) y difieren un 42 %.** Parte se explica por la fracción de
vacío — 0.23 contra 0.55, y en base absoluta ese volumen libre cuenta
como gas comprimido además de película adsorbida, por lo que pasar a
exceso encoge la brecha del 45 % al 34 %. No la cierra.

El punto del haz a 3.8 MPa está **interpolado** entre los puntos
medidos de 3.0 y 5.0 MPa de su isoterma validada (4.43 y 4.58 wt%), y
la figura lo marca.

## Absoluto contra exceso

Kowalczyk dice «absolute value of adsorption» en el pie de su figura 1a,
así que absoluto contra absoluto es la comparación correcta **para él**.
Un experimento mide exceso: a 3.8 MPa y 77 K el H₂ en bulto vale
0.01357 g/cm³, y el exceso es del 70 al 76 % del absoluto en estas
cuatro estructuras. Un 6.07 wt% absoluto son ~4.2 en el banco.

## Reproducirlo

```bash
nanocarbon  # construir las schwarzitas y exportar extxyz
carbonforge adsorcion porosidad primitive.extxyz
carbonforge adsorcion isoterma primitive.extxyz salida/ \
    --temperatura 77 --presiones 1.0,3.8,10.0 \
    --equilibrado 1500000 --produccion 1500000
python -m carbonforge.adsorption.run salida/
```

Unos 45 min por punto. La carpeta reanuda.

## Lo que queda abierto

El punto del diamond a 10 MPa avisa de que la carga seguía derivando
(−9 moléculas de 344). El signo dice que 3.86 es si acaso una ligera
sobreestimación. Los otros ocho no avisan.

Y la pregunta que estos nueve puntos no responden: **cuánto de la
diferencia entre el haz y el gyroid a igual área es física y cuánto es
que un «área accesible geométrica» mide cosas distintas en un haz de
tubos y en una superficie mínima.** Eso necesita una medida que aquí no
está hecha.
