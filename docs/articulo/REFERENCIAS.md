# Referencias — metodología de nanocarbon_lab

Generado de `build_detallado.js` (sección 9.1), que es donde viven. Las citas
proceden de los comentarios y docstrings del código, donde se registraron al
implementar cada método.

**Verifique volumen, página y año contra el original antes de enviar.** Este
archivo acredita la procedencia, no sustituye la comprobación bibliográfica: varias
entradas están incompletas a propósito porque el código sólo registró autores y
tema, y completarlas de memoria sería inventar una cita.

## Fuentes metodológicas

1. Romo-Herrera, Terrones, Terrones, Dag y Meunier — Nano Letters 7 (2007) 570. La jerarquía de superredes de nanotubos. La comprobación del censo de anillos reproduce la que describe su Información de Apoyo S2.

2. Terrones y col. — Physical Review Letters 84 (2000) 1716. Haeckelitas: redes planas y tubulares de pentágonos, hexágonos y heptágonos.

3. Krishnan y col. — Nature 388 (1997). Ángulos de apertura observados en nanoconos de carbono, que fijan los casos construibles.

4. László, I. — Theoretical Chemistry Accounts 134 (2015) 104. Revisión de coordenadas topológicas a partir de la matriz de adyacencia.

5. Fowler y Manolopoulos; Pisanski y Shawe-Taylor — caso esférico de las coordenadas topológicas.

6. Graovac y col. — demostración de que tres vectores propios no bastan para el toro.

7. Popović y col. — Contemporary Materials III-1 (2012) 51; Liu y col. — Nanoscale Research Letters 5 (2010) 478. Relaciones D/d de bobinas monocapa relajadas.

8. Botsch, M. y Kobbelt, L. — esquema de remallado isotrópico (dividir, colapsar, voltear, suavizar, reproyectar).

9. Lorensen, W. E. y Cline, H. E. — Computer Graphics 21 (1987) 163. Marching cubes.

10. Dunlap, B. I. — criterio de disclinaciones en carbonos toroidales; y, con Ihara, la ruta topológica a las bobinas. El codo a inglete de la sección 5 es un codo de Dunlap doble, y el giro de 30° por par pentágono-heptágono es el ángulo del codo publicado: ésa es la comprobación de que la ley empleada es la correcta.

11. Lenosky, T.; Gonze, X.; Teter, M. y Elser, V. — Nature 355 (1992) 333. Carbono grafítico de curvatura negativa sobre la superficie D. La celda primitiva de D es la D216 de ese trabajo, de género 3; las celdas de la sección 5 son las convencionales cúbicas, que son cuatro primitivas de la misma superficie.

12. Stone, A. J. y Wales, D. J. — la rotación de enlace que genera pares 5-7.

## Software

El trabajo se apoya en el ecosistema científico de Python: NumPy, SciPy, ASE,
NetworkX, scikit-image y Matplotlib. Formulación sugerida para los
agradecimientos en el documento completo (sección 9.2).

## Figuras

Las seis figuras del documento son estructuras reales renderizadas directamente
del paquete por `figuras.py` y los guiones `fig_*.py`. Ninguna es un esquema
dibujado a mano, y los censos de anillos impresos en cada panel salen del mismo
cálculo que produjo la geometría.

| archivo | qué muestra |
|---|---|
| `figuras/fig1_metodo.png` | las etapas de la ruta implícita, de campo a red |
| `figuras/fig2_basicas.png` | familias tejidas directamente |
| `figuras/fig3_toroide.png` | toroides y la contabilidad de sus disclinaciones |
| `figuras/fig4_superredes.png` | superredes de nanotubos |
| `figuras/fig5_implicitas.png` | familias nacidas de un campo implícito |
| `figuras/fig6_rodillas.png` | las nueve estructuras de la ruta de rodillas |
| `figuras/prueba_c60.png`, `prueba_cnt.png` | casos de contraste de la validación |
