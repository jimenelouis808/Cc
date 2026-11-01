# Fixture mínimo para `99_check_r_setup.R`

25 obras sintéticas en formato nativo de Scopus (CSV) y Web of Science (texto
etiquetado), generadas con `python -m nanocarbon_biblio.cli demo --n-works 25`.

Existe para que la comprobación del entorno de R sea **autónoma**: verifica que
`convert2df` parsea los dos formatos y produce un data frame con los campos que
los análisis necesitan, sin depender de Python ni de que ya tengas corpus.

**Son datos inventados.** No los uses para nada que acabe en el manuscrito.
