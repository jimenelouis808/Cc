# Dependencias de R. Ejecutar una sola vez.
# renv es opcional pero recomendado: congela las versiones y hace el análisis
# reproducible dentro de un año, cuando bibliometrix haya cambiado de API.
packages <- c("bibliometrix", "dplyr", "tidyr", "ggplot2", "readr", "igraph")
missing <- setdiff(packages, rownames(installed.packages()))
if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org")

# Recomendado:
#   install.packages("renv"); renv::init(); renv::snapshot()
