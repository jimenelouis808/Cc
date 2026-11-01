# Lanza biblioshiny. Después: Data -> Load bibliometrix file -> data/processed/M.rds
#
# NO uses "Import raw files": volvería a parsear los ficheros crudos y perderías
# las columnas de etiquetas que añadió el pipeline de Python (dopant, defect,
# study_type, morphology, application), que son las que permiten segmentar
# cualquier análisis por faceta.
if (!requireNamespace("bibliometrix", quietly = TRUE)) {
  stop("bibliometrix no está instalado.\n",
       "  Ejecuta:  Rscript R/install_deps.R\n",
       "  Y después: Rscript R/99_check_r_setup.R", call. = FALSE)
}
suppressPackageStartupMessages(library(bibliometrix))

if (!file.exists(file.path("data", "processed", "M.rds"))) {
  stop("Falta data/processed/M.rds. Corre antes:\n",
       "  python -m nanocarbon_biblio.cli run --raw data/raw --out data/processed\n",
       "  Rscript R/00_build_M.R")
}
biblioshiny()
