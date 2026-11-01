# =============================================================================
# install_deps.R — instala las dependencias de R. Ejecutar una sola vez.
#
#   Rscript R/install_deps.R
#
# Después comprueba que todo funciona de verdad:
#   Rscript R/99_check_r_setup.R
# =============================================================================

packages <- c("bibliometrix", "dplyr", "tidyr", "ggplot2", "readr", "igraph")

installed <- rownames(installed.packages())
missing <- setdiff(packages, installed)

if (!length(missing)) {
  message("Todo instalado ya: ", paste(packages, collapse = ", "))
} else {
  message("Faltan ", length(missing), ": ", paste(missing, collapse = ", "))
  message("bibliometrix arrastra muchas dependencias; la primera vez puede ",
          "tardar bastantes minutos.\n")
  install.packages(missing, repos = "https://cloud.r-project.org")
}

# Instalar no es lo mismo que funcionar: un paquete puede quedar a medias si
# falta una librería del sistema. Se comprueba carga a carga.
failed <- character(0)
for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) failed <- c(failed, pkg)
}

if (length(failed)) {
  message("\nNO se pudieron cargar: ", paste(failed, collapse = ", "))
  message("Causas habituales:")
  message("  · Sin salida a internet, o un proxy bloquea CRAN.")
  message("  · Faltan librerías del sistema. En Ubuntu/Debian suele bastar:")
  message("      sudo apt install build-essential libcurl4-openssl-dev \\")
  message("                       libssl-dev libxml2-dev libfontconfig1-dev \\")
  message("                       libharfbuzz-dev libfribidi-dev")
  message("  · En macOS: xcode-select --install")
  quit(status = 1)
}

message("\nTodo cargado correctamente.")
message("Siguiente paso:  Rscript R/99_check_r_setup.R")

# Recomendado para que el análisis siga siendo reproducible dentro de un año,
# cuando bibliometrix haya cambiado de API:
#   install.packages("renv"); renv::init(); renv::snapshot()
