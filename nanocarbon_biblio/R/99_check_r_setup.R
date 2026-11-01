# =============================================================================
# 99_check_r_setup.R — comprueba que el lado de R funciona ANTES de gastar
#                      tiempo con el corpus real.
#
#   Rscript R/99_check_r_setup.R
#
# Es el equivalente en R de `python3 start.py --check`. Verifica, en orden:
#   1. la versión de R,
#   2. que bibliometrix está instalado y con qué versión,
#   3. que existen todas las funciones que usan 00_build_M.R y 01_core_analyses.R
#      (la API de bibliometrix cambia entre versiones menores),
#   4. que convert2df parsea de verdad los dos formatos nativos, sobre un
#      fixture de 25 obras,
#   5. que el data frame resultante trae los campos críticos, CR incluido.
#
# Cada comprobación dice qué hacer si falla. El script nunca aborta a la
# primera: las ejecuta todas y resume al final.
# =============================================================================

PASS <- 0L; FAILS <- character(0); WARNS <- character(0)

ok   <- function(msg) { PASS <<- PASS + 1L; cat("  [OK] ", msg, "\n", sep = "") }
bad  <- function(msg, fix = "") {
  FAILS <<- c(FAILS, msg)
  cat("  [X]  ", msg, "\n", sep = "")
  if (nzchar(fix)) for (l in strsplit(fix, "\n")[[1]]) cat("        ", l, "\n", sep = "")
}
warn <- function(msg, fix = "") {
  WARNS <<- c(WARNS, msg)
  cat("  [!]  ", msg, "\n", sep = "")
  if (nzchar(fix)) for (l in strsplit(fix, "\n")[[1]]) cat("        ", l, "\n", sep = "")
}

cat("\n  nanocarbon_biblio — revisión del entorno de R\n")
cat("  ", strrep("-", 62), "\n", sep = "")

# ----------------------------------------------------------------- 1. R
if (getRversion() < "4.1.0") {
  bad(paste0("R ", getRversion(), " es antiguo (se recomienda 4.1 o superior)."),
      "Descarga una versión actual desde https://cran.r-project.org")
} else {
  ok(paste0(R.version.string))
}

# -------------------------------------------------------- 2. bibliometrix
has_biblio <- requireNamespace("bibliometrix", quietly = TRUE)
if (!has_biblio) {
  bad("bibliometrix NO está instalado.",
      "Rscript R/install_deps.R\n  o, en una sesión de R:\n  install.packages(\"bibliometrix\")")
} else {
  v <- as.character(utils::packageVersion("bibliometrix"))
  ok(paste0("bibliometrix ", v))
  if (utils::packageVersion("bibliometrix") < "4.0.0") {
    warn(paste0("bibliometrix ", v, " es anterior a 4.0; los scripts se",
                " escribieron contra la serie 4.x."),
         "update.packages(ask = FALSE)")
  }
}

# ------------------------------------------- 3. funciones que los scripts usan
# La API de bibliometrix se mueve entre versiones menores. Comprobarlo aquí
# convierte un fallo a mitad del análisis en un aviso de treinta segundos.
needed <- c("convert2df", "mergeDbSources", "biblioAnalysis", "biblioNetwork",
            "networkPlot", "thematicMap", "thematicEvolution", "termExtraction",
            "metaTagExtraction", "localCitations", "bradford", "lotka", "rpys")
if (has_biblio) {
  exported <- ls(getNamespace("bibliometrix"))
  missing_fns <- setdiff(needed, exported)
  if (length(missing_fns) == 0) {
    ok(paste0("Las ", length(needed), " funciones que usan los scripts existen"))
  } else {
    warn(paste0("No existen en esta versión: ", paste(missing_fns, collapse = ", ")),
         paste0("Los pasos que las usan se saltarán y quedarán anotados en\n",
                "results/failed_steps.txt. Actualiza bibliometrix si los necesitas."))
  }
}

# --------------------------------------- 4 y 5. convert2df sobre el fixture
FIX <- file.path("R", "fixtures")
if (!dir.exists(FIX)) {
  bad(paste0("Falta la carpeta ", FIX, "."),
      "El checkout está incompleto:\n  git checkout claude/binance-scalping-bot-cut2op")
} else if (has_biblio) {
  scopus_file <- file.path(FIX, "scopus_mini.csv")
  wos_file    <- file.path(FIX, "wos_mini.txt")

  Ms <- tryCatch(
    suppressWarnings(suppressMessages(
      bibliometrix::convert2df(scopus_file, dbsource = "scopus", format = "csv"))),
    error = function(e) { bad(paste0("convert2df falló con el CSV de Scopus: ",
                                     conditionMessage(e)),
                              "Pega este error y se diagnostica."); NULL })
  if (!is.null(Ms)) ok(paste0("convert2df lee el formato de Scopus (", nrow(Ms), " registros)"))

  Mw <- tryCatch(
    suppressWarnings(suppressMessages(
      bibliometrix::convert2df(wos_file, dbsource = "wos", format = "plaintext"))),
    error = function(e) { bad(paste0("convert2df falló con el texto de WoS: ",
                                     conditionMessage(e)),
                              "Pega este error y se diagnostica."); NULL })
  if (!is.null(Mw)) ok(paste0("convert2df lee el formato de WoS (", nrow(Mw), " registros)"))

  if (!is.null(Ms) && !is.null(Mw)) {
    M <- tryCatch(
      suppressWarnings(suppressMessages(
        bibliometrix::mergeDbSources(Ms, Mw, remove.duplicated = TRUE))),
      error = function(e) { bad(paste0("mergeDbSources falló: ", conditionMessage(e))); NULL })
    if (!is.null(M)) {
      ok(paste0("mergeDbSources une las dos bases (", nrow(M), " registros)"))

      # Los campos sin los que los análisis no significan nada.
      critical <- c(TI = "título", AB = "resumen", PY = "año",
                    SO = "revista", CR = "referencias citadas")
      for (tag in names(critical)) {
        if (!tag %in% names(M)) {
          bad(paste0("Falta el campo ", tag, " (", critical[[tag]], ") tras convert2df."))
        } else {
          filled <- sum(!is.na(M[[tag]]) & nchar(as.character(M[[tag]])) > 1)
          pct <- 100 * filled / nrow(M)
          if (pct < 50) {
            warn(sprintf("%s (%s) solo viene en el %.0f%% de los registros.",
                         tag, critical[[tag]], pct))
          } else {
            ok(sprintf("%-2s %-20s %3.0f%% de cobertura", tag, critical[[tag]], pct))
          }
        }
      }

      # Un análisis real de punta a punta, no solo el parseo.
      res <- tryCatch(
        suppressWarnings(suppressMessages(bibliometrix::biblioAnalysis(M, sep = ";"))),
        error = function(e) { bad(paste0("biblioAnalysis falló: ", conditionMessage(e))); NULL })
      if (!is.null(res)) ok("biblioAnalysis corre de principio a fin")
    }
  }
}

# ---------------------------------------------------------------- resumen
cat("  ", strrep("-", 62), "\n", sep = "")
if (length(FAILS) == 0 && length(WARNS) == 0) {
  cat("  ", PASS, " comprobaciones correctas. El lado de R está listo.\n", sep = "")
  cat("  Siguiente:  Rscript R/00_build_M.R\n\n")
  quit(status = 0)
}
cat("  ", PASS, " correctas, ", length(WARNS), " avisos, ",
    length(FAILS), " fallos.\n", sep = "")
if (length(FAILS)) {
  cat("\n  Hay que arreglar antes de seguir:\n")
  for (f in FAILS) cat("    - ", f, "\n", sep = "")
  cat("\n")
  quit(status = 1)
}
cat("\n  Los avisos no impiden continuar, pero léelos.\n\n")
quit(status = 0)
