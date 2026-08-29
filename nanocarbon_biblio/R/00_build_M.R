# =============================================================================
# 00_build_M.R — construye el data frame de bibliometrix a partir del bundle
#                que escribe el pipeline de Python, y lo deja listo para
#                biblioshiny.
#
# Entrada  (data/processed/, escrito por `python -m nanocarbon_biblio.cli run`):
#   scopus_kept.csv   registros de Scopus supervivientes, en formato NATIVO
#   wos_kept.txt      registros de WoS supervivientes, en formato NATIVO tagged
#   labels.csv        etiquetas de Python (dopante, defecto, study_type, ...)
#
# Salida:
#   data/processed/M.rds     <- esto es lo que se carga en biblioshiny
#   data/processed/M.rdata
#   data/processed/join_report.txt
#
# Por qué los ficheros van en formato nativo: convert2df() parsea las
# referencias citadas (CR) exactamente como bibliometrix espera. Reconstruir CR
# desde Python rompería co-citación, acoplamiento bibliográfico y RPYS.
# =============================================================================

suppressPackageStartupMessages({
  library(bibliometrix)
})

PROCESSED <- file.path("data", "processed")
stopifnot(dir.exists(PROCESSED))

scopus_file <- file.path(PROCESSED, "scopus_kept.csv")
wos_file    <- file.path(PROCESSED, "wos_kept.txt")
labels_file <- file.path(PROCESSED, "labels.csv")

# ---------------------------------------------------------------- 1. importar
Ms <- NULL
Mw <- NULL

if (file.exists(scopus_file) && file.size(scopus_file) > 0) {
  message("convert2df: Scopus …")
  Ms <- convert2df(scopus_file, dbsource = "scopus", format = "csv")
  message(sprintf("  %d registros, %d campos", nrow(Ms), ncol(Ms)))
}

if (file.exists(wos_file) && file.size(wos_file) > 0) {
  message("convert2df: Web of Science …")
  Mw <- convert2df(wos_file, dbsource = "wos", format = "plaintext")
  message(sprintf("  %d registros, %d campos", nrow(Mw), ncol(Mw)))
}

if (is.null(Ms) && is.null(Mw)) {
  stop("No hay ficheros de entrada en ", PROCESSED,
       ". Corre antes: python -m nanocarbon_biblio.cli run")
}

# ------------------------------------------------------------------ 2. unir
# Python ya deduplicó entre bases y escribió cada representante en UN solo
# fichero, así que mergeDbSources no debería encontrar nada. Se deja
# remove.duplicated = TRUE como segunda red de seguridad y se reporta cuánto
# elimina: si elimina mucho, tu umbral de similitud en Python es demasiado alto.
if (!is.null(Ms) && !is.null(Mw)) {
  n_before <- nrow(Ms) + nrow(Mw)
  M <- mergeDbSources(Ms, Mw, remove.duplicated = TRUE)
  message(sprintf("mergeDbSources: %d -> %d (R eliminó %d duplicados extra)",
                  n_before, nrow(M), n_before - nrow(M)))
  if ((n_before - nrow(M)) > 0.02 * n_before) {
    warning("R eliminó >2% extra: baja --title-threshold en el paso de Python.")
  }
} else {
  M <- if (is.null(Ms)) Mw else Ms
}

# ------------------------------------- 3. comprobaciones que importan de verdad
check <- function(field, label) {
  if (!field %in% names(M)) {
    warning(sprintf("Falta el campo %s (%s).", field, label)); return(invisible(0))
  }
  filled <- sum(!is.na(M[[field]]) & nchar(as.character(M[[field]])) > 1)
  message(sprintf("  %-3s %-28s %6d / %6d  (%.1f%%)",
                  field, label, filled, nrow(M), 100 * filled / nrow(M)))
  invisible(filled)
}
message("Cobertura de campos críticos:")
n_cr <- check("CR", "referencias citadas")
check("AB", "resumen")
check("DE", "keywords de autor")
check("ID", "keywords plus")
check("C1", "afiliaciones")
check("DI", "DOI")
check("TC", "citas")

if (is.null(n_cr) || n_cr < 0.5 * nrow(M)) {
  warning("Menos de la mitad de los registros tienen CR. Co-citación, ",
          "acoplamiento bibliográfico y RPYS serán poco fiables. ",
          "Re-exporta incluyendo las referencias citadas.")
}

# --------------------------------------------- 4. unir las etiquetas de Python
# Clave primaria: DOI normalizado. Respaldo: título normalizado.
norm_title <- function(x) {
  x <- tolower(as.character(x))
  x <- gsub("[^a-z0-9]+", " ", x)
  trimws(gsub("\\s+", " ", x))
}

if (file.exists(labels_file)) {
  labels <- utils::read.csv(labels_file, stringsAsFactors = FALSE, check.names = FALSE)
  message(sprintf("labels.csv: %d filas, %d columnas", nrow(labels), ncol(labels)))

  M$.doi_key   <- tolower(trimws(ifelse(is.na(M$DI), "", as.character(M$DI))))
  M$.title_key <- norm_title(M$TI)
  labels$.doi_key   <- tolower(trimws(ifelse(is.na(labels$doi), "", labels$doi)))
  labels$.title_key <- norm_title(labels$title)

  label_cols <- setdiff(names(labels),
                        c("doi", "title", "year", "doc_type", "cited_by",
                          "source", "uid", ".doi_key", ".title_key"))

  by_doi <- labels[labels$.doi_key != "", ]
  by_doi <- by_doi[!duplicated(by_doi$.doi_key), ]
  idx <- match(M$.doi_key, by_doi$.doi_key)
  idx[M$.doi_key == ""] <- NA

  by_title <- labels[!duplicated(labels$.title_key), ]
  idx_title <- match(M$.title_key, by_title$.title_key)

  for (col in label_cols) {
    v <- by_doi[[col]][idx]
    fallback <- by_title[[col]][idx_title]
    v[is.na(v)] <- fallback[is.na(v)]
    M[[col]] <- v
  }

  matched <- sum(!is.na(M[[label_cols[1]]]))
  message(sprintf("Etiquetas unidas: %d / %d (%.1f%%)",
                  matched, nrow(M), 100 * matched / nrow(M)))
  if (matched < 0.9 * nrow(M)) {
    warning("Menos del 90% de los registros recibieron etiquetas. ",
            "Revisa que labels.csv venga de la MISMA corrida que los ficheros nativos.")
  }
  M$.doi_key <- NULL
  M$.title_key <- NULL
} else {
  warning("No hay labels.csv: seguirás sin poder segmentar por dopante/tipo de estudio.")
}

# Las operaciones de data frame pueden tumbar la clase S3 que biblioshiny
# necesita. Se vuelve a poner explícitamente: sin esto, biblioshiny rechaza
# el fichero o lo trata como un data frame cualquiera.
class(M) <- c("bibliometrixDB", "data.frame")

# --------------------------------------------------------------- 5. guardar
saveRDS(M, file.path(PROCESSED, "M.rds"))
save(M, file = file.path(PROCESSED, "M.rdata"))

report <- c(
  sprintf("Generado: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
  sprintf("Documentos: %d", nrow(M)),
  sprintf("Campos: %d", ncol(M)),
  sprintf("Periodo: %s - %s", min(M$PY, na.rm = TRUE), max(M$PY, na.rm = TRUE)),
  sprintf("Fuentes (revistas) distintas: %d", length(unique(M$SO))),
  sprintf("Con CR: %d", sum(!is.na(M$CR) & nchar(as.character(M$CR)) > 1)),
  if ("study_type" %in% names(M))
    paste0("study_type: ", paste(names(table(M$study_type)), table(M$study_type),
                                 sep = "=", collapse = ", ")) else NULL
)
writeLines(report, file.path(PROCESSED, "join_report.txt"))
cat(paste(report, collapse = "\n"), "\n")

message("\nListo. En biblioshiny: Data -> Load bibliometrix file -> data/processed/M.rds")
message("NO uses 'Import raw files': perderías las etiquetas de Python.")
