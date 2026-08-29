# =============================================================================
# 01_core_analyses.R — análisis guionizados y reproducibles.
#
# biblioshiny es para EXPLORAR. Este script es para los resultados que van al
# paper: mismo corpus, mismos parámetros, misma salida, cada vez. Todo lo que
# aparezca en una figura del manuscrito debería salir de aquí, no de un clic.
#
# Entrada:  data/processed/M.rds   (lo escribe R/00_build_M.R)
# Salida:   results/*.csv, results/*.png, results/session_info.txt
# =============================================================================

suppressPackageStartupMessages({
  library(bibliometrix)
})

PROCESSED <- file.path("data", "processed")
RESULTS   <- "results"
dir.create(RESULTS, showWarnings = FALSE, recursive = TRUE)

M <- readRDS(file.path(PROCESSED, "M.rds"))
message(sprintf("Corpus: %d documentos, %d-%d",
                nrow(M), min(M$PY, na.rm = TRUE), max(M$PY, na.rm = TRUE)))

save_csv <- function(x, name) {
  utils::write.csv(x, file.path(RESULTS, paste0(name, ".csv")), row.names = FALSE)
  message("  -> results/", name, ".csv")
}
open_png <- function(name, width = 2000, height = 1400) {
  grDevices::png(file.path(RESULTS, paste0(name, ".png")),
                 width = width, height = height, res = 180)
}

# Un análisis que falla no debe llevarse por delante los otros once. La API de
# bibliometrix cambia entre versiones menores; si un paso cae, se registra y el
# script sigue. Revisa results/failed_steps.txt al terminar.
FAILED <- character(0)
step <- function(label, expr) {
  message("[", label, "]")
  out <- tryCatch(force(expr), error = function(e) {
    msg <- sprintf("%s: %s", label, conditionMessage(e))
    message("  FALLO -> ", msg)
    FAILED <<- c(FAILED, msg)
    NULL
  })
  invisible(out)
}

# ------------------------------------------------------- 1. descriptivos base
results <- biblioAnalysis(M, sep = ";")
s <- summary(results, k = 25, pause = FALSE, verbose = FALSE)
capture.output(s, file = file.path(RESULTS, "summary.txt"))

save_csv(as.data.frame(s$AnnualProduction), "annual_production")
save_csv(as.data.frame(s$MostRelSources),   "most_relevant_sources")
save_csv(as.data.frame(s$MostRelAuthors),   "most_relevant_authors")
save_csv(as.data.frame(s$MostCitedPapers),  "most_cited_papers")
save_csv(as.data.frame(s$MostRelCountries), "most_relevant_countries")

# Leyes bibliométricas: núcleo de revistas (Bradford) y productividad (Lotka).
step("Bradford", {
  bradford_res <- bradford(M)
  save_csv(bradford_res$table, "bradford")
})
step("Lotka", {
  lotka_res <- bibliometrix::lotka(results)
  save_csv(as.data.frame(lotka_res$AuthorProd), "lotka_author_productivity")
})

# -------------------------------------- 2. RQ2: acoplamiento teoría/experimento
# Esta es la sección original del review. Requiere las etiquetas de Python.
if ("study_type" %in% names(M)) {
  by_year <- as.data.frame(table(M$PY, M$study_type))
  names(by_year) <- c("year", "study_type", "n")
  by_year$year <- as.integer(as.character(by_year$year))
  save_csv(by_year, "study_type_by_year")

  wide <- stats::xtabs(n ~ year + study_type, data = by_year)
  share <- prop.table(wide, margin = 1)
  save_csv(as.data.frame.matrix(round(share, 4)), "study_type_share_by_year")

  open_png("study_type_share")
  graphics::matplot(
    as.integer(rownames(share)), share, type = "l", lty = 1, lwd = 2,
    xlab = "Año", ylab = "Cuota de documentos",
    main = "Tipo de estudio a lo largo del tiempo"
  )
  graphics::legend("topleft", colnames(share), lty = 1, lwd = 2,
                   col = seq_len(ncol(share)), bty = "n")
  grDevices::dev.off()
  message("  -> results/study_type_share.png")

  # Citación cruzada: ¿la comunidad experimental lee a la teórica y viceversa?
  # Se aproxima con acoplamiento local; el cálculo exacto requiere resolver CR
  # contra el propio corpus (ver docs/WORKFLOW.md, paso 8).
  cited <- localCitations(M, sep = ";")
  save_csv(cited$Papers, "local_citations_papers")
} else {
  warning("No hay columna study_type: corre antes el pipeline de Python. ",
          "Te estás saltando RQ2, que es la parte original del review.")
}

# -------------------------------------------- 3. RQ3: cobertura dopante x uso
if (all(c("dopant", "application") %in% names(M))) {
  explode <- function(x) strsplit(ifelse(is.na(x), "", as.character(x)), "|", fixed = TRUE)
  pairs <- do.call(rbind, lapply(seq_len(nrow(M)), function(i) {
    d <- explode(M$dopant[i])[[1]]; a <- explode(M$application[i])[[1]]
    d <- d[nzchar(d)]; a <- a[nzchar(a)]
    if (!length(d) || !length(a)) return(NULL)
    expand.grid(dopant = d, application = a, stringsAsFactors = FALSE)
  }))
  if (!is.null(pairs)) {
    matrix_da <- as.data.frame.matrix(table(pairs$dopant, pairs$application))
    utils::write.csv(matrix_da, file.path(RESULTS, "matrix_dopant_application.csv"))
    message("  -> results/matrix_dopant_application.csv")
    message("     Las celdas vacías son candidatas a 'hueco de investigación'. ",
            "Cruza con la faceta teórica antes de afirmarlo.")
  }
}

# ------------------------------------------------- 4. estructura intelectual
# Co-ocurrencia de palabras clave. Carga el tesauro ANTES: sin él,
# 'n-doped cnt' y 'nitrogen-doped carbon nanotube' son dos nodos distintos.
thes <- file.path("queries", "thesaurus.txt")
if (!file.exists(thes)) {
  warning("No existe queries/thesaurus.txt. El mapa de co-palabras estará ",
          "fragmentado por variantes ortográficas. Genéralo con: ",
          "python -m nanocarbon_biblio.cli thesaurus")
  thes <- NULL
}

M_terms <- termExtraction(
  M, Field = "DE", remove.numbers = TRUE, stemming = FALSE,
  language = "english", verbose = FALSE,
  synonyms = if (!is.null(thes)) readLines(thes, warn = FALSE) else NULL
)

net_matrix <- biblioNetwork(M_terms, analysis = "co-occurrences",
                            network = "keywords", sep = ";")
open_png("cooccurrence_keywords")
invisible(networkPlot(net_matrix, n = 60, Title = "Co-ocurrencia de keywords",
                      type = "fruchterman", size.cex = TRUE, size = 20,
                      remove.multiple = FALSE, labelsize = 0.8,
                      edgesize = 5, edges.min = 3, verbose = FALSE))
grDevices::dev.off()
message("  -> results/cooccurrence_keywords.png")

# Mapa temático de Callon (centralidad x densidad).
step("Mapa temático", {
  tm <- thematicMap(M, field = "DE", n = 250, minfreq = 5,
                    stemming = FALSE, size = 0.5, repel = TRUE)
  save_csv(tm$words, "thematic_map_words")
  save_csv(tm$clusters, "thematic_map_clusters")
})

# Evolución temática con CORTES CON SENTIDO FÍSICO (ver docs/PROTOCOL.md §6),
# no con periodos de igual longitud.
cuts <- c(1999, 2004, 2010, 2016, 2021)
cuts <- cuts[cuts > min(M$PY, na.rm = TRUE) & cuts < max(M$PY, na.rm = TRUE)]
if (length(cuts) >= 2) {
  step("Evolución temática", {
    te <- thematicEvolution(M, field = "DE", years = cuts, n = 250, minFreq = 3)
    save_csv(te$Nodes, "thematic_evolution_nodes")
    save_csv(te$Edges, "thematic_evolution_edges")
  })
}

# ------------------------------------- 5. raíces intelectuales del campo (RPYS)
# Reference Publication Year Spectroscopy: los picos son los trabajos
# fundacionales. Espera picos en 1991 (Iijima) y 1986 (Stone & Wales).
if ("CR" %in% names(M) && sum(nchar(as.character(M$CR)) > 1) > 0.3 * nrow(M)) {
  step("RPYS", {
    rpys_res <- rpys(M, sep = ";", graph = FALSE)
    # Los nombres de los elementos cambian entre versiones: se guarda todo lo
    # que sea un data frame y ya se elige después.
    for (nm in names(rpys_res)) {
      if (is.data.frame(rpys_res[[nm]])) save_csv(rpys_res[[nm]], paste0("rpys_", nm))
    }
    message("  -> RPYS listo. Contrasta los picos con CRExplorer para el análisis fino.")
  })
} else {
  warning("CR insuficiente para RPYS. Re-exporta con referencias citadas.")
}

# --------------------------------- 6. estructura social y colaboración
countries <- metaTagExtraction(M, Field = "AU_CO", sep = ";")
country_net <- biblioNetwork(countries, analysis = "collaboration",
                             network = "countries", sep = ";")
open_png("collaboration_countries")
invisible(networkPlot(country_net, n = 40, Title = "Colaboración entre países",
                      type = "circle", size = 12, remove.multiple = TRUE,
                      labelsize = 0.7, verbose = FALSE))
grDevices::dev.off()
message("  -> results/collaboration_countries.png")

# --------------------------------------------------------- 7. trazabilidad
if (length(FAILED)) {
  writeLines(FAILED, file.path(RESULTS, "failed_steps.txt"))
  message("\n", length(FAILED), " paso(s) fallaron. Ver results/failed_steps.txt")
}

writeLines(
  c(capture.output(utils::sessionInfo()),
    "",
    sprintf("Corpus: %d documentos", nrow(M)),
    sprintf("Generado: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"))),
  file.path(RESULTS, "session_info.txt")
)
message("\nHecho. Salidas en results/")
