#!/usr/bin/env Rscript

## ======================================================================
## MaxEnt模型可解释性分析：预测函数重建、SHAP与PDP
## ======================================================================
##
## 先使用Java MaxEnt已经训练完成的10个重复模型，恢复其在每个栅格上的
## logistic适宜性预测计算过程；再检验R中恢复的预测值是否与Java MaxEnt原始
## 输出一致；在确认两者一致后，进一步开展SHAP和部分依赖分析，用于解释
## MaxEnt预测结果背后的环境变量贡献和响应关系。
##
## 流程：
## 1. 读取Java MaxEnt输出的.lambdas文件，恢复每个重复模型的预测计算公式。
## 2. 分别在样点和全量有效环境栅格上比较R计算值与Java MaxEnt输出值，确认预测函数重建过程没有改变原MaxEnt模型结果。
## 3. 将解释范围限定为研究区域，即青海湖流域及共和盆地。
## 4. 背景样本限定为剔除湖面、水体后的陆地有效栅格。
## 5. 基于验证后的Java MaxEnt logistic预测函数，计算SHAP、1D-PDP和2D-PDP。
##
##
## 选项：
##   MAXENT_XAI_PROJECT_DIR   目录，默认/mnt/d/work/XAI。
##   ANALYSIS_SHAP_NSIM      SHAP近似计算的重复抽样次数，默认64。


project_dir <- normalizePath(Sys.getenv("MAXENT_XAI_PROJECT_DIR", "/mnt/d/work/XAI"), mustWork = TRUE)
local_lib <- file.path(project_dir, "04.tmp", "Rlib")
conda_lib <- file.path(project_dir, "04.tmp", "conda_envs", "maxent_xai_r", "lib", "R", "library")
## 加载R包
for (lib in c(local_lib, conda_lib)) {
  if (dir.exists(lib)) .libPaths(c(normalizePath(lib), .libPaths()))
}

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(fastshap)
})

set.seed(20260528)

## ----------------------------------------------------------------------
## 1. 设置输入、输出路径和分析参数
## ----------------------------------------------------------------------

input_maxent_dir <- file.path(project_dir, "00.input", "psyl_max10")
table_dir <- file.path(project_dir, "03.results", "tables")
prepared_table_dir <- file.path(project_dir, "00.input", "prepared_tables")
figure_dir <- file.path(project_dir, "03.results", "figures")
log_dir <- file.path(project_dir, "03.results", "logs")
object_dir <- file.path(project_dir, "03.results", "objects")
doc_dir <- file.path(project_dir, "05.docs", "maxent_region_xai_20260528")

dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(prepared_table_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(object_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(doc_dir, recursive = TRUE, showWarnings = FALSE)

log_stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
log_file <- file.path(log_dir, paste0("maxent_xai_shap_pdp_analysis_", log_stamp, ".log"))
latest_log_file <- file.path(log_dir, "maxent_xai_shap_pdp_analysis.latest.log")
writeLines(character(), log_file)
writeLines(character(), latest_log_file)
log_msg <- function(...) {
  msg <- paste0(...)
  cat(msg, "\n")
  line <- paste0(format(Sys.time(), "[%Y-%m-%d %H:%M:%S] "), msg)
  cat(line, "\n", file = log_file, append = TRUE)
  cat(line, "\n", file = latest_log_file, append = TRUE)
}

as_flag <- function(x, default = TRUE) {
  if (!nzchar(x)) return(default)
  toupper(x) %in% c("TRUE", "T", "1", "YES", "Y")
}

run_shap <- as_flag(Sys.getenv("ANALYSIS_RUN_SHAP"), TRUE)
run_pdp <- as_flag(Sys.getenv("ANALYSIS_RUN_PDP"), TRUE)
write_shap_values <- as_flag(Sys.getenv("ANALYSIS_WRITE_SHAP"), TRUE)
shap_nsim <- as.integer(Sys.getenv("ANALYSIS_SHAP_NSIM", "64"))
pdp_1d_grid_n <- as.integer(Sys.getenv("ANALYSIS_PDP_1D_GRID", "100"))
pdp_2d_grid_n <- as.integer(Sys.getenv("ANALYSIS_PDP_2D_GRID", "50"))
pdp_2d_background_n <- as.integer(Sys.getenv("ANALYSIS_PDP_2D_BACKGROUND", "5000"))

vars <- c(
  "FD_qh", "alt_qhn", "asp_qh", "bio17_qh", "bio2_qh", "bio3_qh",
  "bio9_qh", "cur_qh", "hfi_qh", "ldcvqh", "ndvi_qh", "pop_qh",
  "slop_qh", "veg_qh"
)

var_labels <- c(
  FD_qh = "Fence density (FD)",
  alt_qhn = "Elevation",
  asp_qh = "Aspect",
  bio17_qh = "Dry-season precipitation (Bio17)",
  bio2_qh = "Mean diurnal range (Bio2)",
  bio3_qh = "Isothermality (Bio3)",
  bio9_qh = "Dry-season mean temperature (Bio9)",
  cur_qh = "Curvature",
  hfi_qh = "Human footprint / HII",
  ldcvqh = "Land-cover variable",
  ndvi_qh = "NDVI",
  pop_qh = "Population density",
  slop_qh = "Slope",
  veg_qh = "Vegetation type"
)

## ----------------------------------------------------------------------
## 2. 读取 .lambdas 文件并重建 Java MaxEnt 预测函数
## ----------------------------------------------------------------------

## Java MaxEnt完成训练后，会把每个重复模型的主要参数保存为.lambdas文件。
## 该文件相当于已经训练好的MaxEnt模型说明书，记录了模型使用了哪些环境变量
## 及其组合、每一项的系数，以及把模型计算值转换为raw和logistic预测值所需的
## 常数。因此，读取.lambdas文件可以在R中重建Java MaxEnt的预测结果。
##
## .lambdas文件中的四列记录为feature, lambda, min, max：
##   feature：模型中的变量项，可以是单个变量、变量平方项或两个变量的乘积项；
##   lambda：该变量项在MaxEnt模型中的系数；
##   min/max：该变量项在模型计算中使用的取值范围，用于截断和标准化。
## 文件末尾的linearPredictorNormalizer、densityNormalizer和entropy用于把模型
## 线性计算值转换为Java MaxEnt输出中的raw和logistic预测值。

parse_lambdas <- function(path) {
  lines <- readLines(path, warn = FALSE)
  pieces <- strsplit(lines[nzchar(lines)], ",")
  features <- list()
  constants <- list()

  for (parts in pieces) {
    parts <- trimws(parts)
    if (length(parts) == 4L) {
      features[[length(features) + 1L]] <- data.frame(
        feature = parts[1],
        lambda = as.numeric(parts[2]),
        min = as.numeric(parts[3]),
        max = as.numeric(parts[4]),
        stringsAsFactors = FALSE
      )
    } else if (length(parts) == 2L) {
      constants[[parts[1]]] <- as.numeric(parts[2])
    }
  }

  required_constants <- c("linearPredictorNormalizer", "densityNormalizer", "entropy")
  missing_constants <- setdiff(required_constants, names(constants))
  if (length(missing_constants) > 0L) {
    stop("Missing constants in ", basename(path), ": ", paste(missing_constants, collapse = ", "))
  }

  list(features = dplyr::bind_rows(features), constants = constants, source = path)
}

## 按.lambdas文件中记录的变量项计算每个栅格的取值。
## 例如，alt_qhn表示直接使用海拔值，alt_qhn^2表示海拔平方，
## bio17_qh*hfi_qh表示旱季降水与人为干扰指数的乘积。
feature_value <- function(feature_name, newdata) {
  if (grepl("\\^2$", feature_name)) {
    v <- sub("\\^2$", "", feature_name)
    return(newdata[[v]] ^ 2)
  }
  if (grepl("\\*", feature_name)) {
    vv <- strsplit(feature_name, "\\*")[[1]]
    return(newdata[[vv[1]]] * newdata[[vv[2]]])
  }
  newdata[[feature_name]]
}

## 对每个栅格，脚本依次完成以下计算：
##   1. 根据.lambdas文件计算所有变量项的取值。
##   2. 将变量项限制在Java MaxEnt训练模型时记录的取值范围内。
##   3. 将变量项转换到0-1尺度，与Java MaxEnt内部计算方式一致。
##   4. 将各变量项乘以对应lambda系数并求和，得到模型线性计算值。
##   5. 使用.lambdas文件末尾保存的常数，将线性计算值转换为raw和logistic预测值。
##
## 所有参数均来自原Java MaxEnt输出。
predict_java_maxent <- function(lambda_obj, newdata, type = c("logistic", "raw", "linear")) {
  type <- match.arg(type)
  feat <- lambda_obj$features
  lp <- numeric(nrow(newdata))

  for (i in seq_len(nrow(feat))) {
    z <- feature_value(feat$feature[i], newdata)
    z <- pmin(pmax(z, feat$min[i]), feat$max[i])
    if (feat$max[i] != feat$min[i]) {
      z <- (z - feat$min[i]) / (feat$max[i] - feat$min[i])
    } else {
      z <- z * 0
    }
    lp <- lp + feat$lambda[i] * z
  }

  if (type == "linear") return(lp)

  raw <- exp(lp - lambda_obj$constants$linearPredictorNormalizer) /
    lambda_obj$constants$densityNormalizer

  if (type == "raw") return(raw)

  exp(lambda_obj$constants$entropy) * raw /
    (1 + exp(lambda_obj$constants$entropy) * raw)
}

predict_matrix <- function(lambdas, newdata, type = "logistic") {
  out <- matrix(NA_real_, nrow = nrow(newdata), ncol = length(lambdas))
  for (i in seq_along(lambdas)) {
    out[, i] <- predict_java_maxent(lambdas[[i]], newdata, type = type)
  }
  colnames(out) <- sprintf("spyl_%d", seq_along(lambdas) - 1L)
  out
}

## 读取ASCII栅格文件用于验证。
read_asc <- function(path) {
  con <- file(path, open = "rt")
  on.exit(close(con), add = TRUE)
  header_lines <- readLines(con, n = 6)
  header <- strsplit(trimws(header_lines), "\\s+")
  header_names <- vapply(header, `[`, character(1), 1)
  header_values <- as.numeric(vapply(header, `[`, character(1), 2))
  hdr <- setNames(as.list(header_values), tolower(header_names))
  data <- data.table::fread(
    path,
    skip = 6,
    header = FALSE,
    data.table = FALSE,
    showProgress = FALSE
  )
  mat <- as.matrix(data)
  storage.mode(mat) <- "double"
  list(header = hdr, values = mat)
}

vector_from_matrix <- function(mat) as.vector(mat)

lambda_files <- file.path(input_maxent_dir, sprintf("spyl_%d.lambdas", 0:9))
if (!all(file.exists(lambda_files))) {
  stop("Missing .lambdas files: ", paste(lambda_files[!file.exists(lambda_files)], collapse = ", "))
}

log_msg("Parsing Java MaxEnt .lambdas files")
lambdas <- lapply(lambda_files, parse_lambdas)

feature_audit <- dplyr::bind_rows(lapply(seq_along(lambdas), function(i) {
  cbind(model = sprintf("spyl_%d", i - 1L), lambdas[[i]]$features)
}))
constant_audit <- dplyr::bind_rows(lapply(seq_along(lambdas), function(i) {
  data.frame(
    model = sprintf("spyl_%d", i - 1L),
    constant = names(lambdas[[i]]$constants),
    value = unlist(lambdas[[i]]$constants),
    row.names = NULL
  )
}))
data.table::fwrite(feature_audit, file.path(table_dir, "region_lambdas_feature_terms.tsv"), sep = "\t")
data.table::fwrite(constant_audit, file.path(table_dir, "region_lambdas_constants.tsv"), sep = "\t")

## ----------------------------------------------------------------------
## 3. 一致性验证一：与 Java MaxEnt 样点预测表比较
## ----------------------------------------------------------------------

## Java MaxEnt会为训练样点和测试样点输出预测值表。
## 本步骤在相同样点上重新计算raw和logistic预测值，并与Java输出逐项比较。
## 如果.lambdas重建过程正确，两组结果应高度一致。

sample_env_path <- file.path(prepared_table_dir, "sample_environment_values_by_replicate.tsv")
if (!file.exists(sample_env_path)) {
  stop("Missing sample environment table: ", sample_env_path)
}

log_msg("Validating reconstructed predictions against Java samplePredictions.csv")
sample_env_all <- data.table::fread(sample_env_path, data.table = FALSE)
sample_validation <- list()

for (i in 0:9) {
  model_name <- sprintf("spyl_%d", i)
  java_sample_file <- file.path(input_maxent_dir, sprintf("%s_samplePredictions.csv", model_name))
  java_sample <- read.csv(java_sample_file, check.names = FALSE)
  sample_env <- sample_env_all |>
    dplyr::filter(.data$model == model_name)

  if (nrow(sample_env) != nrow(java_sample)) {
    stop("Sample row mismatch for ", model_name)
  }

  env_i <- as.data.frame(lapply(sample_env[, vars], as.numeric))
  names(env_i) <- vars
  pred_raw <- predict_java_maxent(lambdas[[i + 1L]], env_i, "raw")
  pred_logistic <- predict_java_maxent(lambdas[[i + 1L]], env_i, "logistic")

  sample_validation[[i + 1L]] <- data.frame(
    model = model_name,
    n = nrow(java_sample),
    raw_rmse = sqrt(mean((pred_raw - java_sample$`Raw prediction`) ^ 2)),
    raw_max_abs_error = max(abs(pred_raw - java_sample$`Raw prediction`)),
    raw_correlation = cor(pred_raw, java_sample$`Raw prediction`),
    logistic_rmse = sqrt(mean((pred_logistic - java_sample$`Logistic prediction`) ^ 2)),
    logistic_max_abs_error = max(abs(pred_logistic - java_sample$`Logistic prediction`)),
    logistic_correlation = cor(pred_logistic, java_sample$`Logistic prediction`)
  )
}

sample_validation <- dplyr::bind_rows(sample_validation)
data.table::fwrite(sample_validation, file.path(table_dir, "maxent_sample_prediction_validation.tsv"), sep = "\t")

## ----------------------------------------------------------------------
## 4. 一致性验证二：与 Java MaxEnt 栅格预测结果比较
## ----------------------------------------------------------------------

## 样点验证只能说明发生点上的预测一致。本步骤进一步在全部有效环境栅格上
## 进行验证：直接读取14个环境变量栅格，筛选所有无缺失值的有效像元，
## 在每个像元上重新计算logistic预测值，并与Java MaxEnt输出的spyl_*.asc
## 栅格预测值比较。这样可以确认重建函数不仅在样点上正确，也能重建空间预测图。

env_dir <- file.path(project_dir, "00.input", "lay_psyl")
env_asc_files <- file.path(env_dir, paste0(vars, ".asc"))
if (!all(file.exists(env_asc_files))) {
  stop("Missing environmental ASC files: ", paste(env_asc_files[!file.exists(env_asc_files)], collapse = ", "))
}

log_msg("Reading environmental rasters for full raster-level validation")
env_rasters <- lapply(setNames(env_asc_files, vars), read_asc)
env_vectors <- lapply(env_rasters, function(x) vector_from_matrix(x$values))

valid_mask <- Reduce(`&`, Map(function(z, v) {
  nodata <- env_rasters[[v]]$header$nodata_value
  is.finite(z) & z != nodata
}, env_vectors, vars))
valid_cells <- which(valid_mask)

env_all <- as.data.frame(lapply(env_vectors, function(z) z[valid_cells]))
names(env_all) <- vars

full_raster_input_audit <- data.frame(
  metric = c("full_environment_cells", "valid_environment_cells", "environment_variables"),
  value = c(length(valid_mask), length(valid_cells), length(vars))
)
data.table::fwrite(full_raster_input_audit, file.path(table_dir, "maxent_full_raster_input_audit.tsv"), sep = "	")

log_msg("Validating reconstructed predictions against Java logistic rasters on full valid cells")
raster_validation <- list()

for (i in 0:9) {
  model_name <- sprintf("spyl_%d", i)
  java_asc_file <- file.path(input_maxent_dir, sprintf("%s.asc", model_name))
  java_asc <- read_asc(java_asc_file)
  java_logistic <- vector_from_matrix(java_asc$values)[valid_cells]
  pred_logistic <- predict_java_maxent(lambdas[[i + 1L]], env_all, "logistic")

  raster_validation[[i + 1L]] <- data.frame(
    model = model_name,
    n = length(java_logistic),
    logistic_rmse = sqrt(mean((pred_logistic - java_logistic) ^ 2, na.rm = TRUE)),
    logistic_max_abs_error = max(abs(pred_logistic - java_logistic), na.rm = TRUE),
    logistic_correlation = cor(pred_logistic, java_logistic, use = "complete.obs")
  )
}

raster_validation <- dplyr::bind_rows(raster_validation)
data.table::fwrite(raster_validation, file.path(table_dir, "maxent_raster_prediction_validation.tsv"), sep = "	")

rm(env_rasters, env_vectors, env_all, valid_mask)
gc(verbose = FALSE)

## ----------------------------------------------------------------------
## 5. 研究区域、陆地背景样本与平均预测值
## ----------------------------------------------------------------------


## 研究区域为sixian_alt6边界与青海湖流域边界的并集。
## 湖面和水体栅格不作为SHAP和PDP的背景样本。依据土地覆盖和植被字段剔除这些栅格：ldcvqh == 210 或 veg_qh == 0。

region_path <- file.path(prepared_table_dir, "research_area_environment_cells.tsv")
if (!file.exists(region_path)) {
  stop("Missing formal study-area environment table: ", region_path)
}

log_msg("Preparing formal study-area and terrestrial background")
region <- data.table::fread(region_path, data.table = FALSE)
env_region <- as.data.frame(lapply(region[, vars], as.numeric))
names(env_region) <- vars

lake_water_mask <- as.numeric(region$ldcvqh) == 210 | as.numeric(region$veg_qh) == 0
land_ids <- which(!lake_water_mask)
env_land <- env_region[land_ids, , drop = FALSE]

terrestrial_audit <- data.frame(
  metric = c(
    "union_research_area_cells",
    "excluded_lake_or_water_cells",
    "terrestrial_xai_cells",
    "excluded_lake_or_water_fraction",
    "terrestrial_fraction"
  ),
  value = c(
    nrow(region),
    sum(lake_water_mask),
    nrow(env_land),
    sum(lake_water_mask) / nrow(region),
    nrow(env_land) / nrow(region)
  )
)
data.table::fwrite(terrestrial_audit, file.path(table_dir, "region_terrestrial_mask_audit.tsv"), sep = "\t")

log_msg("Recomputing replicate and mean predictions within formal study area")
region_pred_mat <- predict_matrix(lambdas, env_region, "logistic")
region_prediction <- cbind(
  region[, c("cell", "x", "y", "inside_sixian_alt6", "inside_qinghai_lake_basin", "inside_both")],
  is_lake_or_water = lake_water_mask,
  is_terrestrial = !lake_water_mask,
  mean_logistic_prediction = rowMeans(region_pred_mat),
  sd_logistic_prediction = apply(region_pred_mat, 1, sd)
)
data.table::fwrite(region_prediction, file.path(table_dir, "region_xai_prediction_by_cell.tsv"), sep = "\t")

## ----------------------------------------------------------------------
## 6. SHAP 分析：单栅格预测值的变量贡献分解
## ----------------------------------------------------------------------

## SHAP：在某一个栅格上，哪些环境变量提高或降低了MaxEnt预测的
## 生境适宜性。计算时使用经过一致性验证的Java MaxEnt logistic预测函数。
## 背景样本为研究区域内的陆地有效栅格，代表研究区可用于比较的环境条件。
## 对每个重复模型分别计算SHAP值，再在10个重复模型之间汇总。
## 全局变量重要性以陆地有效栅格上绝对SHAP值的均值表示。

if (run_shap) {
  log_msg(
    "Computing SHAP values: eval_cells=", nrow(env_region),
    ", terrestrial_background_cells=", nrow(env_land),
    ", nsim=", shap_nsim
  )

  eval_meta <- cbind(
    eval_id = seq_len(nrow(region)),
    region[, c("cell", "x", "y", "inside_sixian_alt6", "inside_qinghai_lake_basin", "inside_both")],
    is_lake_or_water = lake_water_mask,
    is_terrestrial = !lake_water_mask,
    mean_logistic_prediction = region_prediction$mean_logistic_prediction,
    sd_logistic_prediction = region_prediction$sd_logistic_prediction
  )

  shap_by_model <- list()
  shap_summary_by_model <- list()
  baseline <- numeric(length(lambdas))

  for (i in seq_along(lambdas)) {
    model_name <- sprintf("spyl_%d", i - 1L)
    log_msg("  SHAP replicate ", model_name)
    lambda_obj <- lambdas[[i]]
    wrapper <- function(object, newdata) predict_java_maxent(object, newdata, "logistic")
    baseline[i] <- mean(predict_java_maxent(lambda_obj, env_land, "logistic"), na.rm = TRUE)

    shap_i <- fastshap::explain(
      object = lambda_obj,
      X = env_land,
      pred_wrapper = wrapper,
      nsim = shap_nsim,
      newdata = env_region,
      adjust = TRUE
    )

    shap_i <- as.data.frame(shap_i)
    names(shap_i) <- vars
    shap_by_model[[i]] <- shap_i

    long_i <- cbind(model = model_name, eval_meta, shap_i) |>
      tidyr::pivot_longer(cols = dplyr::all_of(vars), names_to = "variable", values_to = "shap_value")

    if (write_shap_values) {
      data.table::fwrite(
        long_i,
        file.path(table_dir, sprintf("region_fastshap_values_%s.tsv.gz", model_name)),
        sep = "\t"
      )
    }

    shap_summary_by_model[[i]] <- long_i |>
      dplyr::filter(.data$is_terrestrial) |>
      dplyr::group_by(.data$model, .data$variable) |>
      dplyr::summarise(
        mean_abs_shap = mean(abs(.data$shap_value), na.rm = TRUE),
        mean_shap = mean(.data$shap_value, na.rm = TRUE),
        median_shap = median(.data$shap_value, na.rm = TRUE),
        q05_shap = quantile(.data$shap_value, 0.05, na.rm = TRUE),
        q95_shap = quantile(.data$shap_value, 0.95, na.rm = TRUE),
        .groups = "drop"
      )
  }

  mean_shap_df <- Reduce(`+`, shap_by_model) / length(shap_by_model)
  names(mean_shap_df) <- vars

  shap_summary_model <- dplyr::bind_rows(shap_summary_by_model)
  shap_summary <- shap_summary_model |>
    dplyr::group_by(.data$variable) |>
    dplyr::summarise(
      mean_abs_shap_mean = mean(.data$mean_abs_shap),
      mean_abs_shap_sd = sd(.data$mean_abs_shap),
      mean_abs_shap_min = min(.data$mean_abs_shap),
      mean_abs_shap_max = max(.data$mean_abs_shap),
      signed_mean_shap = mean(.data$mean_shap),
      .groups = "drop"
    ) |>
    dplyr::arrange(dplyr::desc(.data$mean_abs_shap_mean))

  data.table::fwrite(shap_summary_model, file.path(table_dir, "region_fastshap_global_importance_by_replicate.tsv"), sep = "\t")
  data.table::fwrite(shap_summary, file.path(table_dir, "region_fastshap_global_importance_summary.tsv"), sep = "\t")

  baseline_mean <- mean(baseline)
  closure <- data.frame(
    eval_meta[, c("eval_id", "cell", "x", "y", "is_lake_or_water", "is_terrestrial", "mean_logistic_prediction")],
    baseline_mean = baseline_mean,
    sum_shap = rowSums(mean_shap_df),
    reconstructed_prediction = baseline_mean + rowSums(mean_shap_df)
  ) |>
    dplyr::mutate(
      closure_error = .data$mean_logistic_prediction - .data$reconstructed_prediction,
      abs_closure_error = abs(.data$closure_error)
    )

  data.table::fwrite(closure, file.path(table_dir, "region_local_shap_additivity_validation.tsv"), sep = "\t")

  shap_qc <- data.frame(
    metric = c(
      "union_region_cells",
      "excluded_lake_or_water_cells",
      "terrestrial_xai_cells",
      "shap_nsim",
      "shap_background_cells",
      "shap_evaluated_cells",
      "baseline_mean",
      "terrestrial_max_abs_closure_error",
      "terrestrial_median_abs_closure_error",
      "terrestrial_p95_abs_closure_error"
    ),
    value = c(
      nrow(region),
      sum(lake_water_mask),
      nrow(env_land),
      shap_nsim,
      nrow(env_land),
      nrow(env_region),
      baseline_mean,
      max(closure$abs_closure_error[land_ids]),
      median(closure$abs_closure_error[land_ids]),
      unname(quantile(closure$abs_closure_error[land_ids], 0.95))
    )
  )
  data.table::fwrite(shap_qc, file.path(table_dir, "region_shap_terrestrial_background_quality_control.tsv"), sep = "\t")

  ## 绘制SHAP汇总图。横轴为SHAP值，纵轴为环境变量；每个点代表一个陆地栅格。
  ## 点的颜色表示该变量在相应栅格上的取值高低，用于同时展示变量贡献方向和变量取值分布。
  order_vars <- shap_summary$variable
  plot_idx <- land_ids
  if (length(plot_idx) > 45000L) plot_idx <- sort(sample(plot_idx, 45000L))
  env_plot <- env_region[plot_idx, , drop = FALSE] |>
    dplyr::mutate(eval_id = plot_idx) |>
    tidyr::pivot_longer(cols = dplyr::all_of(vars), names_to = "variable", values_to = "feature_value")
  shap_plot <- mean_shap_df[plot_idx, , drop = FALSE] |>
    dplyr::mutate(eval_id = plot_idx) |>
    tidyr::pivot_longer(cols = dplyr::all_of(vars), names_to = "variable", values_to = "shap_value") |>
    dplyr::left_join(env_plot, by = c("eval_id", "variable")) |>
    dplyr::group_by(.data$variable) |>
    dplyr::mutate(
      q05_feature = quantile(.data$feature_value, 0.05, na.rm = TRUE),
      q95_feature = quantile(.data$feature_value, 0.95, na.rm = TRUE),
      feature_scaled = ifelse(
        .data$q95_feature > .data$q05_feature,
        pmin(1, pmax(0, (.data$feature_value - .data$q05_feature) / (.data$q95_feature - .data$q05_feature))),
        0.5
      )
    ) |>
    dplyr::ungroup() |>
    dplyr::mutate(variable_label = factor(unname(var_labels[.data$variable]), levels = rev(unname(var_labels[order_vars]))))
  p_shap <- ggplot(shap_plot, aes(x = .data$shap_value, y = .data$variable_label, colour = .data$feature_scaled)) +
    geom_point(alpha = 0.28, size = 0.32, position = position_jitter(height = 0.24, width = 0), stroke = 0) +
    geom_vline(xintercept = 0, linewidth = 0.25, colour = "grey35") +
    scale_colour_gradient(low = "#2166AC", high = "#B2182B", name = "Feature value", breaks = c(0, 1), labels = c("Low", "High")) +
    labs(x = "SHAP value (impact on MaxEnt logistic suitability)", y = NULL) +
    theme_bw(base_size = 10) +
    theme(panel.grid.major.y = element_blank(), legend.position = "right")
  suppressWarnings(ggsave(file.path(figure_dir, "fig_region_shap_summary_beeswarm.png"), p_shap, width = 7.4, height = 5.6, dpi = 360))
  suppressWarnings(ggsave(file.path(figure_dir, "fig_region_shap_summary_beeswarm.pdf"), p_shap, width = 7.4, height = 5.6))

  ## 绘制SHAP全局变量重要性图，用于检查变量贡献强度排序。
  p_imp <- shap_summary |>
    dplyr::mutate(variable_label = unname(var_labels[.data$variable])) |>
    ggplot(aes(x = .data$mean_abs_shap_mean, y = reorder(.data$variable_label, .data$mean_abs_shap_mean))) +
    geom_col(fill = "#496F7A", width = 0.72) +
    geom_errorbarh(
      aes(xmin = .data$mean_abs_shap_min, xmax = .data$mean_abs_shap_max),
      height = 0.2,
      colour = "grey20"
    ) +
    labs(x = "Mean absolute SHAP value", y = NULL) +
    theme_bw(base_size = 10)
  suppressWarnings(ggsave(file.path(figure_dir, "fig_region_shap_global_importance.png"), p_imp, width = 6.8, height = 4.8, dpi = 360))
  suppressWarnings(ggsave(file.path(figure_dir, "fig_region_shap_global_importance.pdf"), p_imp, width = 6.8, height = 4.8))

  saveRDS(
    list(
      baseline = baseline,
      shap_summary = shap_summary,
      shap_qc = shap_qc,
      mean_shap = mean_shap_df
    ),
    file.path(object_dir, "region_shap_terrestrial_background_summary.rds")
  )
} else {
  log_msg("Skipping SHAP because ANALYSIS_RUN_SHAP is FALSE")
}

## ----------------------------------------------------------------------
## 7. PDP 分析：变量边际响应和变量组合效应
## ----------------------------------------------------------------------

## PDP：当某个环境变量或两个环境变量组合发生变化时，MaxEnt预测的
## 平均适宜性如何变化。计算时保持其他变量为陆地背景样本中的真实观测值，
## 仅改变目标变量的取值。因此，PDP反映的是模型预测结果中的平均边际响应，

make_grid_values <- function(x, n = 100L) {
  x <- x[is.finite(x)]
  lo <- unname(quantile(x, 0.01, na.rm = TRUE))
  hi <- unname(quantile(x, 0.99, na.rm = TRUE))
  ux <- sort(unique(x[x >= lo & x <= hi]))
  if (length(ux) <= n) return(ux)
  unique(as.numeric(quantile(x, probs = seq(0.01, 0.99, length.out = n), na.rm = TRUE)))
}

make_even_grid_values <- function(x, n = 50L) {
  x <- x[is.finite(x)]
  lo <- unname(quantile(x, 0.01, na.rm = TRUE))
  hi <- unname(quantile(x, 0.99, na.rm = TRUE))
  if (!is.finite(lo) || !is.finite(hi) || hi <= lo) return(sort(unique(x)))
  seq(lo, hi, length.out = n)
}

nearest_bin_index <- function(x, centers) {
  centers <- sort(unique(centers))
  if (length(centers) == 1L) return(rep(1L, length(x)))
  mids <- (centers[-1L] + centers[-length(centers)]) / 2
  findInterval(x, c(-Inf, mids, Inf), rightmost.closed = TRUE)
}

if (run_pdp) {
  log_msg("Computing 1D PDPs with terrestrial background")

  ## 1D-PDP仅用于连续变量；veg_qh为分类变量，不作为连续响应曲线展示。
  one_d_vars <- c("hfi_qh", "bio9_qh", "bio17_qh", "alt_qhn", "bio2_qh")
  pdp_1d_rep <- list()

  for (v in one_d_vars) {
    log_msg("  1D PDP ", v)
    grid <- make_grid_values(env_land[[v]], n = pdp_1d_grid_n)
    for (g in grid) {
      nd <- env_land
      nd[[v]] <- g
      pred <- predict_matrix(lambdas, nd, "logistic")
      pdp_1d_rep[[length(pdp_1d_rep) + 1L]] <- data.frame(
        variable = v,
        variable_label = unname(var_labels[v]),
        value = g,
        model = sprintf("spyl_%d", 0:9),
        pdp = colMeans(pred, na.rm = TRUE)
      )
    }
  }

  pdp_1d_rep <- dplyr::bind_rows(pdp_1d_rep)
  pdp_1d_summary <- pdp_1d_rep |>
    dplyr::group_by(.data$variable, .data$variable_label, .data$value) |>
    dplyr::summarise(
      pdp_mean = mean(.data$pdp),
      pdp_sd_model = sd(.data$pdp),
      pdp_q05_model = quantile(.data$pdp, 0.05),
      pdp_q95_model = quantile(.data$pdp, 0.95),
      .groups = "drop"
    )

  data.table::fwrite(pdp_1d_rep, file.path(table_dir, "region_pdp_1d_by_replicate.tsv"), sep = "\t")
  data.table::fwrite(pdp_1d_summary, file.path(table_dir, "region_pdp_1d_summary.tsv"), sep = "\t")
  data.table::fwrite(
    pdp_1d_summary |>
      dplyr::group_by(.data$variable, .data$variable_label) |>
      dplyr::slice_max(.data$pdp_mean, n = 1, with_ties = FALSE) |>
      dplyr::ungroup(),
    file.path(table_dir, "region_pdp_1d_optima_summary.tsv"),
    sep = "\t"
  )

  rug_source <- dplyr::bind_rows(lapply(one_d_vars, function(v) {
    data.frame(variable = v, variable_label = unname(var_labels[v]), value = env_land[[v]])
  }))
  rug_df <- rug_source |>
    dplyr::group_by(.data$variable) |>
    dplyr::group_modify(function(.x, .y) dplyr::slice_sample(.x, n = min(2500L, nrow(.x)))) |>
    dplyr::ungroup()

  p_pdp1 <- ggplot() +
    geom_line(data = pdp_1d_rep, aes(x = .data$value, y = .data$pdp, group = .data$model), colour = "grey72", linewidth = 0.22, alpha = 0.75) +
    geom_ribbon(data = pdp_1d_summary, aes(x = .data$value, ymin = .data$pdp_q05_model, ymax = .data$pdp_q95_model), fill = "#9FB5B2", alpha = 0.28) +
    geom_line(data = pdp_1d_summary, aes(x = .data$value, y = .data$pdp_mean), colour = "#1E4E62", linewidth = 0.58) +
    geom_smooth(data = pdp_1d_summary, aes(x = .data$value, y = .data$pdp_mean), method = "loess", se = FALSE, colour = "#B53B2E", linewidth = 0.72, span = 0.45) +
    geom_rug(data = rug_df, aes(x = .data$value), sides = "b", alpha = 0.08, linewidth = 0.12) +
    facet_wrap(~ factor(variable_label, levels = unname(var_labels[one_d_vars])), scales = "free_x", ncol = 3) +
    labs(x = "Predictor value", y = "Partial dependence (logistic suitability)") +
    theme_bw(base_size = 9) +
    theme(strip.background = element_rect(fill = "grey92", colour = "grey55"))

  suppressWarnings(ggsave(file.path(figure_dir, "fig_region_pdp_1d_fig8_style.png"), p_pdp1, width = 9.6, height = 6.0, dpi = 360))
  suppressWarnings(ggsave(file.path(figure_dir, "fig_region_pdp_1d_fig8_style.pdf"), p_pdp1, width = 9.6, height = 6.0))

  log_msg("Computing 2D PDPs with terrestrial background")

  two_d_pairs <- list(
    c("bio17_qh", "hfi_qh"),
    c("bio17_qh", "alt_qhn"),
    c("hfi_qh", "alt_qhn"),
    c("bio17_qh", "bio2_qh")
  )

  ## 分析中使用5000个陆地栅格作为背景样本。

  final_bg_path <- file.path(table_dir, "region_pdp_2d_background_row_ids.tsv")
  if (file.exists(final_bg_path)) {
    final_bg <- data.table::fread(final_bg_path, data.table = FALSE)
    final_bg_rows <- as.integer(final_bg[[1]])
    bg2_idx <- match(final_bg_rows, land_ids)
    bg2_idx <- bg2_idx[!is.na(bg2_idx)]
    log_msg("  Reusing final 2D PDP background rows: ", length(bg2_idx))
  } else {
    bg2_n <- min(pdp_2d_background_n, nrow(env_land))
    bg2_idx <- sort(sample(seq_len(nrow(env_land)), bg2_n))
    log_msg("  Sampling 2D PDP background rows: ", length(bg2_idx))
  }
  bg2 <- env_land[bg2_idx, , drop = FALSE]
  data.table::fwrite(
    data.frame(source_row = land_ids[bg2_idx]),
    file.path(table_dir, "region_pdp_2d_background_row_ids.tsv"),
    sep = "	"
  )

  pdp_2d_rep <- list()
  pdp_2d_support <- list()

  for (pair in two_d_pairs) {
    v1 <- pair[1]
    v2 <- pair[2]
    log_msg("  2D PDP ", v1, " x ", v2)
    grid1 <- make_even_grid_values(env_land[[v1]], n = pdp_2d_grid_n)
    grid2 <- make_even_grid_values(env_land[[v2]], n = pdp_2d_grid_n)

    b1 <- nearest_bin_index(env_land[[v1]], grid1)
    b2 <- nearest_bin_index(env_land[[v2]], grid2)
    support <- as.data.frame(table(b1, b2), stringsAsFactors = FALSE)
    names(support) <- c("bin_1", "bin_2", "support_n")
    support$bin_1 <- as.integer(as.character(support$bin_1))
    support$bin_2 <- as.integer(as.character(support$bin_2))

    pdp_2d_support[[length(pdp_2d_support) + 1L]] <- support |>
      dplyr::mutate(
        variable_1 = v1,
        variable_2 = v2,
        value_1 = grid1[.data$bin_1],
        value_2 = grid2[.data$bin_2]
      ) |>
      dplyr::select("variable_1", "variable_2", "value_1", "value_2", "support_n")

    for (g1 in grid1) {
      for (g2 in grid2) {
        nd <- bg2
        nd[[v1]] <- g1
        nd[[v2]] <- g2
        pred <- predict_matrix(lambdas, nd, "logistic")
        pdp_2d_rep[[length(pdp_2d_rep) + 1L]] <- data.frame(
          variable_1 = v1,
          variable_2 = v2,
          pair_label = paste0(var_labels[v1], " x ", var_labels[v2]),
          value_1 = g1,
          value_2 = g2,
          model = sprintf("spyl_%d", 0:9),
          pdp = colMeans(pred, na.rm = TRUE)
        )
      }
    }
  }

  pdp_2d_rep <- dplyr::bind_rows(pdp_2d_rep)
  pdp_2d_summary <- pdp_2d_rep |>
    dplyr::group_by(.data$variable_1, .data$variable_2, .data$pair_label, .data$value_1, .data$value_2) |>
    dplyr::summarise(
      pdp_mean = mean(.data$pdp),
      pdp_sd_model = sd(.data$pdp),
      pdp_q05_model = quantile(.data$pdp, 0.05),
      pdp_q95_model = quantile(.data$pdp, 0.95),
      .groups = "drop"
    ) |>
    dplyr::left_join(
      dplyr::bind_rows(pdp_2d_support),
      by = c("variable_1", "variable_2", "value_1", "value_2")
    ) |>
    dplyr::mutate(support_n = ifelse(is.na(.data$support_n), 0L, .data$support_n))

  data.table::fwrite(pdp_2d_rep, file.path(table_dir, "region_pdp_2d_by_replicate.tsv"), sep = "\t")
  data.table::fwrite(pdp_2d_summary, file.path(table_dir, "region_pdp_2d_summary.tsv"), sep = "\t")
  data.table::fwrite(
    pdp_2d_summary |>
      dplyr::filter(.data$support_n >= 10L) |>
      dplyr::group_by(.data$variable_1, .data$variable_2, .data$pair_label) |>
      dplyr::slice_max(.data$pdp_mean, n = 1, with_ties = FALSE) |>
      dplyr::ungroup(),
    file.path(table_dir, "region_pdp_2d_supported10_optima_summary.tsv"),
    sep = "\t"
  )

  p_pdp2 <- pdp_2d_summary |>
    dplyr::mutate(pair_label = factor(.data$pair_label, levels = unique(.data$pair_label))) |>
    ggplot(aes(x = .data$value_1, y = .data$value_2, z = .data$pdp_mean)) +
    geom_contour_filled(bins = 9, alpha = 0.98) +
    geom_contour(bins = 9, colour = "grey30", linewidth = 0.18, alpha = 0.65) +
    facet_wrap(~ pair_label, scales = "free", ncol = 2) +
    scale_fill_viridis_d(option = "C", direction = -1, name = "PDP") +
    labs(x = "Predictor 1", y = "Predictor 2") +
    theme_bw(base_size = 9) +
    theme(strip.background = element_rect(fill = "grey92", colour = "grey55"), panel.grid = element_blank(), legend.position = "right")

  suppressWarnings(ggsave(file.path(figure_dir, "fig_region_pdp_2d_fig9_style.png"), p_pdp2, width = 9.6, height = 7.8, dpi = 360))
  suppressWarnings(ggsave(file.path(figure_dir, "fig_region_pdp_2d_fig9_style.pdf"), p_pdp2, width = 9.6, height = 7.8))
} else {
  log_msg("Skipping PDP because ANALYSIS_RUN_PDP is FALSE")
}
