# ==============================================================================
# MISCADA ASML Classification Summative Coursework
# Dataset: Heart Failure Clinical Records
# Objective: Predict whether a patient will suffer a fatal myocardial
#            infarction during the follow-up period (fatal_mi).
#
# Target variable definition: 'fatal_mi' is defined by the course data
# description as "whether the patient suffered a fatal myocardial infarction
# during the follow-up period" (see heart_failure.txt provided by lecturer).
#
# NOTE: The variable 'time' (follow-up period in days) is EXCLUDED from all
# models. It represents the observation window length and would not be
# available at the point of clinical prediction — including it would
# constitute data leakage ("future information").
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Setup and Package Loading
# ------------------------------------------------------------------------------
set.seed(42)

packages <- c("caret", "ggplot2", "gridExtra", "pROC", "randomForest", "glmnet")

new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
if (length(new_packages)) {
  install.packages(new_packages, repos = "https://cloud.r-project.org")
}

invisible(lapply(packages, library, character.only = TRUE))

# ------------------------------------------------------------------------------
# 2. Data Loading and Preprocessing
# ------------------------------------------------------------------------------
cat("\n===== 1. Data Loading and Preprocessing =====\n")
df <- read.csv("heart_failure.csv")
cat("Dimensions:", nrow(df), "rows x", ncol(df), "columns\n\n")

# Convert binary categorical variables to factors
factor_vars <- c("anaemia", "diabetes", "high_blood_pressure", "sex", "smoking")
df[factor_vars] <- lapply(df[factor_vars], factor)

# Convert target to factor:
#   1 -> "Yes" (patient suffered fatal MI during follow-up)
#   0 -> "No"  (patient did not suffer fatal MI during follow-up)
df$fatal_mi <- factor(ifelse(df$fatal_mi == 1, "Yes", "No"), levels = c("Yes", "No"))

# IMPORTANT: Remove 'time' — it is the follow-up period and leaks future info
df$time <- NULL
cat("'time' variable removed to prevent data leakage.\n")

# Define predictor variables explicitly
predictor_vars <- c(
  "age", "anaemia", "creatinine_phosphokinase", "diabetes",
  "ejection_fraction", "high_blood_pressure", "platelets",
  "serum_creatinine", "serum_sodium", "sex", "smoking"
)

# Build explicit formula (instead of fatal_mi ~ .)
model_formula <- as.formula(paste(
  "fatal_mi ~",
  paste(predictor_vars, collapse = " + ")
))
cat("Model formula:", deparse(model_formula), "\n")

# Summary
str(df)
cat("\nSummary statistics:\n")
summary(df)

# ------------------------------------------------------------------------------
# 3. Exploratory Data Analysis (EDA)
# ------------------------------------------------------------------------------
cat("\n===== 2. Exploratory Data Analysis =====\n")
cat("Missing values:", sum(is.na(df)), "\n")

target_dist <- table(df$fatal_mi)
cat("Target distribution (fatal_mi):\n")
print(prop.table(target_dist) * 100)

cat("Generating EDA plots -> EDA_plots.pdf\n")
pdf("EDA_plots.pdf", width = 12, height = 10)

p1 <- ggplot(df, aes(x = fatal_mi, y = age, fill = fatal_mi)) +
  geom_boxplot(alpha = 0.7) +
  scale_fill_manual(values = c("#E74C3C", "#2ECC71")) +
  theme_minimal(base_size = 13) +
  labs(title = "Age by Fatal MI Status", x = "Fatal MI", y = "Age (years)") +
  theme(legend.position = "none")

p2 <- ggplot(df, aes(x = fatal_mi, y = ejection_fraction, fill = fatal_mi)) +
  geom_boxplot(alpha = 0.7) +
  scale_fill_manual(values = c("#E74C3C", "#2ECC71")) +
  theme_minimal(base_size = 13) +
  labs(
    title = "Ejection Fraction by Fatal MI Status", x = "Fatal MI",
    y = "Ejection Fraction (%)"
  ) +
  theme(legend.position = "none")

p3 <- ggplot(df, aes(x = fatal_mi, y = serum_creatinine, fill = fatal_mi)) +
  geom_boxplot(alpha = 0.7) +
  scale_fill_manual(values = c("#E74C3C", "#2ECC71")) +
  theme_minimal(base_size = 13) +
  labs(
    title = "Serum Creatinine by Fatal MI Status", x = "Fatal MI",
    y = "Serum Creatinine (mg/dL)"
  ) +
  theme(legend.position = "none")

p4 <- ggplot(df, aes(x = fatal_mi, y = serum_sodium, fill = fatal_mi)) +
  geom_boxplot(alpha = 0.7) +
  scale_fill_manual(values = c("#E74C3C", "#2ECC71")) +
  theme_minimal(base_size = 13) +
  labs(
    title = "Serum Sodium by Fatal MI Status", x = "Fatal MI",
    y = "Serum Sodium (mEq/L)"
  ) +
  theme(legend.position = "none")

grid.arrange(p1, p2, p3, p4,
  ncol = 2,
  top = "Exploratory Data Analysis: Key Clinical Variables"
)
dev.off()

# Correlation heatmap for continuous variables
cat("Generating correlation heatmap -> Correlation_Heatmap.pdf\n")
cont_vars <- c(
  "age", "creatinine_phosphokinase", "ejection_fraction",
  "platelets", "serum_creatinine", "serum_sodium"
)
cor_matrix <- cor(df[, cont_vars])

pdf("Correlation_Heatmap.pdf", width = 8, height = 7)
cor_melted <- expand.grid(Var1 = cont_vars, Var2 = cont_vars)
cor_melted$value <- as.vector(cor_matrix)
ggplot(cor_melted, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(value, 2)), size = 3.5) +
  scale_fill_gradient2(
    low = "#3498DB", mid = "white", high = "#E74C3C",
    midpoint = 0, limits = c(-1, 1)
  ) +
  theme_minimal(base_size = 12) +
  labs(
    title = "Correlation Matrix of Continuous Predictors",
    x = "", y = "", fill = "Correlation"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
dev.off()

# ------------------------------------------------------------------------------
# 4. Data Splitting (Train / Test)
# ------------------------------------------------------------------------------
cat("\n===== 3. Data Splitting =====\n")
# 75% train, 25% test — stratified split to maintain class proportions
train_index <- createDataPartition(df$fatal_mi, p = 0.75, list = FALSE)
train_data <- df[train_index, ]
test_data <- df[-train_index, ]

cat("Training set:", nrow(train_data), "observations\n")
cat("Test set:    ", nrow(test_data), "observations\n")

# ------------------------------------------------------------------------------
# 5. Cross-Validation Setup
# ------------------------------------------------------------------------------
cat("\n===== 4. Model Training and Hyperparameter Tuning =====\n")

ctrl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# 5.1 Logistic Regression (Baseline)
cat("Training Logistic Regression (baseline)...\n")
set.seed(42)
model_lr <- train(
  model_formula,
  data = train_data,
  method = "glm",
  family = "binomial",
  trControl = ctrl,
  metric = "ROC"
)

# 5.2 Elastic Net (penalised logistic regression)
cat("Training Elastic Net (regularised)...\n")
set.seed(42)
enet_grid <- expand.grid(
  alpha  = seq(0, 1, by = 0.1),
  lambda = 10^seq(-4, 0, length.out = 20)
)
model_enet <- train(
  model_formula,
  data = train_data,
  method = "glmnet",
  trControl = ctrl,
  tuneGrid = enet_grid,
  metric = "ROC"
)
cat(
  "  Best alpha:", model_enet$bestTune$alpha,
  " Best lambda:", round(model_enet$bestTune$lambda, 5), "\n"
)

# 5.3 Random Forest
cat("Training Random Forest...\n")
set.seed(42)
rf_grid <- expand.grid(mtry = 2:8)
model_rf <- train(
  model_formula,
  data = train_data,
  method = "rf",
  trControl = ctrl,
  tuneGrid = rf_grid,
  metric = "ROC",
  ntree = 500,
  importance = TRUE
)
cat("  Best mtry:", model_rf$bestTune$mtry, "\n")

# ------------------------------------------------------------------------------
# 6. Model Comparison (Programmatic Best Model Selection)
# ------------------------------------------------------------------------------
cat("\n===== 5. Model Comparison (Cross-Validation) =====\n")

model_list <- list(
  Logistic     = model_lr,
  ElasticNet   = model_enet,
  RandomForest = model_rf
)
resamps <- resamples(model_list)
cat("\nCV Summary:\n")
print(summary(resamps))

# Programmatically select the best model based on best CV ROC
# (i.e., the ROC achieved at the optimal hyperparameter configuration)
cv_roc_best <- sapply(model_list, function(m) {
  max(m$results$ROC)
})
cat("\nBest CV ROC for each model (at optimal hyperparameters):\n")
print(round(cv_roc_best, 4))

best_model_name <- names(which.max(cv_roc_best))
best_model <- model_list[[best_model_name]]
cat(
  "\n>>> Best model selected automatically:", best_model_name,
  "(Best CV ROC =", round(max(cv_roc_best), 4), ") <<<\n"
)

# CV comparison boxplot
pdf("Model_Comparison_CV.pdf", width = 10, height = 7)
bwplot(resamps,
  metric = "ROC",
  main = "Cross-Validation AUC Comparison (10-fold, 3 repeats)"
)
dev.off()

# Variable importance
pdf("Variable_Importance.pdf", width = 8, height = 6)
plot(varImp(best_model),
  main = paste("Variable Importance -", best_model_name)
)
dev.off()

# ------------------------------------------------------------------------------
# 7. Test Set Evaluation
# ------------------------------------------------------------------------------
cat("\n===== 6. Test Set Evaluation =====\n")

prob_preds <- predict(best_model, newdata = test_data, type = "prob")

# 7.1 Default threshold (0.5)
preds_default <- factor(ifelse(prob_preds$Yes > 0.5, "Yes", "No"),
  levels = c("Yes", "No")
)
cm_default <- confusionMatrix(preds_default, test_data$fatal_mi, positive = "Yes")
cat("\n--- Threshold = 0.5 ---\n")
print(cm_default$table)
cat("Sensitivity:", round(cm_default$byClass["Sensitivity"], 3), "\n")
cat("Specificity:", round(cm_default$byClass["Specificity"], 3), "\n")
cat("Accuracy:   ", round(cm_default$overall["Accuracy"], 3), "\n")

# 7.2 ROC curve and AUC
roc_obj <- roc(test_data$fatal_mi, prob_preds$Yes, levels = c("No", "Yes"))
cat("\nTest AUC:", round(auc(roc_obj), 4), "\n")

pdf("ROC_Curve.pdf", width = 8, height = 7)
plot(roc_obj,
  main = paste("ROC Curve on Test Set -", best_model_name),
  col = "#2980B9", lwd = 2.5, legacy.axes = TRUE
)
abline(a = 0, b = 1, lty = 2, col = "grey50")
text(0.55, 0.15, paste("AUC =", round(auc(roc_obj), 3)),
  cex = 1.3, col = "#2980B9", font = 2
)
dev.off()

# ------------------------------------------------------------------------------
# 8. Probability Calibration Assessment
# ------------------------------------------------------------------------------
cat("\n===== 7. Probability Calibration =====\n")

# Brier Score (lower is better; perfect = 0)
actual_binary <- as.numeric(test_data$fatal_mi == "Yes")
brier_score <- mean((prob_preds$Yes - actual_binary)^2)
cat("Brier Score:", round(brier_score, 4), "\n")

# Calibration plot: bin predicted probabilities and compare to observed rates
n_bins <- 10
cal_df <- data.frame(
  predicted = prob_preds$Yes,
  actual    = actual_binary
)
cal_df$bin <- cut(cal_df$predicted,
  breaks = seq(0, 1, length.out = n_bins + 1),
  include.lowest = TRUE
)

cal_summary <- aggregate(cbind(predicted, actual) ~ bin, data = cal_df, FUN = mean)
cal_counts <- aggregate(actual ~ bin, data = cal_df, FUN = length)
names(cal_counts)[2] <- "n"
cal_summary <- merge(cal_summary, cal_counts, by = "bin")

cat("\nCalibration table (predicted vs observed by decile):\n")
print(cal_summary)

# Calibration intercept and slope via logistic regression on logit scale
# Standard approach: regress actual outcome on logit(predicted probability)
cal_df$logit_pred <- log(pmax(cal_df$predicted, 1e-8) /
  pmax(1 - cal_df$predicted, 1e-8))
cal_model <- glm(actual ~ logit_pred, data = cal_df, family = "binomial")
cal_intercept <- coef(cal_model)[1]
cal_slope <- coef(cal_model)[2]
cat(
  "\nCalibration intercept (logit scale):", round(cal_intercept, 3),
  "(ideal = 0)\n"
)
cat(
  "Calibration slope (logit scale):    ", round(cal_slope, 3),
  "(ideal = 1)\n"
)

pdf("Calibration_Plot.pdf", width = 8, height = 7)
ggplot(cal_summary, aes(x = predicted, y = actual)) +
  geom_point(aes(size = n), colour = "#E74C3C", alpha = 0.8) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "grey40") +
  scale_size_continuous(range = c(3, 10), name = "Bin Count") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal(base_size = 13) +
  labs(
    title = paste("Calibration Plot -", best_model_name),
    subtitle = paste0(
      "Brier Score = ", round(brier_score, 4),
      "  |  Cal. Intercept = ", round(cal_intercept, 2),
      "  |  Cal. Slope = ", round(cal_slope, 2)
    ),
    x = "Mean Predicted Probability",
    y = "Observed Proportion of Fatal MI"
  )
dev.off()

# ------------------------------------------------------------------------------
# 9. Threshold Analysis (Illustrative Post-Model Analysis)
# ------------------------------------------------------------------------------
cat("\n===== 8. Threshold Sweep & Cost-Sensitive Analysis =====\n")

# In a clinical setting: Missing a fatal MI (False Negative) is far worse
# than a false alarm (False Positive). We illustrate how varying the
# decision threshold affects Sensitivity vs Specificity on the test set.
# NOTE: This is an exploratory analysis to demonstrate the trade-off,
# not a formal threshold optimisation process.
thresholds <- seq(0.05, 0.95, by = 0.05)
threshold_results <- data.frame(
  threshold   = thresholds,
  sensitivity = NA,
  specificity = NA,
  ppv         = NA,
  npv         = NA,
  f1          = NA
)

for (i in seq_along(thresholds)) {
  t <- thresholds[i]
  preds_t <- factor(ifelse(prob_preds$Yes > t, "Yes", "No"),
    levels = c("Yes", "No")
  )
  cm_t <- confusionMatrix(preds_t, test_data$fatal_mi, positive = "Yes")
  threshold_results$sensitivity[i] <- cm_t$byClass["Sensitivity"]
  threshold_results$specificity[i] <- cm_t$byClass["Specificity"]
  threshold_results$ppv[i] <- cm_t$byClass["Pos Pred Value"]
  threshold_results$npv[i] <- cm_t$byClass["Neg Pred Value"]
  prec <- cm_t$byClass["Pos Pred Value"]
  rec <- cm_t$byClass["Sensitivity"]
  threshold_results$f1[i] <- ifelse(is.na(prec) | is.na(rec) | (prec + rec) == 0,
    0, 2 * prec * rec / (prec + rec)
  )
}

cat("\nThreshold sweep results:\n")
print(threshold_results)

# Illustrative clinical threshold: prioritise high Sensitivity (>= 0.85)
clinical_candidates <- threshold_results[
  !is.na(threshold_results$sensitivity) & threshold_results$sensitivity >= 0.85,
]
if (nrow(clinical_candidates) > 0) {
  best_clinical <- clinical_candidates[which.max(clinical_candidates$specificity), ]
  optimal_threshold <- best_clinical$threshold
} else {
  optimal_threshold <- 0.30
}
cat("\nIllustrative clinical threshold (Sensitivity >= 0.85):", optimal_threshold, "\n")

# Threshold sweep plot
pdf("Threshold_Sweep.pdf", width = 10, height = 7)
par(mar = c(5, 4, 4, 2))
plot(threshold_results$threshold, threshold_results$sensitivity,
  type = "l", col = "#E74C3C", lwd = 2.5, ylim = c(0, 1),
  xlab = "Decision Threshold", ylab = "Rate",
  main = "Threshold Sweep: Sensitivity vs Specificity Trade-off"
)
lines(threshold_results$threshold, threshold_results$specificity,
  col = "#2980B9", lwd = 2.5
)
lines(threshold_results$threshold, threshold_results$f1,
  col = "#27AE60", lwd = 2, lty = 2
)
abline(v = 0.5, lty = 3, col = "grey50")
abline(v = optimal_threshold, lty = 2, col = "#8E44AD", lwd = 2)
legend("right",
  legend = c(
    "Sensitivity", "Specificity", "F1 Score",
    "Default (0.5)", paste0("Clinical (", optimal_threshold, ")")
  ),
  col = c("#E74C3C", "#2980B9", "#27AE60", "grey50", "#8E44AD"),
  lty = c(1, 1, 2, 3, 2), lwd = c(2.5, 2.5, 2, 1, 2), cex = 0.9
)
dev.off()

# Performance at the illustrative clinical threshold
cat("\n--- Performance at Illustrative Clinical Threshold (", optimal_threshold, ") ---\n")
preds_clinical <- factor(ifelse(prob_preds$Yes > optimal_threshold, "Yes", "No"),
  levels = c("Yes", "No")
)
cm_clinical <- confusionMatrix(preds_clinical, test_data$fatal_mi, positive = "Yes")
print(cm_clinical$table)
cat("Sensitivity:", round(cm_clinical$byClass["Sensitivity"], 3), "\n")
cat("Specificity:", round(cm_clinical$byClass["Specificity"], 3), "\n")

# ------------------------------------------------------------------------------
cat("\n===== 9. Precision-Recall Trade-off =====\n")

# NOTE: This is a discrete threshold-based approximation of the
# precision-recall relationship, not a continuous PR curve.
# It shows how precision and recall trade off at different decision
# thresholds, which is directly relevant to clinical decision-making.
pr_data <- data.frame(
  threshold = threshold_results$threshold,
  precision = threshold_results$ppv,
  recall    = threshold_results$sensitivity
)
pr_data <- pr_data[!is.na(pr_data$precision) & !is.na(pr_data$recall), ]

pdf("Precision_Recall_Tradeoff.pdf", width = 8, height = 7)
ggplot(pr_data, aes(x = recall, y = precision)) +
  geom_line(colour = "#E67E22", linewidth = 1.2) +
  geom_point(aes(colour = threshold), size = 3) +
  scale_colour_gradient(low = "#2ECC71", high = "#E74C3C", name = "Threshold") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal(base_size = 13) +
  labs(
    title = paste("Precision-Recall Trade-off -", best_model_name),
    subtitle = "Discrete threshold sweep (test set)",
    x = "Recall (Sensitivity)", y = "Precision (PPV)"
  )
dev.off()

# ------------------------------------------------------------------------------
# 11. Session Information (for reproducibility)
# ------------------------------------------------------------------------------
cat("\n===== 10. Session Information =====\n")
sessionInfo()

cat("\n\n====================================================\n")
cat("Analysis complete. All figures saved to working directory.\n")
cat("====================================================\n")
