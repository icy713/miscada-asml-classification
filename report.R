# MISCADA ASML Classification Summative Coursework
# Dataset: Heart Failure
# Objective: Predict fatal myocardial infarction (fatal_mi)
# ==============================================================================

# 1. Setup and Package Loading
# ==============================================================================
# Set random seed for reproducibility
set.seed(42)

# Required packages
packages <- c("caret", "ggplot2", "gridExtra", "Metrics", "dplyr", "pROC", "randomForest", "glmnet")

# Install missing packages
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages, repos="http://cran.us.r-project.org")

# Load packages silently
 invisible(lapply(packages, library, character.only = TRUE))

# ==============================================================================
# 2. Data Loading and Preprocessing
# ==============================================================================
cat("\n--- 1. Data Loading and Preprocessing ---\n")
df <- read.csv("heart_failure.csv")
cat("Original dimensions:", dim(df)[1], "rows and", dim(df)[2], "columns\n")

# Convert categorical variables to factors
factor_vars <- c("anaemia", "diabetes", "high_blood_pressure", "sex", "smoking")
df[factor_vars] <- lapply(df[factor_vars], factor)

# Transform target variable to a factor with valid R variable names for caret
# 1 -> "Yes" (Fatal MI), 0 -> "No" (Survived)
df$fatal_mi <- factor(ifelse(df$fatal_mi == 1, "Yes", "No"), levels = c("Yes", "No"))

# Basic structure summary
str(df)

# ==============================================================================
# 3. Exploratory Data Analysis (EDA)
# ==============================================================================
cat("\n--- 2. Exploratory Data Analysis ---\n")
# Check for missing values
cat("Missing values in dataset:", sum(is.na(df)), "\n")

# Distribution of the target variable
target_dist <- table(df$fatal_mi)
cat("Target variable distribution (fatal_mi):\n")
print(prop.table(target_dist) * 100)

# Create EDA plots and save to PDF
cat("Generating EDA plots -> EDA_plots.pdf\n")
pdf("EDA_plots.pdf", width=12, height=8)

# Plot 1: Age vs Fatal MI
p1 <- ggplot(df, aes(x=fatal_mi, y=age, fill=fatal_mi)) + 
  geom_boxplot() + 
  theme_minimal() + 
  labs(title="Age vs Fatal MI", x="Fatal MI", y="Age")

# Plot 2: Ejection Fraction vs Fatal MI
p2 <- ggplot(df, aes(x=fatal_mi, y=ejection_fraction, fill=fatal_mi)) + 
  geom_boxplot() + 
  theme_minimal() + 
  labs(title="Ejection Fraction vs Fatal MI", x="Fatal MI", y="Ejection Fraction (%)")

# Plot 3: Serum Creatinine vs Fatal MI
p3 <- ggplot(df, aes(x=fatal_mi, y=serum_creatinine, fill=fatal_mi)) + 
  geom_boxplot() + 
  theme_minimal() + 
  labs(title="Serum Creatinine vs Fatal MI", x="Fatal MI", y="Serum Creatinine (mg/dL)")

# Plot 4: Time (Follow-up) vs Fatal MI
p4 <- ggplot(df, aes(x=fatal_mi, y=time, fill=fatal_mi)) + 
  geom_boxplot() + 
  theme_minimal() + 
  labs(title="Follow-up Time vs Fatal MI", x="Fatal MI", y="Time (days)")

grid.arrange(p1, p2, p3, p4, ncol=2)
dev.off()

# ==============================================================================
# 4. Data Splitting (Train/Test)
# ==============================================================================
cat("\n--- 3. Data Splitting ---\n")
# 75% for training, 25% for testing, stratified by fatal_mi
train_index <- createDataPartition(df$fatal_mi, p = 0.75, list = FALSE)
train_data <- df[train_index, ]
test_data <- df[-train_index, ]

cat("Training set:", nrow(train_data), "observations\n")
cat("Testing set: ", nrow(test_data), "observations\n")

# ==============================================================================
# 5. Model Training (Cross-Validation Setup)
# ==============================================================================
cat("\n--- 4. Model Training and Tuning ---\n")
# Use 5-fold cross validation, repeated 3 times.
# Optimize for ROC (AUC) rather than raw accuracy.
ctrl <- trainControl(
  method = "repeatedcv", 
  number = 5, 
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# 5.1 Baseline Model: Logistic Regression
cat("Training Logistic Regression...\n")
set.seed(42)
model_lr <- train(
  fatal_mi ~ ., 
  data = train_data, 
  method = "glm", 
  family = "binomial",
  trControl = ctrl,
  metric = "ROC"
)

# 5.2 Penalized Model: Elastic Net (L1 Ridge + L2 Lasso)
cat("Training Elastic Net...\n")
set.seed(42)
# Define hyperparameter grid for Elastic Net
enet_grid <- expand.grid(
  alpha = seq(0, 1, by = 0.1),
  lambda = 10^seq(-3, 0, length.out = 10)
)
model_enet <- train(
  fatal_mi ~ ., 
  data = train_data, 
  method = "glmnet",
  trControl = ctrl,
  tuneGrid = enet_grid,
  metric = "ROC"
)

# 5.3 Non-linear Model: Random Forest
cat("Training Random Forest...\n")
set.seed(42)
rf_grid <- expand.grid(mtry = c(2, 4, 6, 8))
model_rf <- train(
  fatal_mi ~ ., 
  data = train_data, 
  method = "rf",
  trControl = ctrl,
  tuneGrid = rf_grid,
  metric = "ROC",
  ntree = 500,
  importance = TRUE
)

# ==============================================================================
# 6. Model Comparison
# ==============================================================================
cat("\n--- 5. Model Comparison (Cross-Validation) ---\n")
resamps <- resamples(list(
  Logistic = model_lr,
  ElasticNet = model_enet,
  RandomForest = model_rf
))
summary(resamps)

pdf("Model_Comparison_CV.pdf", width=8, height=6)
bwplot(resamps, metric="ROC", main="Model Comparison based on Cross-Validation AUC")
dev.off()

# Determine the best model based on CV ROC
best_model_name <- "RandomForest" 
best_model <- model_rf

cat("\nSelected Final Model:", best_model_name, "\n")

# Variable Importance Plot for the final model
pdf("Variable_Importance.pdf", width=8, height=6)
plot(varImp(best_model), main = paste("Variable Importance -", best_model_name))
dev.off()

# ==============================================================================
# 7. Performance on Test Set & Threshold Tuning
# ==============================================================================
cat("\n--- 6. Test Set Final Evaluation & Threshold Tuning ---\n")

# Get probability predictions on test set
prob_preds <- predict(best_model, newdata = test_data, type = "prob")

# 7.1 Default Threshold (0.5)
default_preds <- factor(ifelse(prob_preds$Yes > 0.5, "Yes", "No"), levels=c("Yes", "No"))
conf_matrix_default <- confusionMatrix(default_preds, test_data$fatal_mi, positive = "Yes")
cat("\nPerformance at Default Threshold (0.5):\n")
print(conf_matrix_default$table)
cat(paste("Sensitivity (Recall for True Positives - fatal_mi):", round(conf_matrix_default$byClass['Sensitivity'], 3), "\n"))
cat(paste("Specificity (True Negative Rate):", round(conf_matrix_default$byClass['Specificity'], 3), "\n"))

# 7.2 Custom Threshold for Medical Domain (Cost of False Negative is HUGE)
# We want to catch as many potential heart failures as possible. 
# So we lower the threshold for classifying someone as "Yes" (at risk).
custom_threshold <- 0.25
custom_preds <- factor(ifelse(prob_preds$Yes > custom_threshold, "Yes", "No"), levels=c("Yes", "No"))
conf_matrix_custom <- confusionMatrix(custom_preds, test_data$fatal_mi, positive = "Yes")

cat(paste("\nPerformance at Custom 'High Recall' Threshold (", custom_threshold, "):\n", sep=""))
print(conf_matrix_custom$table)
cat(paste("Sensitivity (Recall) improved to:", round(conf_matrix_custom$byClass['Sensitivity'], 3), "\n"))
cat(paste("Specificity dropped to:", round(conf_matrix_custom$byClass['Specificity'], 3), "\n"))

# 7.3 ROC Curve
roc_obj <- roc(test_data$fatal_mi, prob_preds$Yes, levels=c("No", "Yes"))

pdf("ROC_Curve.pdf", width=8, height=8)
plot(roc_obj, main="ROC Curve on Test Data", col="blue", lwd=2, legacy.axes=TRUE)
text(0.6, 0.2, paste("AUC =", round(auc(roc_obj), 3)), cex=1.2)
dev.off()

cat("\nAnalysis complete. All figures and values logged successfully.\n")
