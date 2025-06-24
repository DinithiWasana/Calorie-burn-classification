# Load necessary libraries
library(plumber)
library(ranger)
library(randomForest)  # For rf models in tier_models
library(nnet)         # For multinom models in specialized_models
library(dplyr)
library(jsonlite)
library(caret)

# Log file for debugging
log_file <- "prediction_log.txt"

# Load the model (only once at the start)
load("../models/calorie_prediction_model.RData")  # Loads MI_model, tier_models, specialized_models, etc.

# Check if all required models are loaded successfully
if (exists("MI_model") && exists("tier_models") && exists("specialized_models")) {
  cat("All models loaded successfully!\n", file = log_file, append = TRUE)
} else {
  cat("Error: One or more models failed to load.\n", file = log_file, append = TRUE)
  stop("Required models are missing.")
}

# Enable CORS by adding a filter
#* @filter cors
cors <- function(req, res) {
  res$setHeader("Access-Control-Allow-Origin", "*")  # Allow all origins (for development)
  res$setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
  res$setHeader("Access-Control-Allow-Headers", "Content-Type")

  # Handle preflight OPTIONS request
  if (req$REQUEST_METHOD == "OPTIONS") {
    res$status <- 200
    return(list())
  }

  plumber::forward()
}

# Function to predict for constant_model (from calorie_model.R)
predict.constant_model <- function(object, newdata, type = "raw", ...) {
  if (type == "raw") {
    # Return a factor of the single class
    return(factor(rep(object$single_class, nrow(newdata)), levels = object$all_classes))
  } else if (type == "prob") {
    # Create a probability matrix with 1 for the single class, 0 for others
    probs <- matrix(0, nrow = nrow(newdata), ncol = length(object$all_classes))
    colnames(probs) <- object$all_classes
    probs[, object$single_class] <- 1
    return(as.data.frame(probs))
  } else {
    stop("Type must be either 'raw' or 'prob'")
  }
}

# Prediction function from calorie_model.R
predict_calories_class <- function(new_data, MI_model, tier_models, specialized_models) {
  log_file <- "prediction_log.txt"  # Log file to capture debug info
  
  # Log the incoming new data
  cat("Received new data:\n", file = log_file, append = TRUE)
  write.table(new_data, file = log_file, append = TRUE, col.names = !file.exists(log_file), row.names = FALSE)
  
  # Step 1: Preprocess the data (create the same features as in training)
  new_data <- new_data %>%
    mutate(
      BMI = weight / ((height / 100)^2),
      heartRateRatio = heartRate / restingHeartRate,
      cardioStress = (heartRate - restingHeartRate) / restingHeartRate
    )
  
  # Log the mutated data
  cat("\nMutated Data:\n", file = log_file, append = TRUE)
  write.table(new_data, file = log_file, append = TRUE, col.names = !file.exists(log_file), row.names = FALSE)
  
  # Handle NA values in derived features
  new_data <- new_data %>%
    mutate(across(c(BMI, heartRateRatio, cardioStress),
                  ~ifelse(is.infinite(.) | is.na(.), median(., na.rm = TRUE), .)))
  
  # Step 2: Predict metabolicIntensity
  predictions <- tryCatch({
    predict(MI_model, new_data)  # Default type for regression in caret::train
  }, error = function(e) {
    cat("\nError while predicting metabolicIntensity:\n", e$message, "\n", file = log_file, append = TRUE)
    stop("Error during metabolicIntensity prediction.")
  })
  
  new_data$predictedMetabolicIntensity <- predictions
  
  # Log the predicted metabolic intensity
  cat("\nPredicted Metabolic Intensity:\n", file = log_file, append = TRUE)
  write.table(predictions, file = log_file, append = TRUE, col.names = !file.exists(log_file), row.names = FALSE)
  
  # Handle any NA values in the predictedMetabolicIntensity
  if (any(is.na(new_data$predictedMetabolicIntensity))) {
    med_val <- median(new_data$predictedMetabolicIntensity, na.rm = TRUE)
    new_data$predictedMetabolicIntensity[is.na(new_data$predictedMetabolicIntensity)] <- med_val
  }
  
  # Step 3: Convert categorical variables to factors with consistent levels
  categorical_vars <- c("gender", "workoutType", "workoutIntensity", 
                        "moodBeforeWorkout", "moodAfterWorkout")
  
  # Define expected factor levels (based on training data)
  expected_levels <- list(
    gender = c("Male", "Female"),
    workoutType = c("Running", "Yoga", "Cycling", "HIIT"),
    workoutIntensity = c("Low", "Medium", "High"),
    moodBeforeWorkout = c("Neutral", "Happy", "Stressed"),
    moodAfterWorkout = c("Neutral", "Energized", "Tired")
  )
  
  new_data <- new_data %>%
    mutate(across(all_of(categorical_vars), ~factor(., levels = expected_levels[[cur_column()]])))
  
  # Handle NA in categorical variables
  for (col in categorical_vars) {
    if (any(is.na(new_data[[col]]))) {
      new_data[[col]][is.na(new_data[[col]])] <- expected_levels[[col]][1]  # Use first level as default
    }
  }
  
  # Handle NA in numeric variables
  for (col in names(new_data)) {
    if (is.numeric(new_data[[col]]) && any(is.na(new_data[[col]]))) {
      new_data[[col]][is.na(new_data[[col]])] <- median(new_data[[col]], na.rm = TRUE)
    }
  }
  
  # Step 4: Add placeholder for caloriesClass (required by prediction function)
  new_data$caloriesClass <- factor(rep("Low", nrow(new_data)), 
                                   levels = c("Low", "Medium.Low", "Medium.High", "High"))
  
  # Step 5: Make predictions using the ensemble model
  predictions <- tryCatch({
    predict_with_multinom_ensemble(tier_models, specialized_models, new_data)
  }, error = function(e) {
    cat("\nError while predicting with ensemble:\n", e$message, "\n", file = log_file, append = TRUE)
    stop("Error during ensemble prediction.")
  })
  
  # Log the final predictions
  cat("\nFinal Predictions (Class):\n", file = log_file, append = TRUE)
  cat(paste(predictions$class_predictions, collapse = ", "), "\n", file = log_file, append = TRUE)
  
  return(predictions)
}

# Include the predict_with_multinom_ensemble function from calorie_model.R
predict_with_multinom_ensemble <- function(tier_models, specialized_models, new_data) {
  # Get tier predictions
  is_low_pred <- tryCatch({
    predict(tier_models$low, new_data, type = "prob")[,"Yes"]
  }, error = function(e) {
    cat("\nError predicting low tier:\n", e$message, "\n", file = log_file, append = TRUE)
    stop("Error in low tier prediction.")
  })
  
  is_high_pred <- tryCatch({
    predict(tier_models$high, new_data, type = "prob")[,"Yes"]
  }, error = function(e) {
    cat("\nError predicting high tier:\n", e$message, "\n", file = log_file, append = TRUE)
    stop("Error in high tier prediction.")
  })
  
  # Determine tier probabilities
  tier_probs <- data.frame(
    low = is_low_pred,
    mid = pmax(0, 1 - is_low_pred - is_high_pred), 
    high = is_high_pred
  )
  
  # Normalize probabilities
  tier_probs <- tier_probs / rowSums(tier_probs)
  
  # Log tier probabilities
  cat("\nTier Probabilities:\n", file = log_file, append = TRUE)
  write.table(tier_probs, file = log_file, append = TRUE, col.names = TRUE, row.names = FALSE)
  
  # Get the class levels
  all_classes <- levels(new_data$caloriesClass)
  
  # Get predictions from each specialized multinom model
  models <- specialized_models$multinom
  
  # Function to safely get predictions
  get_safe_predictions <- function(model, tier_name) {
    if (is.null(model)) {
      cat("\nWarning: Model for", tier_name, "is NULL. Returning uniform probabilities.\n", file = log_file, append = TRUE)
      result <- matrix(1/length(all_classes), nrow = nrow(new_data), ncol = length(all_classes))
      colnames(result) <- all_classes
      return(as.data.frame(result))
    }
    
    if (inherits(model, "constant_model")) {
      cat("\nUsing constant model for", tier_name, "\n", file = log_file, append = TRUE)
      return(predict.constant_model(model, new_data, type = "prob"))
    }
    
    if (attr(model, "type") == "subset_model") {
      cat("\nUsing subset model for", tier_name, "\n", file = log_file, append = TRUE)
      avail_classes <- attr(model, "available_classes")
      raw_preds <- predict(model, new_data, type = "prob")
      full_pred <- matrix(0, nrow = nrow(new_data), ncol = length(all_classes))
      colnames(full_pred) <- all_classes
      
      for (cls in avail_classes) {
        full_pred[, cls] <- raw_preds[, cls]
      }
      
      row_sums <- rowSums(full_pred)
      idx_zero <- row_sums == 0
      if (any(idx_zero)) {
        full_pred[idx_zero, ] <- 1/length(all_classes)
      }
      row_sums <- ifelse(row_sums == 0, 1, row_sums)
      full_pred <- full_pred / row_sums
      
      return(as.data.frame(full_pred))
    }
    
    cat("\nUsing standard model for", tier_name, "\n", file = log_file, append = TRUE)
    return(predict(model, new_data, type = "prob"))
  }
  
  # Get predictions from each tier model
  low_preds <- get_safe_predictions(models$low, "low tier")
  mid_preds <- get_safe_predictions(models$mid, "mid tier")
  high_preds <- get_safe_predictions(models$high, "high tier")
  
  # Log specialized model predictions
  cat("\nLow Tier Predictions:\n", file = log_file, append = TRUE)
  write.table(low_preds, file = log_file, append = TRUE, col.names = TRUE, row.names = FALSE)
  cat("\nMid Tier Predictions:\n", file = log_file, append = TRUE)
  write.table(mid_preds, file = log_file, append = TRUE, col.names = TRUE, row.names = FALSE)
  cat("\nHigh Tier Predictions:\n", file = log_file, append = TRUE)
  write.table(high_preds, file = log_file, append = TRUE, col.names = TRUE, row.names = FALSE)
  
  # Create weighted predictions
  weighted_preds <- tier_probs$low * low_preds + 
    tier_probs$mid * mid_preds + 
    tier_probs$high * high_preds
  
  # Log weighted predictions
  cat("\nWeighted Predictions:\n", file = log_file, append = TRUE)
  write.table(weighted_preds, file = log_file, append = TRUE, col.names = TRUE, row.names = FALSE)
  
  # Get final class predictions
  final_pred_class <- factor(colnames(weighted_preds)[apply(weighted_preds, 1, which.max)],
                             levels = all_classes)
  
  return(list(
    class_predictions = final_pred_class,
    class_probabilities = weighted_preds,
    tier_probabilities = tier_probs
  ))
}

# Plumber API endpoint
#* @post /predict
#* @serializer json
function(req) {
  # Parse incoming JSON
  input_data <- tryCatch({
    fromJSON(req$postBody)
  }, error = function(e) {
    cat("\nError parsing JSON input:\n", e$message, "\n", file = log_file, append = TRUE)
    stop("Invalid JSON input.")
  })
  
  # Convert to data.frame
  df <- as.data.frame(input_data)
  
  # Ensure all required columns are present
  required_cols <- c("age", "gender", "height", "weight", "workoutType", "workoutDuration",
                     "heartRate", "stepsTaken", "distance", "workoutIntensity", "sleepHours",
                     "dailyCaloriesIntake", "restingHeartRate", "moodBeforeWorkout", "moodAfterWorkout")
  missing_cols <- setdiff(required_cols, names(df))
  if (length(missing_cols) > 0) {
    cat("\nMissing columns in input data:", paste(missing_cols, collapse = ", "), "\n", file = log_file, append = TRUE)
    stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")))
  }
  
  # Call the prediction function
  preds <- tryCatch({
    predict_calories_class(df, MI_model, tier_models, specialized_models)
  }, error = function(e) {
    cat("\nError in prediction function:\n", e$message, "\n", file = log_file, append = TRUE)
    stop("Prediction failed.")
  })
  
  # Return prediction with probabilities
  list(
    prediction = preds$class_predictions,
    probabilities = preds$class_probabilities
  )
}