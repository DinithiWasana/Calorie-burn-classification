attach(workout_fitness_tracker_data)
View(workout_fitness_tracker_data)


#############################################################################################
#                                     DATA PRE-PROCESSING                                   #
#############################################################################################

sum(duplicated(workout_fitness_tracker_data)) # No. of duplicates

# =============================== Dropping irrelevant columns ===============================
library(dplyr)

workout_fitness_tracker_data <- subset(workout_fitness_tracker_data, select = -c(`Water Intake (liters)`, `VO2 Max`, `Body Fat (%)`))

# ============================== Feature engineering: 'Age' =================================

# Unique ages in 'Age' column
unique(workout_fitness_tracker_data$Age)

# Define age groups as:
#    Young:       18–30
#    Middle-aged: 31–50
#    Senior:      51–59

workout_fitness_tracker_data$`Age Group` <- cut(
  workout_fitness_tracker_data$Age, 
  breaks = c(18, 30, 50, 59),                   # Define the breaks for groups
  labels = c("Young", "Middle-aged", "Senior"),
  include.lowest = TRUE                         # Include the lowest value (18) in the first group
)

# ==================================== Detecting outliers ====================================

# --------------------------------- Method: Isolation Forest ---------------------------------
library(isotree)

data_subset <- subset(workout_fitness_tracker_data, select = -c(Gender, `Workout Type`, `Workout Intensity`, `Mood Before Workout`, `Mood After Workout`, `Age Group`))
iso_forest <- isolation.forest(data_subset, ntrees = 100)

# Compute anomaly scores
anomaly_scores <- predict(iso_forest, data_subset, type = "score")

# Define threshold (top 5% most anomalous points)
threshold <- quantile(anomaly_scores, 0.95)

# Identify outliers (scores above the threshold)
outliers <- anomaly_scores > threshold

# Count total outliers
total_outliers <- sum(outliers)
percentage_outliers <- (total_outliers / nrow(workout_fitness_tracker_data)) * 100

# Print results
print(paste("Total number of outliers detected:", total_outliers))
print(paste("Percentage of outliers:", round(percentage_outliers, 2), "%"))

# Create two separate dataset with outliers 
outlier_dataset <- workout_fitness_tracker_data[outliers, ]
View(outlier_dataset)

write.csv(outlier_dataset, "outlier_dataset.csv", row.names = FALSE)

# ----------------------------- Domain-Specific Outlier Detection ----------------------------
attach(outlier_dataset)

# Part 1: Implausible Physiological Metrics

# Calculate BMI and flag extremes
outlier_dataset$BMI <- outlier_dataset$`Weight (kg)` / ((outlier_dataset$`Height (cm)` / 100)^2)

physio_outliers <- outlier_dataset %>%
  filter(
    BMI < 16 | BMI > 40 |                                  # Extreme under/overweight
      `Resting Heart Rate (bpm)` < 40 |                    # Elite athlete or error
      `Resting Heart Rate (bpm)` > 100 |                   # Unusually high resting HR
      `Heart Rate (bpm)` < 50 | `Heart Rate (bpm)` > 200   # Impossible exercise HR
  )

# Part 2: Impossible Activity Metrics
activity_outliers <- outlier_dataset %>%
  filter(
    `Steps Taken` / `Workout Duration (mins)` > 200 |                  # >200 steps/minute is impossible
      (`Workout Type` == "Yoga" & `Workout Duration (mins)` > 120) |   # Yoga sessions >2 hours
      (`Calories Burned` / `Workout Duration (mins)` > 20)             # >20 cal/min is extreme
  )

# Part 3: Contradictory Records
contradictions <- outlier_dataset %>%
  filter(
    (`Workout Intensity` == "Low" & `Calories Burned` > 500) |
      (`Workout Type` == "Cycling" & `Distance (km)` < 1 & `Steps Taken` > 5000)   # Cycling with steps
  )

# Combine and Classify Outliers
true_outliers <- unique(rbind(physio_outliers, activity_outliers, contradictions))
valid_extremes <- outlier_dataset[!outlier_dataset$`User ID` %in% true_outliers$`User ID`, ]  

cat("Total outliers flagged by Isolation Forest:", nrow(outlier_dataset), "\n")
cat("True outliers (errors/impossibilities):", nrow(true_outliers), "\n")
cat("Valid extreme values:", nrow(valid_extremes), "\n")

View(true_outliers)

# Create a dataset with valid extremes 

cleaned_workout_fitness_tracker_data <- workout_fitness_tracker_data[!workout_fitness_tracker_data$`User ID` %in% true_outliers$`User ID`, ] %>% select(-`User ID`)
View(cleaned_workout_fitness_tracker_data)

percentage_existing_outliers <- (nrow(valid_extremes) / nrow(cleaned_workout_fitness_tracker_data)) * 100
print(paste("Percentage of existing outliers:", round(percentage_existing_outliers, 2), "%"))

write.csv(cleaned_workout_fitness_tracker_data, "cleaned_workout_fitness_tracker_data.csv", row.names = FALSE)

# =================== Split the data set into 'training' and 'testing' ======================
attach(cleaned_workout_fitness_tracker_data)

set.seed(123)

# Part 1: Define the split ratio as 80% for training and 20% for testing
SplitRatio <- 0.8
split <- sample(1:nrow(cleaned_workout_fitness_tracker_data), size = round(SplitRatio*nrow(cleaned_workout_fitness_tracker_data)))

# Part 2: Create training and testing sets
train <- cleaned_workout_fitness_tracker_data[split, ]
test <- cleaned_workout_fitness_tracker_data[-split, ]

# Part 3: Check the dimensions of the split data sets
dim(train)
dim(test)

View(train)
View(test)

# Part 4: Ensuring a balanced distribution of calories burned in 'training' and 'testing'

# Create the density data
generate_density_data <- function(variable) {
  density_data <- density(variable, na.rm = T)
  data.frame(x = density_data$x, y = density_data$y)
}

train_density <- generate_density_data(train$`Calories Burned`)
test_density <- generate_density_data(test$`Calories Burned`)

library(plotly)
plot_ly() %>%
  add_trace(data = train_density, x = ~x, y = ~y, type = 'scatter', mode = 'lines', 
            name = 'Train Set', line = list(color = 'red', width = 2)) %>%
  add_trace(data = test_density, x = ~x, y = ~y, type = 'scatter', mode = 'lines', 
            name = 'Test Set', line = list(color = 'black', width = 2, dash = 'dash')) %>%
  layout(title = list(text = 'Density Plot of Calories Burned in Train and Test Sets', x = 0.5),
         xaxis = list(title = 'Calories Burned', titlefont = list(size = 14, face = 'bold'),
                      tickfont = list(size = 12, face = 'bold'), showgrid = TRUE, gridcolor = 'gray80'),
         yaxis = list(title = 'Density', titlefont = list(size = 14, face = 'bold'),
                      tickfont = list(size = 12, face = 'bold'), showgrid = TRUE, gridcolor = 'gray80'),
         legend = list(x = 0.95, y = 0.95),
         template = "plotly_white")

# =============================== Save pre-processed data ===================================

write.csv(train, "train.csv", row.names = FALSE)
write.csv(test, "test.csv", row.names = FALSE)


#############################################################################################
#                              EXPLORATORY DATA ANALYSIS (EDA)                              #
#############################################################################################
attach(train)

# ================================== Uni-variate Analysis ===================================


# ============================== Part 1: Qualitative Variables ==============================

# ------------------------------- Graphical method: Pie charts ------------------------------

# Define a function to create pie charts
create_pie_chart <- function(data, variable, title, color_mapping, text_color = "black") {
  counts <- data %>%
    count(!!sym(variable)) %>%
    mutate(percentage = round(n / sum(n) * 100, 1))
  
  # Filter color mapping to match existing categories
  val_filtered <- color_mapping[names(color_mapping) %in% counts[[variable]]]
  
  fig <- plot_ly(
    counts,
    labels = ~get(variable),
    values = ~n,
    type = 'pie',
    textinfo = 'label+percent',
    insidetextorientation = 'radial',
    marker = list(colors = val_filtered),
    textfont = list(size = 16, color = text_color, family = "Arial", weight = "bold")
  )
  
  fig <- fig %>%
    layout(
      title = list(text = title, x = 0.5),
      showlegend = F
    )
  
  fig
}

#1.1 Gender
gender_colors <- c("Male" = "red", "Female" = "orange", "Other" = "yellow")
create_pie_chart(train, "Gender", "Gender Categories", gender_colors, "black")

#1.2 Workout Intensity
workout_intensity_colors <- c("High" = "yellow", "Medium" = "orange", "Low" = "gold")
create_pie_chart(train, "Workout Intensity", "Workout Intensity Categories", workout_intensity_colors, "black")

#1.3 Mood Before Workout
mood_before_workout_colors <- c("Tired" = "red", "Happy" = "darkorange", "Neutral" = "gold", "Stressed" = "yellow")
create_pie_chart(train, "Mood Before Workout", "Mood Before Workout Categories", mood_before_workout_colors)

#1.4 Mood After Workout
mood_after_workout_colors <- c("Fatigued" = "darkorange", "Energized" = "gold", "Neutral" = "yellow")
create_pie_chart(train, "Mood After Workout", "Mood After Workout Categories", mood_after_workout_colors)

#1.5 Age Group
age_group_colors <- c("Young" = "red", "Middle-aged" = "orange", "Senior" = "gold")
create_pie_chart(train, "Age Group", "Age Group Distribution", age_group_colors)


# ------------------------------- Graphical method: Bar charts -------------------------------

#1.6 Workout Type
workout_type <- table(train$`Workout Type`)
workout_percentage <- (workout_type / sum(workout_type)) * 100  # Convert to %

# Convert the table to a data frame
workout_df <- as.data.frame(workout_percentage)
colnames(workout_df) <- c("Workout_Type", "Percentage")  # Rename columns

# Create the plot
plot_ly(
  data = workout_df,  # Use the data frame as the first argument
  x = ~Percentage,    # Use Percentage for the x-axis
  y = ~reorder(Workout_Type, Percentage),  # Reorder Workout_Type by Percentage
  type = 'bar', 
  orientation = 'h',
  text = ~paste0(round(Percentage, 1), '%'),  # Add percentage labels
  textposition = 'outside', 
  marker = list(color = colorRampPalette(c("red", "darkorange", "yellow"))(nrow(workout_df)))
) %>%
  layout(
    title = 'Workout Type Categories',
    xaxis = list(title = 'Percentage'),
    yaxis = list(title = 'Workout Type'),
    showlegend = FALSE
  )


# =============================== Part 2: Quantitative Variables ==============================

# -------------------------------- Graphical method: Histograms -------------------------------

# Define a function to create histograms
create_histogram <- function(data, column, title, xaxis_title, yaxis_title = "Count", color = "gold", opacity = 0.6, nbinsx = 50) {
  plot_ly(
    data = data,
    x = ~get(column),  # Use get() to dynamically access the column
    type = "histogram",
    nbinsx = nbinsx,
    marker = list(color = color, line = list(color = 'black', width = 1)),
    opacity = opacity
  ) %>%
    layout(
      title = title,
      xaxis = list(title = xaxis_title),
      yaxis = list(title = yaxis_title),
      template = "plotly_white"
    )
}

# Calories Burned 
create_histogram(data = train, column = "Calories Burned", title = "Histogram of Calories Burned", 
                 xaxis_title = "Calories Burned", color = "gold")


# -------------------------------- Graphical method: Box plots --------------------------------

# Define a function to create box plots
create_boxplot <- function(data, column, title, yaxis_title, fillcolor = "gold", opacity = 0.8) {
  plot_ly(
    data = data,
    y = ~get(column),  # Use get() to dynamically access the column
    type = 'box',
    boxpoints = 'outliers', 
    marker = list(color = 'black'), 
    fillcolor = fillcolor, 
    line = list(color = 'black'), 
    opacity = opacity
  ) %>% 
    layout(
      title = list(text = title, x = 0.5),
      yaxis = list(
        title = yaxis_title, 
        titlefont = list(size = 14, face = 'bold'),
        tickfont = list(size = 12, face = 'bold')
      ),
      xaxis = list(showticklabels = FALSE),
      template = "plotly_white"
    )
}

# Calories Burned 
create_boxplot(data = train, column = "Calories Burned", title = "Boxplot of Calories Burned", 
               yaxis_title = "Calories Burned")


# ------------------------------------ Summary Statistics -------------------------------------
library(knitr)

generate_summary_stats <- function(data, variable, caption) {
  summary_stats <- summary(data[[variable]])
  summary_table <- data.frame(
    Statistic = names(summary_stats),
    Value = as.numeric(summary_stats)
  )
  
  # Calculate additional statistics
  std_dev <- sd(data[[variable]], na.rm = TRUE)
  variance <- var(data[[variable]], na.rm = TRUE)
  skew <- e1071::skewness(data[[variable]], na.rm = TRUE)
  kur <- e1071::kurtosis(data[[variable]], na.rm = TRUE)
  additional_stats <- data.frame(
    Statistic = c("Std. Dev.", "Variance", "Skewness", "Kurtosis"),
    Value = c(std_dev, variance, skew, kur)
  )
  
  # Combine with the original summary table
  summary_table <- rbind(summary_table, additional_stats)
  kable(summary_table, caption = caption, align = 'c')
}

# Calories Burned 
generate_summary_stats(train, "Calories Burned", "Summary Statistics of Calories Burned")


# =================================== Bi-variate Analysis ====================================


# ========================== Part 1: Qualitative vs. Quantitative ============================

# -------------------------------- Graphical method: Box plots -------------------------------

# Define a function to create grouped box plots
generate_boxplot <- function(data, x_var, y_var, title, x_label, colors_range) {
  custom_colors <- colorRampPalette(colors_range)(length(unique(data[[x_var]])))
  
  plot_ly(data = data, 
          x = as.formula(paste("~factor(`", x_var, "`)", sep="")), 
          y = as.formula(paste("~`", y_var, "`", sep="")), 
          type = 'box', 
          color = as.formula(paste("~factor(`", x_var, "`)", sep="")), 
          colors = custom_colors,  # Apply custom color palette
          boxpoints = 'outliers',
          marker = list(size = 5),
          line = list(color = 'black', width = 1)) %>%  # Black borders for clarity
    layout(title = list(text = title, x = 0.5),
           xaxis = list(title = x_label, titlefont = list(size = 14, face = 'bold'),
                        tickfont = list(size = 12, face = 'bold')),
           yaxis = list(title = 'Burned calories', titlefont = list(size = 14, face = 'bold'),
                        tickfont = list(size = 12, face = 'bold')),
           showlegend = FALSE,
           template = "plotly_white")
}

#1.1 Gender vs. Calories Burned 
generate_boxplot(train, "Gender", "Calories Burned", "Effect of Gender on Calory Burned", "Gender", 
                 c("red", "orange", "yellow"))

#1.2 Age Group vs. Calories Burned
generate_boxplot(train, "Age Group", "Calories Burned", "Effect of Age Group on Calory Burned", "Age Group", 
                 c("red", "orange", "yellow"))


# ========================== Part 2: Quantitative vs. Quantitative ===========================

# ----------------------------- Graphical method: Scatter plots ------------------------------

plot_scatter_histogram <- function(data, x_var, y_var, x_label, color_scatter, color_hist) {
  
  # Create a temporary dataset with the log-transformed variable
  temp_data <- data %>%
    mutate(!!paste0(x_var, "_log") := log1p(.data[[x_var]]))
  
  # Correlation Calculation
  correlation <- cor(temp_data[[x_var]], temp_data[[y_var]], use = "complete.obs")
  correlation_log <- cor(temp_data[[paste0(x_var, "_log")]], temp_data[[y_var]], use = "complete.obs")
  
  # Scatter plot without Log-Transformed Variable
  scatter_plot_original <- plot_ly(temp_data, 
                                   x = ~get(x_var), 
                                   y = ~get(y_var), 
                                   type = 'scatter', 
                                   mode = 'markers', 
                                   marker = list(color = color_scatter, opacity = 0.5, size = 6)) %>%
    add_lines(x = ~get(x_var), 
              y = fitted(lm(get(y_var) ~ get(x_var), data = temp_data)), 
              line = list(color = 'black', width = 2), 
              name = 'Linear Fit') %>%
    layout(title = list(text = paste("Scatter Plot:", x_label, "vs", y_var, "(Corr =", 
                                     round(correlation, 2), ")"), x = 0.5),
           xaxis = list(title = x_label, titlefont = list(size = 14, face = 'bold'),
                        tickfont = list(size = 12, face = 'bold')),
           yaxis = list(title = y_var, titlefont = list(size = 14, face = 'bold'),
                        tickfont = list(size = 12, face = 'bold')),
           template = 'plotly_white')
  
  # Histograms in the same plot window (1 row, 2 columns)
  hist_original <- plot_ly(temp_data, 
                           x = ~get(x_var), 
                           type = 'histogram', 
                           nbinsx = 30, 
                           marker = list(color = color_hist, line = list(color = 'black', width = 1)), 
                           opacity = 0.7) %>%
    layout(title = list(text = paste("Histogram of", x_label), x = 0.5),
           xaxis = list(title = x_label, titlefont = list(size = 14, face = 'bold'),
                        tickfont = list(size = 12, face = 'bold')),
           yaxis = list(title = "Count", titlefont = list(size = 14, face = 'bold'),
                        tickfont = list(size = 12, face = 'bold')),
           template = 'plotly_white')
  
  hist_log <- plot_ly(temp_data, 
                      x = ~get(paste0(x_var, "_log")), 
                      type = 'histogram', 
                      nbinsx = 30, 
                      marker = list(color = 'red', line = list(color = 'black', width = 1)), 
                      opacity = 0.7) %>%
    layout(title = list(text = paste("Histogram of Log-Transformed", x_label), x = 0.5),
           xaxis = list(title = paste("Log(", x_label, ")", sep = ""), 
                        titlefont = list(size = 14, face = 'bold'),
                        tickfont = list(size = 12, face = 'bold')),
           yaxis = list(title = "Count", titlefont = list(size = 14, face = 'bold'),
                        tickfont = list(size = 12, face = 'bold')),
           template = 'plotly_white')
  
  combined_hist <- subplot(hist_original, hist_log, nrows = 1, shareY = TRUE, titleX = TRUE, titleY = TRUE) %>%
    layout(title = list(text = paste("Histograms of", x_label, "(Original and Log-Transformed)"), x = 0.5))
  
  # Scatter plot with Log-Transformed Variable
  scatter_plot_log <- plot_ly(temp_data, 
                              x = ~get(paste0(x_var, "_log")), 
                              y = ~get(y_var), 
                              type = 'scatter', 
                              mode = 'markers', 
                              marker = list(color = color_scatter, opacity = 0.5, size = 6)) %>%
    add_lines(x = ~get(paste0(x_var, "_log")), 
              y = fitted(lm(get(y_var) ~ get(paste0(x_var, "_log")), data = temp_data)), 
              line = list(color = 'black', width = 2), 
              name = 'Linear Fit') %>%
    layout(title = list(text = paste("Scatter Plot (Log):", x_label, "vs", y_var, 
                                     "(Corr =", round(correlation_log, 2), ")"), x = 0.5),
           xaxis = list(title = paste("Log(", x_label, ")", sep = ""), titlefont = list(size = 14, face = 'bold'),
                        tickfont = list(size = 12, face = 'bold')),
           yaxis = list(title = y_var, titlefont = list(size = 14, face = 'bold'),
                        tickfont = list(size = 12, face = 'bold')),
           template = 'plotly_white')
  
  list(scatter_plot_original = scatter_plot_original, combined_hist = combined_hist, 
       scatter_plot_log = scatter_plot_log)
}

#2.1 Calories Burned vs. Height
plots_height <- plot_scatter_histogram(train, "Height (cm)", "Calories Burned", "Height (cm)", "blue", "orange")
plots_height$scatter_plot_original
plots_height$combined_hist

#2.2 Calories Burned vs. Weight
plots_weight <- plot_scatter_histogram(train, "Weight (kg)", "Calories Burned", "Weight (kg)", "blue", "orange")
plots_weight$scatter_plot_original
plots_weight$combined_hist


# ================================== Multivariate Analysis ===================================

# =================================== Tri-variate Analysis ===================================

# 'Calories Burned' by 'Gender' within 'Age Group'
plot_ly(train, x = ~`Age Group`, y = ~`Calories Burned`, color = ~`Gender`,
        colors = c("red", "orange", "yellow"), # Custom color mapping
        type = "box",
        line = list(color = "black", width = 1), # Black border with width 1
        boxpoints = FALSE # Hide individual points for cleaner look
) %>%
  layout(
    title = "Calories Burned by Gender Across Age Groups",
    xaxis = list(title = "Age Group"),
    yaxis = list(title = "Calories Burned"),
    boxmode = "group",
    boxgap = 0.3,
    boxgroupgap = 0.5,
    legend = list(
      title = list(text = "<b>Gender</b>"), # Bold legend title
      x = 0.95, 
      y = 0.9
    )
  )


# ======================================= Two-Way ANOVA ======================================

# ---------------------- Method (non-parametric): Scheirer-Ray-Hare Test  --------------------

train_clean <- train %>% rename(CaloriesBurned = `Calories Burned`, AgeGroup = `Age Group`)

scheirerRayHare(CaloriesBurned ~ Gender + AgeGroup + Gender:AgeGroup, data = train_clean)


# ========================== Part 2: Quantitative vs. Quantitative ===========================

# --------------------------- Graphical method: Correlation heatmap --------------------------

# Part 1: Only Calories Burned and numeric predictors
correlation_data <- train %>% 
  select(`Calories Burned`, 
         Age, `Height (cm)`, `Workout Duration (mins)`,
         `Heart Rate (bpm)`, `Steps Taken`, `Distance (km)`, 
         `Sleep Hours`, `Daily Calories Intake`, 
         `Resting Heart Rate (bpm)`)

cor_matrix <- cor(correlation_data, use = "complete.obs") %>%
  as.data.frame() %>%
  select(`Calories Burned`) %>%    # Keep only the column for Calories Burned
  t()                              # Transpose to get a single row

plot_ly(x = colnames(cor_matrix), 
        y = "Calories Burned",     # Single row label
        z = cor_matrix, 
        type = "heatmap",
        colors = colorRampPalette(c("blue", "lightblue", "white", "red"))(100),
        text = round(cor_matrix, 2),
        texttemplate = "%{text}",
        hoverinfo = "x+z+text",
        colorbar = list(title = "Correlation")) %>%
  layout(
    title = list(text = "Correlations with Calories Burned", x = 0.5),
    xaxis = list(title = "Quantitative Predictors", tickangle = -45, 
                 tickfont = list(size = 12)),
    yaxis = list(title = "Response Variable", tickfont = list(size = 14)),
    margin = list(l = 100, b = 150),
    hoverlabel = list(bgcolor = "white")
  )

# Part 2: For all numeric variables
correlation_data <- train %>% select(Age, `Height (cm)`, `Workout Duration (mins)`, `Calories Burned`,
                                     `Heart Rate (bpm)`, `Steps Taken`, `Distance (km)`, `Sleep Hours`, 
                                     `Daily Calories Intake`, `Resting Heart Rate (bpm)`)
cor_matrix <- cor(correlation_data, use = "complete.obs")

plot_ly(x = colnames(cor_matrix), 
        y = rownames(cor_matrix), 
        z = cor_matrix, 
        type = "heatmap", 
        text = round(cor_matrix, 2),
        texttemplate = "%{text}",
        colors = colorRamp(c("blue", "lightblue", "red")),
        colorbar = list(title = "Correlation")) %>%
  layout(title = list(text = "Correlation Heatmap", x = 0.5),
         xaxis = list(title = "", tickfont = list(size = 12, face = 'bold')),
         yaxis = list(title = "", tickfont = list(size = 12, face = 'bold')),
         template = 'plotly_white')


# =============================== Part 3: Ordinal vs. Nominal ================================

# ---------------------- Graphical method: Kruskal-Wallis P-value Heatmap --------------------
library(rstatix)

categorical_vars <- c("Gender", "`Age Group`", "`Workout Type`", 
                      "`Workout Intensity`", "`Mood Before Workout`", 
                      "`Mood After Workout`")

# Calculate Kruskal-Wallis tests
results <- lapply(categorical_vars, function(var) {
  test <- kruskal.test(reformulate(var, response = "`Calories Burned`"), 
                       data = train)
  data.frame(
    Predictor = var,
    p_value = test$p.value,
    formatted_p = ifelse(test$p.value < 0.001, 
                         "<0.001", 
                         sprintf("%.3f", test$p.value))
  )
}) %>% bind_rows()

# Create annotation text (position at center of each cell)
annotations <- list()
for(i in 1:nrow(results)) {
  annotations[[i]] <- list(
    x = results$Predictor[i],
    y = "Calories Burned",
    text = results$formatted_p[i],
    xref = "x",
    yref = "y",
    showarrow = FALSE,
    font = list(color = ifelse(results$p_value[i] < 0.14, "black", "white"),
                size = 12)
  )
}

# Create the heatmap with annotations
plot_ly(
  x = results$Predictor,
  y = "Calories Burned",
  z = matrix(-log10(results$p_value), nrow = 1),
  type = "heatmap",
  colorscale = "Viridis",
  colorbar = list(title = "-log10(p-value)"),
  hoverinfo = "text",
  text = matrix(paste0("<b>", results$Predictor, "</b><br>",
                       "p-value: ", results$formatted_p), 
                nrow = 1)
) %>%
  layout(
    title = "Kruskal-Wallis Test: Association with Calories Burned",
    xaxis = list(title = "Categorical Predictors"),
    yaxis = list(title = "Response Variable"),
    annotations = annotations,
    margin = list(l = 100, b = 100),
    hoverlabel = list(bgcolor = "white")
  )

#############################################################################################
#                                         CLUSTERING                                        #
#############################################################################################

# --------------------------------- Part 1: Prepare the Data --------------------------------
set.seed(123)

# Recombine train and test while preserving split information
train$data_split <- "train"
test$data_split <- "test"
full_dataset <- rbind(train, test)

# Remove data_split from variables used for clustering
clustering_vars <- setdiff(names(full_dataset), c("data_split", "Age Group"))
full_dataset_for_clustering <- full_dataset[, clustering_vars]

# Convert categorical variables to factors
categorical_vars <- c("Gender", "Workout Type", "Workout Intensity", 
                      "Mood Before Workout", "Mood After Workout")
full_dataset_for_clustering[categorical_vars] <- lapply(full_dataset_for_clustering[categorical_vars], as.factor)

# ------------------------- Part 2: Distance Matrix for Mixed Data -------------------------
library(cluster)

gower_dist <- daisy(full_dataset_for_clustering, metric = "gower")

# ------------------------ Part 3: Determine Optimal Cluster Count -------------------------
library(factoextra)
library(clValid)
library(plotly)

# Custom PAM wrapper for gap statistic
pam_wrapper <- function(x, k) {
  list(cluster = pam(gower_dist, k = k, diss = TRUE)$clustering)
}

# Silhouette method (Plotly version)
silhouette_analysis <- function(k) {
  pam_result <- pam(gower_dist, k = k, diss = TRUE)
  sil <- silhouette(pam_result$clustering, gower_dist)
  return(mean(sil[, "sil_width"]))
}

k_values <- 2:10
sil_scores <- sapply(k_values, silhouette_analysis)

sil_plot <- plot_ly(x = k_values, y = sil_scores, type = 'scatter', mode = 'lines+markers') %>%
  layout(title = "Silhouette Method",
         xaxis = list(title = "Number of clusters"),
         yaxis = list(title = "Average Silhouette Width"))

# Elbow method (Plotly version)
wss <- sapply(k_values, function(k){pam(gower_dist, k, diss = TRUE)$objective[1]})
wss_plot <- plot_ly(x = k_values, y = wss, type = 'scatter', mode = 'lines+markers') %>%
  layout(title = "Scree Plot",
         xaxis = list(title = "Number of clusters"),
         yaxis = list(title = "Total Within Sum of Squares"))

# -------------------------- Part 4: Apply Clustering Algorithms ---------------------------
optimal_k <- 3 

# K-medoids (PAM) clustering
pam_clusters <- pam(gower_dist, k = optimal_k, diss = TRUE)

# Add cluster assignments
full_dataset$pam_cluster <- pam_clusters$clustering

# -------------------------- Part 5: Evaluate Cluster Quality ------------------------------

# Silhouette plot 
sil <- silhouette(pam_clusters$clustering, gower_dist)
sil_data <- data.frame(cluster = sil[, "cluster"],
                       neighbor = sil[, "neighbor"],
                       sil_width = sil[, "sil_width"])

sil_plot <- plot_ly(sil_data, x = ~sil_width, y = ~factor(cluster), 
                    color = ~factor(cluster), type = 'box', line = list(color = "black", width = 1)) %>%
  layout(title = paste("Silhouette Plot (Avg Width =", round(mean(sil[, "sil_width"]), 3), ")"),
         xaxis = list(title = "Silhouette Width"),
         yaxis = list(title = "Cluster"))

# Dunn index
dunn_index <- dunn(gower_dist, pam_clusters$clustering)
print(paste("Dunn Index:", round(dunn_index, 3)))

# ----------------------------- Part 6: Visualize Clusters ---------------------------------
# t-SNE visualization
library(Rtsne)

tsne_result <- Rtsne(gower_dist, is_distance = TRUE, perplexity = 30)

plot_data <- data.frame(
  tsne1 = tsne_result$Y[,1],
  tsne2 = tsne_result$Y[,2],
  Cluster = as.factor(pam_clusters$clustering),
  DataSplit = full_dataset$data_split
)

# Cluster visualization
cluster_plot <- plot_ly(plot_data, x = ~tsne1, y = ~tsne2, 
                        color = ~Cluster, colors = "Set1",
                        type = 'scatter', mode = 'markers') %>%
  layout(title = "t-SNE Cluster Visualization")

# Cluster + split visualization
split_plot <- plot_ly(plot_data, x = ~tsne1, y = ~tsne2, 
                      color = ~Cluster, colors = "Set1",
                      symbol = ~DataSplit, symbols = c('circle', 'x'),
                      type = 'scatter', mode = 'markers') %>%
  layout(title = "Clusters with Train/Test Split")



#############################################################################################
#                                     ADVANCED ANALYSIS                                     #
#############################################################################################

#############################################################################################
#                        ENHANCED ENSEMBLE SPECIALIZATION APPROACH                          #
#############################################################################################
# Load required libraries
library(dplyr)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(gbm)
library(e1071)
library(nnet)
library(pROC)
library(MLmetrics)
library(glmnet)
library(xgboost)
library(Matrix)
library(parallel)
library(doParallel)
library(ranger)
library(tibble)
library(ggplot2)
library(plotly)
library(purrr)

# Set up parallel processing
cores <- detectCores() - 1
cl <- makeCluster(cores)
registerDoParallel(cl)

# Load data (assuming train and test datasets are already loaded)
# attach(train)
# attach(test)

# Remove Age Group column if it exists
train <- train %>% select(-`Age Group`, -contains("Age Group"))
test <- test %>% select(-`Age Group`, -contains("Age Group"))

# ---------------------- Step 1: Create a model to predict Metabolic.Intensity --------------------
# Calculate the actual Metabolic.Intensity for training the predictor
train_with_MI <- train %>%
  mutate(
    BMI = `Weight (kg)` / ((`Height (cm)` / 100)^2),
    Heart.Rate.Ratio = `Heart Rate (bpm)` / `Resting Heart Rate (bpm)`,
    Cardio.Stress = (`Heart Rate (bpm)` - `Resting Heart Rate (bpm)`) / `Resting Heart Rate (bpm)`,
    Metabolic.Intensity = `Calories Burned` / (`Workout Duration (mins)` * `Weight (kg)`)
  )

# Create formula for Metabolic.Intensity prediction model

# Make names syntactically valid
names(train_with_MI) <- make.names(names(train_with_MI))

# Use only features that would be available at prediction time
MI_predictor_vars <- c(
  "Age", "Gender", "Height..cm.", "Weight..kg.",
  "Workout.Type", "Workout.Duration..mins.",
  "Heart.Rate..bpm.", "Steps.Taken", "Distance..km.",
  "Workout.Intensity", "Sleep.Hours",
  "Daily.Calories.Intake", "Resting.Heart.Rate..bpm.",
  "Mood.Before.Workout", "Mood.After.Workout",
  "BMI", "Heart.Rate.Ratio", "Cardio.Stress"
)

# Create formula
MI_formula <- as.formula(paste("Metabolic.Intensity ~", paste(MI_predictor_vars, collapse = " + ")))

# Set up training control for Metabolic.Intensity model
MI_train_control <- trainControl(
  method = "cv", 
  number = 5,
  verboseIter = TRUE
)

# Train the Metabolic.Intensity prediction model
# Using random forest as it typically performs well for this type of prediction
set.seed(123)
MI_model <- train(
  MI_formula,
  data = train_with_MI,
  method = "ranger",
  trControl = MI_train_control,
  importance = "impurity"
)

# Print model performance
print(MI_model)

# ------------------ Step 2: Use the predicted Metabolic.Intensity in main models -----------------
# Make names syntactically valid
names(train) <- make.names(names(train))
names(test) <- make.names(names(test))

# Create feature sets without the original Metabolic.Intensity (which has data leakage)
train <- train %>%
  mutate(
    BMI = `Weight..kg.` / ((`Height..cm.` / 100)^2),
    Heart.Rate.Ratio = `Heart.Rate..bpm.` / `Resting.Heart.Rate..bpm.`,
    Cardio.Stress = (`Heart.Rate..bpm.` - `Resting.Heart.Rate..bpm.`) / `Resting.Heart.Rate..bpm.`
    # No original Metabolic.Intensity here - we'll use the predicted version
  )

test <- test %>%
  mutate(
    BMI = `Weight..kg.` / ((`Height..cm.` / 100)^2),
    Heart.Rate.Ratio = `Heart.Rate..bpm.` / `Resting.Heart.Rate..bpm.`,
    Cardio.Stress = (`Heart.Rate..bpm.` - `Resting.Heart.Rate..bpm.`) / `Resting.Heart.Rate..bpm.`
    # No original Metabolic.Intensity here - we'll use the predicted version
  )


# Predict Metabolic.Intensity for both train and test sets
train$Predicted.Metabolic.Intensity <- predict(MI_model, train)
test$Predicted.Metabolic.Intensity <- predict(MI_model, test)


# Update predictor variables to use predicted Metabolic.Intensity
predictor_vars <- c(
  "Age", "Gender", "Height..cm.", "Weight..kg.",
  "Workout.Type", "Workout.Duration..mins.",
  "Heart.Rate..bpm.", "Steps.Taken", "Distance..km.",
  "Workout.Intensity", "Sleep.Hours",
  "Daily.Calories.Intake", "Resting.Heart.Rate..bpm.",
  "Mood.Before.Workout", "Mood.After.Workout",
  "BMI", "Heart.Rate.Ratio", "Cardio.Stress",
  "Predicted.Metabolic.Intensity"  # Use the predicted value instead
)

# Handle NAs in newly created features
train <- train %>%
  mutate(across(c(BMI, Heart.Rate.Ratio, Predicted.Metabolic.Intensity, Cardio.Stress,
                  ~ifelse(is.infinite(.) | is.na(.), median(., na.rm = TRUE), .))))

test <- test %>%
  mutate(across(c(BMI, Heart.Rate.Ratio, Predicted.Metabolic.Intensity, Cardio.Stress),
                ~ifelse(is.infinite(.) | is.na(.), median(., na.rm = TRUE), .)))

# Check for any remaining NA values and impute if necessary
for(col in names(train)) {
  if(is.numeric(train[[col]])) {
    med_val <- median(train[[col]], na.rm = TRUE)
    train[[col]][is.na(train[[col]])] <- med_val
    test[[col]][is.na(test[[col]])] <- med_val
  } else if(is.factor(train[[col]]) || is.character(train[[col]])) {
    mode_val <- names(sort(table(train[[col]]), decreasing = TRUE))[1]
    train[[col]][is.na(train[[col]])] <- mode_val
    test[[col]][is.na(test[[col]])] <- mode_val
  }
}

# ---------------------------- Step 3: Specialized Binning Approach -------------------------------
# First, save the original continuous target variable
train_calories_original <- train$`Calories.Burned`
test_calories_original <- test$`Calories.Burned`

# Get quantiles for reference
cal_quantiles <- quantile(train$`Calories.Burned`, probs = seq(0, 1, 0.25))
print("Calories Burned Quantiles:")
print(cal_quantiles)

# Create binned target variable using quantile-based binning
quantile_binning <- function(x) {
  q <- quantile(train$`Calories.Burned`, probs = c(0.25, 0.5, 0.75))
  cut(x, 
      breaks = c(-Inf, q[1], q[2], q[3], Inf),
      labels = c("Low", "Medium.Low", "Medium.High", "High"),
      right = TRUE)
}

train$Calories.Class <- quantile_binning(train$`Calories.Burned`)
test$Calories.Class <- quantile_binning(test$`Calories.Burned`)

# Also create tier indicators for specialized models
train$is_low_tier <- train$`Calories.Burned` <= cal_quantiles[2]
train$is_mid_tier <- train$`Calories.Burned` > cal_quantiles[2] & train$`Calories.Burned` <= cal_quantiles[4]
train$is_high_tier <- train$`Calories.Burned` > cal_quantiles[4]

test$is_low_tier <- test$`Calories.Burned` <= cal_quantiles[2]
test$is_mid_tier <- test$`Calories.Burned` > cal_quantiles[2] & test$`Calories.Burned` <= cal_quantiles[4]
test$is_high_tier <- test$`Calories.Burned` > cal_quantiles[4]

# Convert categorical variables to factors
categorical_vars <- c("Gender", "Workout.Type", "Workout.Intensity", 
                      "Mood.Before.Workout", "Mood.After.Workout")

train <- train %>%
  mutate(across(all_of(categorical_vars), as.factor))

test <- test %>%
  mutate(across(all_of(categorical_vars), as.factor))

# Ensure Calories.Class is a factor
train$Calories.Class <- as.factor(train$Calories.Class)
test$Calories.Class <- as.factor(test$Calories.Class)

# Check if all predictor variables exist in the dataset
missing_predictors <- setdiff(predictor_vars, names(train))
if(length(missing_predictors) > 0) {
  warning(paste("The following predictors are missing:", paste(missing_predictors, collapse=", ")))
  predictor_vars <- intersect(predictor_vars, names(train))
}

# --------------------------- Step 4: Create Specialized Training Sets ---------------------------
# Create specialized datasets
train_low <- train %>% filter(is_low_tier == TRUE)
train_mid <- train %>% filter(is_mid_tier == TRUE)
train_high <- train %>% filter(is_high_tier == TRUE)

# Print dataset sizes
cat("Low tier dataset size:", nrow(train_low), "\n")
cat("Mid tier dataset size:", nrow(train_mid), "\n")
cat("High tier dataset size:", nrow(train_high), "\n")

# --------------------------- Step 5: Set up Cross-Validation -------------------------------
# Set up stratified cross-validation
set.seed(123)
train_control <- trainControl(
  method = "cv", 
  number = 5,
  classProbs = TRUE,
  summaryFunction = multiClassSummary,
  sampling = "up",  # Use upsampling for imbalanced classes
  savePredictions = "final",
  verboseIter = TRUE
)

# --------------------------- Step 6: Define Model Hyperparameter Grids -------------------------------
# Define hyperparameter grids for each model type
rf_grid <- expand.grid(
  mtry = c(floor(sqrt(length(predictor_vars))), floor(length(predictor_vars)/3)),
  splitrule = "gini",      # Drop "extratrees" to simplify
  min.node.size = c(5, 10) # Increase from 1,5,10
)

xgb_grid <- expand.grid(
  nrounds = c(100, 200),
  max_depth = c(3, 4, 5),  # Reduce from 6,9 to limit complexity
  eta = c(0.01, 0.1),
  gamma = c(0.1, 0.5),     # Increase to enforce pruning
  colsample_bytree = c(0.5, 0.7),
  min_child_weight = c(3, 5),  # Increase to prevent overfitting
  subsample = c(0.7, 0.8)      # Reduce from 1.0
)

svm_grid <- expand.grid(
  sigma = c(0.01, 0.1),  # Simpler kernels
  C = c(1, 5)            # Higher C for less regularization
)

dt_grid <- expand.grid(
  cp = c(0.01, 0.05)  # Increase from 0.001,0.01,0.1 to prune more aggressively
)

multinom_grid <- expand.grid(decay = c(0, 0.001, 0.0001))

# --------------------------- Step 7: Create Tier Classifier -------------------------------

# Convert logical variables to factors for caret
train <- train %>%
  mutate(
    is_low_tier = factor(is_low_tier, levels = c(TRUE, FALSE), labels = c("Yes", "No")),
    is_mid_tier = factor(is_mid_tier, levels = c(TRUE, FALSE), labels = c("Yes", "No")),
    is_high_tier = factor(is_high_tier, levels = c(TRUE, FALSE), labels = c("Yes", "No"))
  )

test <- test %>%
  mutate(
    is_low_tier = factor(is_low_tier, levels = c(TRUE, FALSE), labels = c("Yes", "No")),
    is_mid_tier = factor(is_mid_tier, levels = c(TRUE, FALSE), labels = c("Yes", "No")),
    is_high_tier = factor(is_high_tier, levels = c(TRUE, FALSE), labels = c("Yes", "No"))
  )

# This model will decide which specialized model to use
tier_formula <- as.formula(paste("is_low_tier ~", paste(predictor_vars, collapse = " + ")))

tier_low_model <- train(
  tier_formula,
  data = train,
  method = "rf",
  trControl = train_control,
  #tuneGrid = rf_grid,
  metric = "Accuracy",
  importance = TRUE
)

tier_high_formula <- as.formula(paste("is_high_tier ~", paste(predictor_vars, collapse = " + ")))

tier_high_model <- train(
  tier_high_formula,
  data = train,
  method = "rf",
  trControl = train_control,
  #tuneGrid = rf_grid,
  metric = "Accuracy",
  importance = TRUE
)

# --------------------------- Step 8: Create Specialized Models ---------------------------
# Define formula for all models
class_formula <- as.formula(paste("Calories.Class ~", paste(predictor_vars, collapse = " + ")))

# Function to return predictions for constant model
predict.constant_model <- function(object, newdata, type = "raw", ...) {
  if(type == "raw") {
    # Return a factor of the single class
    return(factor(rep(object$single_class, nrow(newdata)), levels = object$all_classes))
  } else if(type == "prob") {
    # Create a probability matrix with 1 for the single class, 0 for others
    probs <- matrix(0, nrow = nrow(newdata), ncol = length(object$all_classes))
    colnames(probs) <- object$all_classes
    probs[, object$single_class] <- 1
    return(as.data.frame(probs))
  }
}

# Train specialized models with different algorithms
train_specialized_model <- function(train_data, method_name, formula, control, tune_grid = NULL) {
  cat("\nTraining", method_name, "model on specialized data with", nrow(train_data), "records...\n")
  
  # Skip if there are too few samples
  if(nrow(train_data) < 30) {
    warning("Too few samples to train a reliable model for this tier")
    return(NULL)
  }
  
  # Check class distribution
  class_counts <- table(train_data$Calories.Class)
  cat("Class distribution:", paste(names(class_counts), "=", class_counts, collapse=", "), "\n")
  
  # If only one class is present, create a dummy model
  if(length(class_counts) == 1 || sum(class_counts > 0) == 1) {
    cat("Only one class present. Creating a constant model for this tier.\n")
    
    # Get the single class
    single_class <- names(class_counts)[which.max(class_counts)]
    
    # Create a simple dummy model
    dummy_model <- list(
      single_class = single_class,
      all_classes = levels(train_data$Calories.Class),
      call = match.call(),
      type = "dummy_constant_model"
    )
    
    class(dummy_model) <- "constant_model"
    return(dummy_model)
  }
  
  # If some classes have zero samples, drop those levels
  if(any(class_counts == 0)) {
    cat("Some classes have zero samples. Creating a model with available classes only.\n")
    
    # Create a copy of the data with dropped levels
    train_data_subset <- train_data
    train_data_subset$Calories.Class <- droplevels(train_data_subset$Calories.Class)
    
    # Train with the subset data - FIX: Using explicit arguments instead of do.call
    set.seed(123)
    
    if(method_name == "ranger") {
      model <- train(
        formula,
        data = train_data_subset,
        method = method_name,
        trControl = control,
        tuneGrid = tune_grid,
        metric = "Accuracy",
        importance = "impurity"
      )
    } else if(method_name == "xgbTree") {
      model <- train(
        formula,
        data = train_data_subset,
        method = method_name,
        trControl = control,
        tuneGrid = tune_grid,
        metric = "Accuracy"
      )
    } else if(method_name == "svmRadial") {
      model <- train(
        formula,
        data = train_data_subset,
        method = method_name,
        trControl = control,
        tuneGrid = tune_grid,
        metric = "Accuracy",
        probability = TRUE
      )
    } else if(method_name == "rpart") {
      model <- train(
        formula,
        data = train_data_subset,
        method = method_name,
        trControl = control,
        tuneGrid = tune_grid,
        metric = "Accuracy"
      )
    } else if(method_name == "multinom") {
      model <- train(
        formula,
        data = train_data_subset,
        method = method_name,
        trControl = control,
        tuneGrid = tune_grid,
        metric = "Accuracy"
      )
    } else {
      stop(paste("Unsupported model type:", method_name))
    }
    
    # Store the available classes
    attr(model, "available_classes") <- levels(train_data_subset$Calories.Class)
    attr(model, "all_classes") <- levels(train_data$Calories.Class)
    attr(model, "type") <- "subset_model"
    
  } else {
    # Original approach if all classes are present
    set.seed(123)
    
    if(method_name == "ranger") {
      model <- train(
        formula,
        data = train_data,
        method = method_name,
        trControl = control,
        tuneGrid = tune_grid,
        metric = "Accuracy",
        importance = "impurity"
      )
    } else if(method_name == "xgbTree") {
      model <- train(
        formula,
        data = train_data,
        method = method_name,
        trControl = control,
        tuneGrid = tune_grid,
        metric = "Accuracy"
      )
    } else if(method_name == "svmRadial") {
      model <- train(
        formula,
        data = train_data,
        method = method_name,
        trControl = control,
        tuneGrid = tune_grid,
        metric = "Accuracy",
        probability = TRUE
      )
    } else if(method_name == "rpart") {
      model <- train(
        formula,
        data = train_data,
        method = method_name,
        trControl = control,
        tuneGrid = tune_grid,
        metric = "Accuracy"
      )
    } else if(method_name == "multinom") {
      model <- train(
        formula,
        data = train_data,
        method = method_name,
        trControl = control,
        tuneGrid = tune_grid,
        metric = "Accuracy"
      )
    } else {
      stop(paste("Unsupported model type:", method_name))
    }
    
    attr(model, "type") <- "standard_model"
  }
  
  return(model)
}

# Define model types to train
model_types <- list(
  rf = list(name = "ranger", grid = rf_grid),
  xgb = list(name = "xgbTree", grid = xgb_grid),
  svm = list(name = "svmRadial", grid = svm_grid),
  dt = list(name = "rpart", grid = dt_grid),
  multinom = list(name = "multinom", grid = multinom_grid)
)

# Train all model types for each tier
specialized_models <- list()

cat("\n=== Training Specialized Models for All Algorithms ===\n")
for(model_key in names(model_types)) {
  model_info <- model_types[[model_key]]
  cat("\n\n---------- Training", model_info$name, "models ----------\n")
  
  # Train models for each tier
  low_model <- train_specialized_model(train_low, model_info$name, class_formula, train_control, model_info$grid)
  mid_model <- train_specialized_model(train_mid, model_info$name, class_formula, train_control, model_info$grid)
  high_model <- train_specialized_model(train_high, model_info$name, class_formula, train_control, model_info$grid)
  
  # Store models
  specialized_models[[model_key]] <- list(
    low = low_model,
    mid = mid_model,
    high = high_model
  )
}

# --------------------------- Step 9: Define prediction and evaluation functions ---------------------------
# Enhanced function to get ensemble predictions
get_ensemble_predictions <- function(tier_models, tier_specialized_models, new_data) {
  # Get tier predictions
  is_low_pred <- predict(tier_models$low, new_data, type = "prob")[,"Yes"]
  is_high_pred <- predict(tier_models$high, new_data, type = "prob")[,"Yes"]
  
  # Determine most likely tier
  tier_probs <- data.frame(
    low = is_low_pred,
    mid = 1 - is_low_pred - is_high_pred, 
    high = is_high_pred
  )
  
  # Ensure no negative probabilities for mid tier
  tier_probs$mid <- pmax(0, tier_probs$mid)
  
  # Normalize to ensure probabilities sum to 1
  row_sums <- rowSums(tier_probs)
  tier_probs <- tier_probs / row_sums
  
  # Get primary tier for each observation
  primary_tiers <- apply(tier_probs, 1, which.max)
  primary_tiers <- factor(c("low", "mid", "high")[primary_tiers], levels = c("low", "mid", "high"))
  
  # Get the class levels
  all_classes <- levels(new_data$Calories.Class)
  
  # Function to safely get predictions
  get_safe_predictions <- function(model, new_data, all_classes) {
    if(is.null(model)) {
      # Return uniform distribution if model is NULL
      result <- matrix(1/length(all_classes), nrow = nrow(new_data), ncol = length(all_classes))
      colnames(result) <- all_classes
      return(as.data.frame(result))
    }
    
    # For constant_model class (single-class tier)
    if(inherits(model, "constant_model")) {
      return(predict(model, new_data, type = "prob"))
    }
    
    # For subset models (with only some classes)
    if(attr(model, "type") == "subset_model") {
      # Get available classes
      avail_classes <- attr(model, "available_classes")
      
      # Get predictions for available classes
      raw_preds <- predict(model, new_data, type = "prob")
      
      # Create full prediction matrix with zeros
      full_pred <- matrix(0, nrow = nrow(new_data), ncol = length(all_classes))
      colnames(full_pred) <- all_classes
      
      # Fill in available predictions
      for(cls in avail_classes) {
        full_pred[, cls] <- raw_preds[, cls]
      }
      
      # Normalize to ensure each row sums to 1
      row_sums <- rowSums(full_pred)
      idx_zero <- row_sums == 0
      if(any(idx_zero)) {
        # If any row sums to 0, assign uniform probabilities
        full_pred[idx_zero, ] <- 1/length(all_classes)
      }
      row_sums <- ifelse(row_sums == 0, 1, row_sums)
      full_pred <- full_pred / row_sums
      
      return(as.data.frame(full_pred))
    }
    
    # Standard model case
    return(predict(model, new_data, type = "prob"))
  }
  
  # Get predictions from each specialized model in each tier
  specialized_preds <- list()
  for(model_key in names(tier_specialized_models)) {
    models <- tier_specialized_models[[model_key]]
    
    # Get predictions from each tier model
    low_preds <- get_safe_predictions(models$low, new_data, all_classes)
    mid_preds <- get_safe_predictions(models$mid, new_data, all_classes)
    high_preds <- get_safe_predictions(models$high, new_data, all_classes)
    
    # Create weighted predictions
    weighted_preds <- data.frame(matrix(0, nrow = nrow(new_data), ncol = length(all_classes)))
    colnames(weighted_preds) <- all_classes
    
    for(i in 1:nrow(new_data)) {
      weighted_preds[i,] <- tier_probs$low[i] * as.numeric(low_preds[i,]) +
        tier_probs$mid[i] * as.numeric(mid_preds[i,]) +
        tier_probs$high[i] * as.numeric(high_preds[i,])
    }
    
    # Get final class predictions
    final_pred_class <- factor(colnames(weighted_preds)[apply(weighted_preds, 1, which.max)],
                               levels = all_classes)
    
    # Store results for this model type
    specialized_preds[[model_key]] <- list(
      tier_probs = tier_probs,
      primary_tier = primary_tiers,
      specialized_preds = list(low = low_preds, mid = mid_preds, high = high_preds),
      weighted_preds = weighted_preds,
      final_pred = final_pred_class
    )
  }
  
  return(specialized_preds)
}

# Function to calculate detailed performance metrics
calculate_metrics <- function(cm, probs, actual) {
  # Get class frequencies
  class_freq <- table(actual) / length(actual)
  
  # Calculate ROC objects and AUC for each class
  roc_objects <- list()
  auc_values <- numeric(length(levels(actual)))
  names(auc_values) <- levels(actual)
  
  for (i in 1:length(levels(actual))) {
    class_label <- levels(actual)[i]
    roc_objects[[class_label]] <- tryCatch({
      roc_obj <- roc(response = (actual == class_label), 
                     predictor = probs[, class_label])
      auc_values[i] <- auc(roc_obj)
      roc_obj
    }, error = function(e) {
      warning(paste("Error calculating ROC for class", class_label, ":", e$message))
      NULL
    })
  }
  
  # Extract performance metrics
  if(is.matrix(cm$byClass)) {
    macro_precision <- mean(cm$byClass[, "Precision"], na.rm = TRUE)
    macro_recall <- mean(cm$byClass[, "Recall"], na.rm = TRUE)
    macro_f1 <- mean(cm$byClass[, "F1"], na.rm = TRUE)
  } else {
    macro_precision <- cm$byClass["Precision"]
    macro_recall <- cm$byClass["Recall"]
    macro_f1 <- cm$byClass["F1"]
  }
  
  # Return metrics
  return(list(
    Accuracy = cm$overall["Accuracy"],
    Macro_Precision = macro_precision,
    Macro_Recall = macro_recall,
    Macro_F1 = macro_f1,
    Macro_AUC = mean(auc_values, na.rm = TRUE),
    AUC_by_class = auc_values,
    ROC_objects = roc_objects,
    ConfusionMatrix = cm$table
  ))
}

# --------------------------- Step 10: Evaluate All Models ---------------------------
# Set up tier models
tier_models <- list(
  low = tier_low_model,
  high = tier_high_model
)

# Get predictions for each model type
all_train_preds <- get_ensemble_predictions(tier_models, specialized_models, train)
all_test_preds <- get_ensemble_predictions(tier_models, specialized_models, test)

# Calculate performance metrics for each model
all_train_metrics <- list()
all_test_metrics <- list()

for(model_key in names(specialized_models)) {
  train_preds <- all_train_preds[[model_key]]
  test_preds <- all_test_preds[[model_key]]
  
  # Calculate confusion matrices
  train_cm <- confusionMatrix(train_preds$final_pred, train$Calories.Class)
  test_cm <- confusionMatrix(test_preds$final_pred, test$Calories.Class)
  
  # Calculate detailed metrics
  train_metrics <- calculate_metrics(train_cm, train_preds$weighted_preds, train$Calories.Class)
  test_metrics <- calculate_metrics(test_cm, test_preds$weighted_preds, test$Calories.Class)
  
  # Store metrics
  all_train_metrics[[model_key]] <- train_metrics
  all_test_metrics[[model_key]] <- test_metrics
}

# Print detailed results for each model
print_all_model_results <- function(all_train_metrics, all_test_metrics) {
  cat("\n\n===== Ensemble Specialization Results for All Models =====\n")
  
  # Create comparison table
  model_comparison <- data.frame(
    Model = character(),
    Train_Accuracy = numeric(),
    Test_Accuracy = numeric(),
    Test_Macro_F1 = numeric(),
    Test_Macro_AUC = numeric()
  )
  
  for(model_key in names(all_train_metrics)) {
    train_metrics <- all_train_metrics[[model_key]]
    test_metrics <- all_test_metrics[[model_key]]
    
    cat("\n\n----- Results for", model_key, "model -----\n")
    
    cat("\nTraining Performance:\n")
    cat("Accuracy:", train_metrics$Accuracy, "\n")
    cat("Macro Precision:", train_metrics$Macro_Precision, "\n")
    cat("Macro Recall:", train_metrics$Macro_Recall, "\n")
    cat("Macro F1:", train_metrics$Macro_F1, "\n")
    cat("Macro AUC:", train_metrics$Macro_AUC, "\n")
    
    cat("\nTest Performance:\n")
    cat("Accuracy:", test_metrics$Accuracy, "\n")
    cat("Macro Precision:", test_metrics$Macro_Precision, "\n")
    cat("Macro Recall:", test_metrics$Macro_Recall, "\n")
    cat("Macro F1:", test_metrics$Macro_F1, "\n")
    cat("Macro AUC:", test_metrics$Macro_AUC, "\n")
    
    cat("\nTest Confusion Matrix:\n")
    print(test_metrics$ConfusionMatrix)
    
    # Add to comparison table
    model_comparison <- rbind(model_comparison, data.frame(
      Model = model_key,
      Train_Accuracy = train_metrics$Accuracy,
      Test_Accuracy = test_metrics$Accuracy,
      Test_Macro_F1 = test_metrics$Macro_F1,
      Test_Macro_AUC = test_metrics$Macro_AUC
    ))
  }
  
  # Print the comparison table
  cat("\n\n----- Model Comparison Summary -----\n")
  print(model_comparison)
  
  # Find the best performing model
  best_model_idx <- which.max(model_comparison$Test_Accuracy)
  cat("\nBest performing model by test accuracy:", model_comparison$Model[best_model_idx], 
      "with accuracy", model_comparison$Test_Accuracy[best_model_idx], "\n")
  
  return(model_comparison)
}

# Print results
model_comparison <- print_all_model_results(all_train_metrics, all_test_metrics)

# --------------------------- Step 11: Add ROC Curve Visualization ---------------------------
# Function to create ROC curve plots for each class using plotly
# Fixed function to create ROC curve plots for each class using plotly
plot_roc_curves <- function(roc_objects, title_prefix) {
  # Create separate plots for each class
  for (class_name in names(roc_objects)) {
    roc_obj <- roc_objects[[class_name]]
    
    if (is.null(roc_obj)) {
      next
    }
    
    # Create dataframe for plotly
    roc_df <- data.frame(
      FPR = 1 - roc_obj$specificities,  # Convert to proper FPR
      TPR = roc_obj$sensitivities
    )
    
    # Add thresholds to the dataframe - ensuring same length
    # Some ROC implementations may have different length thresholds, so handle safely
    if (length(roc_obj$thresholds) == length(roc_df$FPR)) {
      roc_df$Threshold <- roc_obj$thresholds
    } else {
      # If lengths don't match, create a placeholder
      roc_df$Threshold <- NA
    }
    
    # Create plotly figure
    fig <- plot_ly(roc_df, x = ~FPR, y = ~TPR, type = 'scatter', mode = 'lines',
                   name = paste0("Class: ", class_name),
                   line = list(color = 'blue')) %>%
      add_trace(x = c(0, 1), y = c(0, 1), mode = 'lines', line = list(color = 'red', dash = 'dash'),
                name = 'Random Classifier') %>%
      layout(title = paste0(title_prefix, " ROC Curve for Class: ", class_name, 
                            " (AUC = ", round(auc(roc_obj), 3), ")"),
             xaxis = list(title = "False Positive Rate (1-Specificity)", zeroline = FALSE),
             yaxis = list(title = "True Positive Rate (Sensitivity)", zeroline = FALSE),
             legend = list(x = 1, y = 0),
             shapes = list(
               list(type = "rect", 
                    fillcolor = "transparent", 
                    line = list(color = "gray", dash = "dot"), 
                    opacity = 0.3, 
                    x0 = 0, x1 = 1, y0 = 0, y1 = 1)
             ))
    
    print(fig)
  }
}

# Create ROC curve plots for the best model
best_model_key <- model_comparison$Model[which.max(model_comparison$Test_Accuracy)]
plot_roc_curves(all_train_metrics[[best_model_key]]$ROC_objects, "Training")
plot_roc_curves(all_test_metrics[[best_model_key]]$ROC_objects, "Test")

# Function to create combined ROC curve plot for all classes
plot_combined_roc_curves <- function(all_metrics, dataset_name) {
  # Get best model ROC objects
  best_model_key <- model_comparison$Model[which.max(model_comparison$Test_Accuracy)]
  roc_objects <- all_metrics[[best_model_key]]$ROC_objects
  
  # Create plotly figure
  fig <- plot_ly() %>%
    layout(title = paste0(dataset_name, " ROC Curves - All Classes"),
           xaxis = list(title = "False Positive Rate (1-Specificity)", zeroline = FALSE),
           yaxis = list(title = "True Positive Rate (Sensitivity)", zeroline = FALSE),
           legend = list(x = 1, y = 0.5),
           shapes = list(
             list(type = "line", 
                  line = list(color = "gray", dash = "dash"), 
                  opacity = 0.3, 
                  x0 = 0, x1 = 1, y0 = 0, y1 = 1)
           ))
  
  # Color palette for different classes
  colors <- c("blue", "green", "red", "purple", "orange", "brown", "pink")
  
  # Add each class ROC curve
  i <- 1
  for (class_name in names(roc_objects)) {
    roc_obj <- roc_objects[[class_name]]
    
    if (is.null(roc_obj)) {
      next
    }
    
    # Create dataframe for plotly - using 1-specificity for FPR
    roc_df <- data.frame(
      FPR = 1 - roc_obj$specificities,
      TPR = roc_obj$sensitivities
    )
    
    # Get color for this class
    color_idx <- (i - 1) %% length(colors) + 1
    
    # Add to plot
    fig <- fig %>% add_trace(
      data = roc_df,
      x = ~FPR, 
      y = ~TPR, 
      type = 'scatter', 
      mode = 'lines',
      name = paste0(class_name, " (AUC = ", round(auc(roc_obj), 3), ")"),
      line = list(color = colors[color_idx])
    )
    
    i <- i + 1
  }
  
  # Add random classifier line
  fig <- fig %>% add_trace(
    x = c(0, 1), 
    y = c(0, 1), 
    mode = 'lines', 
    line = list(color = 'black', dash = 'dash'),
    name = 'Random Classifier'
  )
  
  print(fig)
}

# Create combined ROC curves
plot_combined_roc_curves(all_train_metrics, "Training")
plot_combined_roc_curves(all_test_metrics, "Test")

# --------------------------- Step 12: Feature Importance Visualization ---------------------------
# Extract and visualize feature importance for the tier classifier
plot_feature_importance <- function(model, title, num_features = 20) {
  # Check if model is NULL
  if (is.null(model)) {
    warning("Model is NULL, cannot extract feature importance")
    return(NULL)
  }
  
  # For Random Forest models (ranger or rf)
  if (inherits(model, "train") && (model$method == "ranger" || model$method == "rf")) {
    # Try different methods to extract importance
    imp_df <- tryCatch({
      # First try varImp from caret
      imp <- varImp(model)
      
      # Check if importance is available
      if (!is.null(imp$importance)) {
        data.frame(
          Feature = rownames(imp$importance),
          Importance = imp$importance$Overall
        )
      } else {
        # If not, try to extract directly from finalModel
        if (model$method == "ranger" && !is.null(model$finalModel$variable.importance)) {
          data.frame(
            Feature = names(model$finalModel$variable.importance),
            Importance = model$finalModel$variable.importance
          )
        } else if (model$method == "rf" && !is.null(model$finalModel$importance)) {
          imp_mat <- model$finalModel$importance
          data.frame(
            Feature = rownames(imp_mat),
            Importance = imp_mat[, "MeanDecreaseGini"]
          )
        } else {
          # Last resort - run importance calculation if model allows
          if (model$method == "ranger") {
            # Re-run ranger with importance
            new_model <- ranger(
              formula = model$terms,
              data = model$trainingData,
              importance = "impurity",
              num.trees = model$finalModel$num.trees
            )
            data.frame(
              Feature = names(new_model$variable.importance),
              Importance = new_model$variable.importance
            )
          } else {
            NULL
          }
        }
      }
    }, error = function(e) {
      warning("Error extracting feature importance: ", e$message)
      NULL
    })
    
    if (is.null(imp_df)) {
      warning("Could not extract feature importance from this random forest model")
      return(NULL)
    }
    
  } else if (inherits(model, "train") && model$method == "xgbTree") {
    # For XGBoost models - try direct extraction or importance function
    imp_df <- tryCatch({
      # Try xgb.importance if model structure allows
      if (!is.null(model$finalModel$feature_names)) {
        imp <- xgb.importance(model$finalModel$feature_names, model = model$finalModel)
        data.frame(
          Feature = imp$Feature,
          Importance = imp$Gain
        )
      } else {
        # Try caret's varImp
        imp <- varImp(model)
        data.frame(
          Feature = rownames(imp$importance),
          Importance = imp$importance$Overall
        )
      }
    }, error = function(e) {
      warning("Error extracting XGBoost feature importance: ", e$message)
      NULL
    })
    
    if (is.null(imp_df)) {
      warning("Could not extract feature importance from this XGBoost model")
      return(NULL)
    }
    
  } else if (inherits(model, "train") && model$method == "rpart") {
    # For decision trees
    imp_df <- tryCatch({
      imp <- varImp(model)
      data.frame(
        Feature = rownames(imp$importance),
        Importance = imp$importance$Overall
      )
    }, error = function(e) {
      warning("Error extracting decision tree importance: ", e$message)
      NULL
    })
    
  } else {
    # Try generic varImp for other model types
    imp_df <- tryCatch({
      imp <- varImp(model)
      
      if (!is.null(imp$importance)) {
        data.frame(
          Feature = rownames(imp$importance),
          Importance = if ("Overall" %in% colnames(imp$importance)) imp$importance$Overall else imp$importance[,1]
        )
      } else {
        NULL
      }
    }, error = function(e) {
      warning(paste("Feature importance not available for this model type:", class(model)[1]))
      NULL
    })
  }
  
  # If we couldn't extract importance, return NULL
  if (is.null(imp_df) || nrow(imp_df) == 0) {
    warning("Could not extract feature importance")
    return(NULL)
  }
  
  # Sort and select top features
  imp_df <- imp_df[order(imp_df$Importance, decreasing = TRUE), ]
  if (nrow(imp_df) > num_features) {
    imp_df <- imp_df[1:num_features, ]
  }
  
  # Reverse for better visualization
  imp_df <- imp_df[order(imp_df$Importance), ]
  
  # Create plotly bar chart
  fig <- plot_ly(imp_df, 
                 x = ~Importance, 
                 y = ~Feature, 
                 type = 'bar', 
                 orientation = 'h',
                 text = ~round(Importance, 3),
                 textposition = 'auto',
                 insidetextanchor = 'middle',
                 marker = list(color = 'rgba(50, 102, 193, 0.7)',
                               line = list(color = 'rgba(50, 102, 193, 1.0)', width = 1))) %>%
    layout(title = title,
           xaxis = list(title = "Importance"),
           yaxis = list(title = ""),
           margin = list(l = 200))  # Increase left margin for long feature names
  
  print(fig)
  
  return(imp_df)
}

# Alternative simplified version that just uses caret's varImp
simple_plot_feature_importance <- function(model, title, num_features = 20) {
  # Check if model is NULL
  if (is.null(model)) {
    warning("Model is NULL, cannot extract feature importance")
    return(NULL)
  }
  
  # Skip non-train objects and constant_model class
  if (!inherits(model, "train") || inherits(model, "constant_model")) {
    warning("Feature importance only available for caret train objects")
    return(NULL)
  }
  
  # Try to extract importance using caret's built-in method
  tryCatch({
    # Use varImp to get variable importance (works with most caret models)
    imp <- varImp(model, scale = TRUE)
    
    # Convert to data frame for plotting
    imp_df <- data.frame(
      Feature = rownames(imp$importance),
      Importance = imp$importance[,1]  # Take first column (often named "Overall")
    )
    
    # Sort and select top features
    imp_df <- imp_df[order(imp_df$Importance, decreasing = TRUE), ]
    if (nrow(imp_df) > num_features) {
      imp_df <- imp_df[1:num_features, ]
    }
    
    # Reverse for better visualization
    imp_df <- imp_df[order(imp_df$Importance), ]
    
    # Create plotly bar chart
    fig <- plot_ly(imp_df, 
                   x = ~Importance, 
                   y = ~Feature, 
                   type = 'bar', 
                   orientation = 'h',
                   text = ~round(Importance, 3),
                   textposition = 'auto',
                   insidetextanchor = 'middle',
                   marker = list(color = 'rgba(50, 102, 193, 0.7)',
                                 line = list(color = 'rgba(50, 102, 193, 1.0)', width = 1))) %>%
      layout(title = title,
             xaxis = list(title = "Importance"),
             yaxis = list(title = ""),
             margin = list(l = 200))  # Increase left margin for long feature names
    
    print(fig)
    
    return(imp_df)
  }, error = function(e) {
    warning(paste("Error extracting/plotting feature importance:", e$message))
    return(NULL)
  })
}

# Plot feature importance for tier classifiers
cat("\n\n===== Feature Importance Analysis =====\n")

# Only plot importance if the model is not a constant model
if (!inherits(tier_low_model, "constant_model")) {
  low_tier_importance <- simple_plot_feature_importance(tier_low_model, "Feature Importance - Low Tier Classifier")
} else {
  cat("Skipping feature importance for low tier - constant model\n")
}

if (!inherits(tier_high_model, "constant_model")) {
  high_tier_importance <- simple_plot_feature_importance(tier_high_model, "Feature Importance - High Tier Classifier")
} else {
  cat("Skipping feature importance for high tier - constant model\n")
}

# And for the best model
best_model_key <- model_comparison$Model[which.max(model_comparison$Test_Accuracy)]
best_models <- specialized_models[[best_model_key]]

# Check each tier model before trying to plot importance
if (!inherits(best_models$low, "constant_model")) {
  low_imp <- tryCatch(
    simple_plot_feature_importance(best_models$low, paste0("Feature Importance - Best ", best_model_key, " Low Tier Model")),
    error = function(e) NULL
  )
} else {
  cat("Skipping feature importance for best low tier - constant model\n")
}

if (!inherits(best_models$mid, "constant_model")) {
  mid_imp <- tryCatch(
    plot_feature_importance(best_models$mid, paste0("Feature Importance - Best ", best_model_key, " Mid Tier Model")),
    error = function(e) NULL
  )
} else {
  cat("Skipping feature importance for best mid tier - constant model\n")
}

if (!inherits(best_models$high, "constant_model")) {
  high_imp <- tryCatch(
    plot_feature_importance(best_models$high, paste0("Feature Importance - Best ", best_model_key, " High Tier Model")),
    error = function(e) NULL
  )
} else {
  cat("Skipping feature importance for best high tier - constant model\n")
}

low_imp <- NULL
high_imp <- NULL

# --------------------------- Step 13: Overall Feature Importance Analysis ---------------------------
# Combine feature importance across all tiers to identify globally important features
calculate_combined_importance <- function() {
  # Start with importance from tier classifiers
  combined_importance <- data.frame()
  
  # Process low tier importance
  if (!is.null(low_tier_importance)) {
    low_tier_importance$Source <- "Low Tier Classifier"
    combined_importance <- rbind(combined_importance, low_tier_importance)
  }
  
  # Process high tier importance
  if (!is.null(high_tier_importance)) {
    high_tier_importance$Source <- "High Tier Classifier"
    combined_importance <- rbind(combined_importance, high_tier_importance)
  }
  
  # Add best model importances if available
  if (!is.null(low_imp)) {
    low_imp$Source <- paste0("Best ", best_model_key, " Low Tier")
    combined_importance <- rbind(combined_importance, low_imp)
  }
  
  if (!is.null(mid_imp)) {
    mid_imp$Source <- paste0("Best ", best_model_key, " Mid Tier")
    combined_importance <- rbind(combined_importance, mid_imp)
  }
  
  if (!is.null(high_imp)) {
    high_imp$Source <- paste0("Best ", best_model_key, " High Tier")
    combined_importance <- rbind(combined_importance, high_imp)
  }
  
  # If we have any importance data
  if (nrow(combined_importance) > 0) {
    # Aggregate importance by feature
    agg_importance <- aggregate(Importance ~ Feature, data = combined_importance, FUN = mean)
    agg_importance <- agg_importance[order(agg_importance$Importance, decreasing = TRUE), ]
    
    # Take top features
    top_n <- min(20, nrow(agg_importance))
    top_features <- agg_importance[1:top_n, ]
    
    # Reverse for better visualization
    top_features <- top_features[order(top_features$Importance), ]
    
    # Create plotly bar chart
    fig <- plot_ly(top_features, 
                   x = ~Importance, 
                   y = ~Feature, 
                   type = 'bar', 
                   orientation = 'h',
                   text = ~round(Importance, 3),
                   textposition = 'auto',
                   insidetextanchor = 'middle',
                   marker = list(color = 'rgba(193, 66, 66, 0.7)',
                                 line = list(color = 'rgba(193, 66, 66, 1.0)', width = 1))) %>%
      layout(title = "Combined Feature Importance Across All Models",
             xaxis = list(title = "Average Importance"),
             yaxis = list(title = ""),
             margin = list(l = 200))  # Increase left margin for long feature names
    
    print(fig)
    
    return(top_features)
  } else {
    warning("No feature importance data available")
    return(NULL)
  }
}

# Calculate and plot combined importance
combined_importance <- calculate_combined_importance()

# --------------------------- Step 14: Generate Interactive Dashboard of Results ---------------------------
# Function to create an interactive heatmap of confusion matrices
plot_confusion_matrix_heatmap <- function(confusion_matrix, title) {
  # Convert confusion matrix to dataframe
  cm_df <- as.data.frame(as.table(confusion_matrix))
  names(cm_df) <- c("Actual", "Predicted", "Count")
  
  # Calculate percentages
  total <- sum(cm_df$Count)
  cm_df$Percentage <- round(100 * cm_df$Count / total, 1)
  
  # Create text for display
  cm_df$DisplayText <- paste0(cm_df$Count, "\n(", cm_df$Percentage, "%)")
  cm_df$HoverText <- paste0(
    "Actual: ", cm_df$Actual, "\n",
    "Predicted: ", cm_df$Predicted, "\n",
    "Count: ", cm_df$Count, "\n",
    "Percentage: ", cm_df$Percentage, "%"
  )
  
  # Create color scale - darker blue for higher values
  colorscale <- list(
    c(0, 'lightblue'),
    c(0.5, '#6baed6'),
    c(1, '#08306b')
  )
  
  # Create the plot
  fig <- plot_ly(
    data = cm_df,
    x = ~Predicted,
    y = ~Actual,
    z = ~Count,
    type = "heatmap",
    colorscale = colorscale,
    text = ~HoverText,
    hoverinfo = "text",
    showscale = TRUE
  ) %>%
    add_annotations(
      x = ~Predicted,
      y = ~Actual,
      text = ~DisplayText,
      showarrow = FALSE,
      font = list(color = "white", size = 12)  # White text for better visibility
    ) %>%
    layout(
      title = list(
        text = title,
        x = 0.05,  # Adjust title position
        y = 0.95,
        xanchor = "left"
      ),
      xaxis = list(
        title = "Predicted Class",
        side = "top",  # Move x-axis to top
        tickangle = -45  # Rotate labels for better fit
      ),
      yaxis = list(title = "Actual Class"),
      margin = list(
        l = 120,  # Left margin
        r = 50,   # Right margin
        b = 120,  # Bottom margin
        t = 100,  # Top margin
        pad = 10  # Padding
      )
    )
  
  print(fig)
}

# Plot confusion matrices for the best model
best_model_key <- model_comparison$Model[which.max(model_comparison$Test_Accuracy)]
plot_confusion_matrix_heatmap(all_train_metrics[[best_model_key]]$ConfusionMatrix, 
                              paste0("Training Confusion Matrix - ", best_model_key, " Model"))
plot_confusion_matrix_heatmap(all_test_metrics[[best_model_key]]$ConfusionMatrix, 
                              paste0("Test Confusion Matrix - ", best_model_key, " Model"))

# Create model performance comparison plot
plot_model_comparison <- function(model_comparison) {
  # Reshape data for plotting
  plot_data <- model_comparison %>%
    tidyr::pivot_longer(cols = c("Train_Accuracy", "Test_Accuracy", "Test_Macro_F1", "Test_Macro_AUC"),
                        names_to = "Metric", values_to = "Value")
  
  # Format values for display
  plot_data$Value_Text <- sprintf("%.3f", plot_data$Value)
  
  # Create grouped bar chart with values
  fig <- plot_ly(plot_data, 
                 x = ~Model, 
                 y = ~Value, 
                 color = ~Metric, 
                 type = "bar",
                 text = ~Value_Text,
                 textposition = "auto") %>%
    layout(title = "Model Performance Comparison",
           xaxis = list(title = "Model"),
           yaxis = list(title = "Performance", range = c(0, 1)),
           barmode = "group",
           legend = list(title = list(text = "Metric")))
  
  print(fig)
}

# Plot model comparison
plot_model_comparison(model_comparison)

# Stop the parallel cluster
stopCluster(cl)
