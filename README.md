# Airbnb Istanbul Price Prediction

This project uses machine learning to predict Airbnb listing prices in Istanbul. It includes data cleaning, feature engineering, model training, and visualization.

---

## Files
- `listings.csv`: Raw Airbnb dataset used for training and analysis.
- `TermProject.ipynb`: Contains all the logic for preprocessing, modeling, and visualization.

---

## Project Structure

### 1. Importing Libraries
Imports all required Python libraries:
- Data manipulation: `pandas`, `numpy`
- Visualization: `matplotlib`, `seaborn`
- Modeling: `sklearn.linear_model`, `RandomForestRegressor`, `Pipeline`, `GridSearchCV`, etc.

---

### 2. Data Loading
- Uploads and reads the `listings.csv` dataset.
- Performs basic inspection (`.head()`, `.info()`, `.isna()`).

---

### 3. Data Cleaning & Feature Engineering
- Converts `last_review` to datetime and creates a new feature: `days_since_last_review`.
- One-hot encodes `neighbourhood` and `room_type`.
- Fills missing `reviews_per_month` with 0.
- Drops irrelevant columns such as `id`, `name`, `host_id`, coordinates, etc.
- Reorders columns to place `price` next to `log_price`.

---

### 4. Outlier Detection & Transformation
Applies outlier clipping or transformation on:
- `minimum_nights`: Clipped at 60.
- `number_of_reviews`: Clipped at 99th percentile (119).
- `reviews_per_month`: Clipped at 99th percentile (3.5).
- `calculated_host_listings_count`: Clipped using IQR method.
- `price`: Clipped at 99th percentile, then log-transformed as `log_price`.

---

### 5. Feature Correlation Analysis
- Boxplots of cleaned numeric columns.
- Heatmap of correlation matrix.
- Visualizations of room type and neighbourhood distributions and their impact on price.

---

### 6. Model Training and Evaluation

#### Models Trained:
- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**
- **Random Forest Regressor**

#### Methodology:
- Uses a shared pipeline (`StandardScaler` + `PolynomialFeatures`).
- Performs `GridSearchCV` with cross-validation for hyperparameter tuning.
- R² score is used as the performance metric.
- Best model configurations and test performance are saved and printed.

---

### 7. Visual Evaluation
- Scatter plots of predicted vs. actual `log_price` for best linear model and Random Forest.
- Geographic price heatmap over Istanbul using latitude/longitude grid bins.

---

## How to Run

1. Open in Google Colab or Jupyter Notebook.
2. Upload the `listings.csv` dataset when prompted.
3. Run the notebook or script cells sequentially.

---

## Results Summary
- The best model is selected based on highest R² on the test set.
- Random Forest typically performs better on non-linear relationships.
- Final visualizations show prediction accuracy and spatial distribution of prices.

---

## Dependencies

- Python 3.7+
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- google.colab (for file upload on Colab)
