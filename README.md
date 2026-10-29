# Yotube_video_prediction
===================================================================

                  YOUTUBE VIEWS PREDICTION MODEL

===================================================================



* PROJECT OVERVIEW

  - Successfully resolved severe positive skewness (power-law distribution) typical of video platforms by operating on a logarithmic scale.





* TARGET & FEATURE ARCHITECTURE

  - Target Variable:

    * Transformed actual view count using: log_views = np.log1p(views)

    * Inverts predicted log values back to real view numbers using: pred_views = np.expm1(pred_log)

  - Numeric Features:

    * log_likes: Natural logarithm of like count [np.log1p(likes)]

    * log_dislikes: Natural logarithm of dislike count [np.log1p(dislikes)]

    * log_comment_count: Natural logarithm of comment count [np.log1p(comment_count)]

    * title_length: Raw character count of video title

  - Categorical Feature:

    * category_name: Topic category (e.g., Entertainment, Music, Gaming, Education)





* CORE COMPONENT: SCLEAN COLUMNTRANSFORMER

  - Centralized feature processor orchestrating distinct transformations per column type without data leakage:

    * Numeric Pipeline: Passes 'log_likes', 'log_dislikes', 'log_comment_count', and 'title_length' through StandardScaler to normalize features to zero mean and unit variance.

    * Categorical Pipeline: Passes 'category_name' through OneHotEncoder(drop='first', handle_unknown='ignore') to generate binary dummy columns while avoiding multicollinearity (the dummy variable trap).

  - Encapsulated directly into a single Scikit-Learn Pipeline so raw DataFrames can be passed directly into inference without manual preprocessing scripts.





* CORE MODEL: LINEAR REGRESSION

  - Model Choice: Ordinary Least Squares (OLS) LinearRegression from scikit-learn.

  - Why Linear Regression Fits Here:

    * High interpretability: Feature coefficients indicate direct elasticities between engagement percentages and view volume.

    * Log-Log Relationship: With both engagement counts and target views on the log scale, linear regression naturally models power-law scaling relationships.

  - Cross-Validation:

    * Evaluated using 5-Fold Cross-Validation on the training dataset.

    * Achieved a mean R^2 score of ~0.77 (explaining ~77% of variance in view counts across folds).





* PROJECT REPOSITORY STRUCTURE

  ml1/

  │

  ├── data/

  │   └── yt_video.ipynb              # Complete EDA, preprocessing, training & validation

  │

  ├── youtube_view_predictor.pkl       # Exported full pipeline (ColumnTransformer + LinearRegression)

  ├── README.txt                       # Project documentation

  └── requirements.txt                 # Project dependencies





* HOW TO CLONE AND RUN LOCALLY

  1. Clone repository from GitHub:

     git clone https://github.com/princetandon80/<YOUR_REPO_NAME>.git

     cd <YOUR_REPO_NAME>



  2. Create and activate a Python virtual environment:

     python -m venv .venv

     # Windows (PowerShell):

     .venv\Scripts\Activate.ps1

     # macOS/Linux:

     source .venv/bin/activate



  3. Install required libraries:

     pip install numpy pandas scikit-learn joblib





* SAMPLE INFERENCE INPUT & PREDICTION

  ```python

  import joblib

  import numpy as np

  import pandas as pd



  # Load the trained full pipeline

  model = joblib.load("youtube_view_predictor.pkl")



  # Define sample raw input video

  raw_video_input = pd.DataFrame([{

      "log_likes": np.log1p(45000),         # 45,000 likes

      "log_dislikes": np.log1p(800),        # 800 dislikes

      "log_comment_count": np.log1p(3200),  # 3,200 comments

      "title_length": 45,                   # 45 characters in title

      "category_name": "Entertainment"

  }])



  # Generate prediction on log scale

  predicted_log = model.predict(raw_video_input)



  # Convert back to real views

  predicted_views = np.expm1(predicted_log)

  print(f"Predicted Views: {int(round(predicted_views[0])):,}")

  ```





* TECH STACK

  - Language: Python

  - Data Processing: Pandas, NumPy

  - Machine Learning: Scikit-Learn (ColumnTransformer, StandardScaler, OneHotEncoder, LinearRegression, Pipeline)

  - Serialization: Joblib

  - Version Control: Git & GitHub

===================================================================
