# Song Popularity Predictor

An ML project that predicts whether a song will be a "hit" or "not hit" based on Spotify audio features. Originally approached as a regression task, the project pivoted to binary classification for better performance.

## Dataset

[Spotify Songs](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs) - ~32,000 tracks across 6 genres (EDM, Latin, Pop, R&B, Rap, Rock) with audio features like danceability, energy, tempo, and more.

## Models

### Regression (exploratory)

All regression models yielded poor results (R² ≈ 0.20), confirming that audio features alone can't predict exact popularity scores:

- Linear Regression (TensorFlow/Keras)
- KNN Regressor
- Neural Network (MLPRegressor)
- Random Forest Regressor
- XGBoost Regressor

### Classification (final)

Pivoted to binary classification - "Hit" (top 25% popularity) vs "Not Hit" (bottom 75%). Trained KNN, Neural Network, Logistic Regression, Random Forest, and XGBoost classifiers, then kept the 3 best for the app:

- **Logistic Regression** - highest recall (0.64), highest F1 (0.57)
- **Random Forest** - highest precision (0.61)
- **XGBoost** - best balance of precision (0.57) and recall (0.54)

## Tech Stack

Python, scikit-learn, XGBoost, TensorFlow/Keras, Flask, pandas, NumPy, matplotlib, seaborn

## Running the Notebook

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow kagglehub
jupyter notebook ecs_171_final_project.ipynb
```

## Running the Frontend

```bash
cd frontend
pip install flask pandas scikit-learn xgboost
python app.py
```

Then open `http://127.0.0.1:5000` in your browser. Enter audio features for a song and get hit/not-hit predictions from all three models.

## Report

[Project Report (PDF)](ECS_171_Project_Report.pdf)
