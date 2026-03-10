import json
import joblib
import pandas as pd

FEATURE_FIELDS = [
    "duration_ms",
    "tempo",
    "loudness",
    "danceability",
    "energy",
    "valence",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "mode",
    "key",
    "playlist_genre_latin",
    "playlist_genre_pop",
    "playlist_genre_r&b",
    "playlist_genre_rap",
    "playlist_genre_rock",
    "playlist_subgenre_big room",
    "playlist_subgenre_classic rock",
    "playlist_subgenre_dance pop",
    "playlist_subgenre_electro house",
    "playlist_subgenre_electropop",
    "playlist_subgenre_gangster rap",
    "playlist_subgenre_hard rock",
    "playlist_subgenre_hip hop",
    "playlist_subgenre_hip pop",
    "playlist_subgenre_indie poptimism",
    "playlist_subgenre_latin hip hop",
    "playlist_subgenre_latin pop",
    "playlist_subgenre_neo soul",
    "playlist_subgenre_new jack swing",
    "playlist_subgenre_permanent wave",
    "playlist_subgenre_pop edm",
    "playlist_subgenre_post-teen pop",
    "playlist_subgenre_progressive electro house",
    "playlist_subgenre_reggaeton",
    "playlist_subgenre_southern hip hop",
    "playlist_subgenre_trap",
    "playlist_subgenre_tropical",
    "playlist_subgenre_urban contemporary",
    "artist_avg_popularity",
]

NUMERIC_FEATURE_COLUMNS = [
    "duration_ms",
    "tempo",
    "loudness",
    "danceability",
    "energy",
    "valence",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "mode",
    "key",
    "artist_avg_popularity",
]

NUMERIC_INPUT_FIELDS = [
    "duration_ms",
    "tempo",
    "loudness",
    "danceability",
    "energy",
    "valence",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "mode",
    "key",
    "artist_avg_popularity",
]

GENRE_OPTIONS = [
    "edm",
    "latin",
    "pop",
    "r&b",
    "rap",
    "rock",
]

SUBGENRE_OPTIONS = [
    "album rock",
    "big room",
    "classic rock",
    "dance pop",
    "electro house",
    "electropop",
    "gangster rap",
    "hard rock",
    "hip hop",
    "hip pop",
    "indie poptimism",
    "latin hip hop",
    "latin pop",
    "neo soul",
    "new jack swing",
    "permanent wave",
    "pop edm",
    "post-teen pop",
    "progressive electro house",
    "reggaeton",
    "southern hip hop",
    "trap",
    "tropical",
    "urban contemporary",
]

GENRE_DUMMY_COLUMNS = [
    "playlist_genre_latin",
    "playlist_genre_pop",
    "playlist_genre_r&b",
    "playlist_genre_rap",
    "playlist_genre_rock",
]

SUBGENRE_DUMMY_COLUMNS = [
    "playlist_subgenre_big room",
    "playlist_subgenre_classic rock",
    "playlist_subgenre_dance pop",
    "playlist_subgenre_electro house",
    "playlist_subgenre_electropop",
    "playlist_subgenre_gangster rap",
    "playlist_subgenre_hard rock",
    "playlist_subgenre_hip hop",
    "playlist_subgenre_hip pop",
    "playlist_subgenre_indie poptimism",
    "playlist_subgenre_latin hip hop",
    "playlist_subgenre_latin pop",
    "playlist_subgenre_neo soul",
    "playlist_subgenre_new jack swing",
    "playlist_subgenre_permanent wave",
    "playlist_subgenre_pop edm",
    "playlist_subgenre_post-teen pop",
    "playlist_subgenre_progressive electro house",
    "playlist_subgenre_reggaeton",
    "playlist_subgenre_southern hip hop",
    "playlist_subgenre_trap",
    "playlist_subgenre_tropical",
    "playlist_subgenre_urban contemporary",
]

BASELINE_GENRE = "edm"
BASELINE_SUBGENRE = "album rock"

logistic_model = joblib.load("logistic_model.pkl")
random_forest_model = joblib.load("random_forest_model.pkl")
xgboost_model = joblib.load("xgboost_model.pkl")

scaler = joblib.load("classification_scaler.pkl")

with open("classification_iqr_bounds.json", "r") as f:
    IQR_BOUNDS = json.load(f)

with open("classification_preprocessing_config.json", "r") as f:
    PREPROCESSING_CONFIG = json.load(f)

GLOBAL_TRAIN_MEAN = PREPROCESSING_CONFIG.get("global_train_mean", 50.0)


def apply_outlier_clipping(feature_frame, feature_to_bounds):
    clipped_features = feature_frame.copy()

    for feature_name, bounds in feature_to_bounds.items():
        lower_bound, upper_bound = bounds
        if feature_name in clipped_features.columns:
            clipped_features[feature_name] = clipped_features[feature_name].clip(
                lower_bound, upper_bound
            )

    return clipped_features


def build_feature_dict(numeric_features, selected_genre, selected_subgenre):
    features = {field: 0.0 for field in FEATURE_FIELDS}

    for field in NUMERIC_INPUT_FIELDS:
        features[field] = float(numeric_features[field])

    genre_column = f"playlist_genre_{selected_genre}"
    if genre_column in GENRE_DUMMY_COLUMNS:
        features[genre_column] = 1.0

    subgenre_column = f"playlist_subgenre_{selected_subgenre}"
    if subgenre_column in SUBGENRE_DUMMY_COLUMNS:
        features[subgenre_column] = 1.0

    return features


def preprocess_input(features):
    input_df = pd.DataFrame(
        [[features[field] for field in FEATURE_FIELDS]],
        columns=FEATURE_FIELDS
    )

    input_df = apply_outlier_clipping(input_df, IQR_BOUNDS)

    input_df[NUMERIC_FEATURE_COLUMNS] = scaler.transform(input_df[NUMERIC_FEATURE_COLUMNS])

    return input_df


def _label_from_prediction(pred_value):
    return "Hit" if int(pred_value) == 1 else "Not Hit"


def _probability_if_available(model, input_df):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)
        return float(probabilities[0][1])
    return None


def predict_all_models(numeric_features, selected_genre, selected_subgenre):
    features = build_feature_dict(
        numeric_features=numeric_features,
        selected_genre=selected_genre,
        selected_subgenre=selected_subgenre,
    )

    processed_df = preprocess_input(features)

    logistic_pred = logistic_model.predict(processed_df)[0]
    random_forest_pred = random_forest_model.predict(processed_df)[0]
    xgboost_pred = xgboost_model.predict(processed_df)[0]

    logistic_prob = _probability_if_available(logistic_model, processed_df)
    random_forest_prob = _probability_if_available(random_forest_model, processed_df)
    xgboost_prob = _probability_if_available(xgboost_model, processed_df)

    return {
        "Logistic Regression": {
            "prediction": _label_from_prediction(logistic_pred),
            "probability": logistic_prob,
        },
        "Random Forest": {
            "prediction": _label_from_prediction(random_forest_pred),
            "probability": random_forest_prob,
        },
        "XGBoost": {
            "prediction": _label_from_prediction(xgboost_pred),
            "probability": xgboost_prob,
        },
    }