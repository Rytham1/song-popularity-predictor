from flask import Flask, render_template, request
from model_utils import (
    predict_all_models,
    NUMERIC_INPUT_FIELDS,
    GENRE_OPTIONS,
    SUBGENRE_OPTIONS,
)

app = Flask(__name__)

DEFAULT_FORM_DATA = {
    "duration_ms": "210000",
    "tempo": "120",
    "loudness": "-5",
    "danceability": "0.6",
    "energy": "0.7",
    "valence": "0.5",
    "speechiness": "0.05",
    "acousticness": "0.1",
    "instrumentalness": "0.0",
    "liveness": "0.2",
    "mode": "1",
    "key": "5",
    "artist_avg_popularity": "50",
    "playlist_genre": "pop",
    "playlist_subgenre": "dance pop",
}


@app.route("/", methods=["GET", "POST"])
def index():
    form_data = DEFAULT_FORM_DATA.copy()
    results = None
    error = None

    if request.method == "POST":
        for field in NUMERIC_INPUT_FIELDS:
            form_data[field] = request.form.get(field, "")

        form_data["playlist_genre"] = request.form.get("playlist_genre", "")
        form_data["playlist_subgenre"] = request.form.get("playlist_subgenre", "")

        try:
            numeric_features = {
                field: float(form_data[field]) for field in NUMERIC_INPUT_FIELDS
            }

            results = predict_all_models(
                numeric_features=numeric_features,
                selected_genre=form_data["playlist_genre"],
                selected_subgenre=form_data["playlist_subgenre"],
            )
        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        form_data=form_data,
        numeric_fields=NUMERIC_INPUT_FIELDS,
        genre_options=GENRE_OPTIONS,
        subgenre_options=SUBGENRE_OPTIONS,
        results=results,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)