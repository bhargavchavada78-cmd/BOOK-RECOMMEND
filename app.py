from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("book_model.pkl", "rb"))
tfidf_vectorizer = pickle.load(open("tfidf.pkl", "rb"))
df = pickle.load(open("books_data.pkl", "rb"))

def recommend_books(book_name):

    matches = df[df['title'].str.contains(
        book_name,
        case=False,
        na=False
    )]

    if matches.empty:
        return []

    idx = matches.index[0]
    query_vec = tfidf_vectorizer.transform([df.loc[idx, "combine"]])

    distances, indices = model.kneighbors(
        query_vec,
        n_neighbors=6
    )

    recommendations = []

    for i in indices[0][1:]:

        recommendations.append({
            "title": df.iloc[i]["title"],
            "author": df.iloc[i]["author"],
            "DESC": df.iloc[i]["desc"],
            "genre": df.iloc[i]["genre"]
        })

    return recommendations


@app.route("/", methods=["GET", "POST"])
def home():

    books = []

    if request.method == "POST":

        book_name = request.form["book_name"]

        books = recommend_books(book_name)

    return render_template(
        "index.html",
        books=books
    )


if __name__ == "__main__":
    app.run(debug=True)