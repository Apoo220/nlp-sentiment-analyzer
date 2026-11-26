"""Small Flask demo UI for the sentiment analyzer.

Run:  python app.py   ->  http://127.0.0.1:5000
"""

from flask import Flask, render_template, request

from sentiment import analyze

app = Flask(__name__)

SAMPLES = [
    "お好み焼きがとても美味しいです。店員さんも親切で、また来たいと思いました。",
    "注文した料理が来るまで40分かかりました。とても遅いです。",
    "味は普通でした。可もなく不可もなく、という感じです。",
]


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    text = ""
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if text:
            result = analyze(text)
    return render_template("index.html", result=result, text=text,
                           samples=SAMPLES)


if __name__ == "__main__":
    app.run(debug=True)
