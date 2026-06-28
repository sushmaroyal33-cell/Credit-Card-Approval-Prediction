from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        age = int(request.form["Age"])
        income = int(request.form["Income"])
        credit_score = int(request.form["CreditScore"])
        loan_amount = int(request.form["LoanAmount"])

        if credit_score >= 650 and income >= 30000:
            prediction = "✅ Credit Card Approved"
        else:
            prediction = "❌ Credit Card Rejected"

        return render_template("result.html", prediction=prediction)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)