from flask import Flask, render_template, request
import pickle
import pandas as pd

# load the model
with open("./conf_model.pkl", "rb") as file:
    model1 = pickle.load(file)
with open("./deaths_model.pkl", "rb") as file:
    model2 = pickle.load(file)

app = Flask(__name__)

conf_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

days_since_1_22 = len(conf_df.columns)-5


@app.route("/", methods=["GET"])
def index():
    # sends the file page1.html from templates directory
    return render_template("home.html")


@app.route("/predict", methods=["GET"])
def predict_cases():
    # get values from request
    print(request.args)
    day = int(request.args.get("day"))
    conf = int(model1.predict([[day+days_since_1_22]]))
    deaths = int(model2.predict([[day + days_since_1_22]]))

    return f"<body background=\"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQoYUIqlDHmOWYenygNiUL0VNEXEQ-iRuunUQ&usqp=CAU\"marginheight=\"150\"><h1 align=\"center\">For The {day}th Day : <br> Confirmed Cases : {'{:,}'.format(conf)} <br> Deaths : {'{:,}'.format(deaths)} </h1></body>"


app.run(host="localhost", port=4500, debug=True)