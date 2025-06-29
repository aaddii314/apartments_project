#!/usr/bin/env python
# coding: utf-8

# In[10]:

# הבעיה כרגע היא שאתה מנסה להמיר מחרוזת ריקה ('') ל-float, וזה גורם לשגיאה
# נבצע טיפול מקדים שממיר ערכים ריקים ל-None (NaN) לפני ההמרה

from flask import Flask, render_template, request
import pandas as pd
import pickle
from assets_data_prep import prepare_data

app = Flask(__name__)

with open("trained_model.pkl", "rb") as f:
    model = pickle.load(f)

reference_tables = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global reference_tables
    try:
        # קבלת נתוני הטופס והמרת ערכים ריקים ל-None
        data = {
            key: (request.form[key] if request.form[key].strip() != '' else None)
            for key in request.form
        }

        # המרה לסוגים מתאימים
        numeric_fields = [
            "area", "room_num", "floor", "total_floors",
            "distance_from_center", "monthly_arnona", "building_tax", "garden_area"
        ]
        for field in numeric_fields:
            if data.get(field) is not None:
                data[field] = float(data[field])

        df = pd.DataFrame([data])

        # טבלאות עזר אם צריך
        if reference_tables is None:
            train_df = pd.read_csv("train.csv")
            _, reference_tables = prepare_data(train_df, mode="train")

        df_prepared = prepare_data(df, mode="test", reference_tables=reference_tables)

        prediction = model.predict(df_prepared)[0]
        prediction = round(prediction)

        return render_template("index.html", prediction=prediction)
    except Exception as e:
        return f"שגיאה בשרת: {e}"

if __name__ == "__main__":
    app.run(debug=True)

# In[ ]:





# In[ ]:




