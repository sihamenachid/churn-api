import json
from typing import Literal

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ValidationError

app = FastAPI(title="Customer churn prediction")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Pipeline (preprocessing + model) saved by churn_prediction.ipynb
model = joblib.load("model.joblib")
with open("model_meta.json") as f:
    META = json.load(f)
FEATURES = META["features"]
ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]


class Customer(BaseModel):
    Contract: Literal["Month-to-month", "One year", "Two year"]
    tenure: int = Field(ge=0, le=72)
    PaymentMethod: Literal["Bank transfer (automatic)", "Credit card (automatic)",
                           "Electronic check", "Mailed check"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["No", "Yes"] = "No"
    OnlineBackup: Literal["No", "Yes"] = "No"
    DeviceProtection: Literal["No", "Yes"] = "No"
    TechSupport: Literal["No", "Yes"] = "No"


def predict_churn(customer: Customer) -> float:
    record = customer.model_dump()
    if record["InternetService"] == "No":  # same convention as the dataset
        for addon in ADDONS:
            record[addon] = "No internet service"
    X = pd.DataFrame([record], columns=FEATURES)
    return float(model.predict_proba(X)[0, 1])


# Data used to build the form
LABELS = {"Contract": "Contract", "tenure": "Months as a customer",
          "PaymentMethod": "Payment method", "InternetService": "Internet service",
          "OnlineSecurity": "Online security", "OnlineBackup": "Online backup",
          "DeviceProtection": "Device protection", "TechSupport": "Tech support"}
GROUPS = [("Account", ["Contract", "tenure", "PaymentMethod"]),
          ("Internet services", ["InternetService"] + ADDONS)]
DEFAULTS = {"Contract": "Month-to-month", "tenure": "", "PaymentMethod": "Electronic check",
            "InternetService": "Fiber optic", **{a: "No" for a in ADDONS}}


def render(request, values, result=None, errors=None, status_code=200):
    return templates.TemplateResponse(request, "index.html", {
        "groups": GROUPS, "labels": LABELS, "categorical": META["categorical"],
        "numeric": META["numeric"], "values": values, "errors": errors or {},
        "result": result, "meta": META}, status_code=status_code)


# ---------- HTML interface ----------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return render(request, DEFAULTS)


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    form = await request.form()
    values = {f: form.get(f, "") for f in FEATURES}
    try:
        customer = Customer(**values)
    except ValidationError as e:
        errors = {err["loc"][0]: err["msg"] for err in e.errors()}
        return render(request, values, errors=errors, status_code=422)
    proba = predict_churn(customer)
    result = {"probability": round(proba * 100), "churn": proba >= 0.5}
    return render(request, values, result=result)


# ---------- JSON API (Swagger: /docs) ----------
@app.post("/make_predictions")
async def make_predictions(customer: Customer):
    proba = predict_churn(customer)
    return {"prediction": "Likely to churn" if proba >= 0.5 else "Likely to stay",
            "churn_probability": round(proba, 3)}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
