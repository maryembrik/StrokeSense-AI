"""Cleaning and feature engineering. Everything that learns from data does so in fit()."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression

TARGET = "stroke"
NUM = ["age", "avg_glucose_level", "bmi"]
BIN = ["hypertension", "heart_disease"]
CAT = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
RAW = NUM + BIN + CAT


def _basis(age):
    a = np.asarray(age, float)
    return np.column_stack([a, np.maximum(a - 18, 0), np.maximum(a - 45, 0), np.maximum(a - 65, 0)])


class StrokeFeatures(BaseEstimator, TransformerMixin):
    """bmi: 'median' | 'peer'  (peer = expected BMI for that age).
    bmi_flag: add a missing-BMI indicator (suspected outcome leak; off by default).
    peer: add deviation of BMI and log-glucose from same-age peers, in robust SDs.
    smoke_split: split 'Unknown' smoking into structural (minor) vs. not recorded (adult).
    burden: count of hypertension, heart disease, glucose >= 126."""

    def __init__(self, bmi="median", bmi_flag=False, peer=False, smoke_split=False, burden=False):
        self.bmi, self.bmi_flag, self.peer = bmi, bmi_flag, peer
        self.smoke_split, self.burden = smoke_split, burden

    def fit(self, X, y=None):
        ok = X["bmi"].notna()
        self.bmi_median_ = X["bmi"].median()
        self.bmi_lr_ = LinearRegression().fit(_basis(X.loc[ok, "age"]), np.log(X.loc[ok, "bmi"]))
        lg = np.log(X["avg_glucose_level"])
        self.glu_lr_ = LinearRegression().fit(_basis(X["age"]), lg)
        mad = lambda r: 1.4826 * np.median(np.abs(r - np.median(r)))
        self.bmi_s_ = mad(np.log(X.loc[ok, "bmi"]) - self.bmi_lr_.predict(_basis(X.loc[ok, "age"])))
        self.glu_s_ = mad(lg - self.glu_lr_.predict(_basis(X["age"])))
        return self

    def transform(self, X):
        X = X[RAW].copy()
        miss = X["bmi"].isna()
        exp_bmi = self.bmi_lr_.predict(_basis(X["age"]))
        X["bmi"] = X["bmi"].fillna(pd.Series(np.exp(exp_bmi), X.index) if self.bmi == "peer" else self.bmi_median_)
        if self.bmi_flag:
            X["bmi_missing"] = miss.astype(int)
        if self.peer:
            X["bmi_vs_peers"] = (np.log(X["bmi"]) - exp_bmi) / self.bmi_s_
            X["glucose_vs_peers"] = (np.log(X["avg_glucose_level"]) - self.glu_lr_.predict(_basis(X["age"]))) / self.glu_s_
        if self.smoke_split:
            X.loc[(X["smoking_status"] == "Unknown") & (X["age"] < 18), "smoking_status"] = "Minor (not asked)"
        if self.burden:
            X["burden"] = X["hypertension"] + X["heart_disease"] + (X["avg_glucose_level"] >= 126).astype(int)
        return X
