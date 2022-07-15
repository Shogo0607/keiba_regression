import streamlit as st
import pandas as pd
import numpy as np
from pycaret.regression import *
from PIL import Image

st.set_page_config(page_title="LightGBM")
st.title("LightGBM")

st.sidebar.title("LightGBM")
train_file = st.sidebar.file_uploader("学習用のCSVファイルを入力してください",type=["csv"])

if not train_file:
    st.warning("CSVファイルを入力してください")
    st.stop()

vote = st.sidebar.selectbox("投票番号を選択してください",("","1,2","3"))

if vote == "":
    st.warning("投票番号を入力してください")
    st.stop()


train_df = pd.read_csv(train_file, encoding="shift-jis")

if vote == "1,2":
    train_df = train_df[(train_df["投票"]==1)|(train_df["投票"]==2)]
    train_df = train_df[["頭数","生後日数","騎手指数","見習い騎手","種牡馬","父系統","母父系統","父年齢","母年齢","血統距離適性","厩舎ランク","調教師","生産者","馬主","蹄","斤量","斤量構成比","馬体重","予想IDM","コース枠番","奇数偶数","調教素点","厩舎素点","CID素点","CID","調教評価","厩舎評価","追切指数","終いF指数","仕上指数","調教パターン","輸送","外厩","放牧先ランク","騎手期待単勝率","騎手期待連対率","騎手期待複勝率","激走指数","LS指数","万券指数","基準支持率","単勝配当"]]

st.dataframe(train_df.head(5))

df = setup(data = train_df, target = "単勝配当",silent=True)
lightgbm = create_model("lightgbm")
lightgbm_tuned = tune_model(lightgbm,n_iter=100,optimize="R2")
plot_model(estimator=lightgbm_tuned,save=True)
image = Image.open("./Residuals.png")
st.image(image)