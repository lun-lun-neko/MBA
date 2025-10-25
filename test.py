import os
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()


dataDirection = "./data"

trainMeta = pd.read_csv(os.path.join(dataDirection, "train.csv"))
trainA = pd.read_csv(os.path.join(dataDirection, "train", "A.csv"))
trainB = pd.read_csv(os.path.join(dataDirection, "train", "B.csv"))

def convert_age(val):   #나이 변환 메소드 
    if pd.isna(val): return np.nan
    try:
        base = int(str(val)[:-1])   #몇십대 인지 추출 후 int 변환
        return base if str(val)[-1] == "a" else base + 5  #b 붙으면 +5살
    except:
        return np.nan

def split_testdate(val):    #테스트 날짜 변환 년, 월 분리
    try:
        v = int(val)    #int 변환
        return v // 100, v % 100    #몫(년도), 나머지(월) 반환
    except:
        return np.nan, np.nan

def seq_mean(series):   #series의 평균값 반환 메소드
    return series.fillna("").progress_apply(
        lambda x: np.fromstring(x, sep=",").mean() if x else np.nan
    )   #series를 숫자 배열로 변환하여 평균값 반환, progress_apply는 진행률 표시

def seq_std(series):    #series의 표준편차 반환 메소드
    return series.fillna("").progress_apply(
        lambda x: np.fromstring(x, sep=",").std() if x else np.nan
    )

def seq_rate(series, target="1"):   #series의 target 비율 구하는 메소드
    return series.fillna("").progress_apply(
        lambda x: str(x).split(",").count(target) / len(x.split(",")) if x else np.nan
    )   #target="1"로 고정되어 있는데 고정 풀 필요 보임

#두 series에서 한 series를 조건으로 평균을 구하는 메소드
#cond_series는 조건이 될 series, val_series는 평균을 구할 유효 series mask_val 조건이 될 값
def masked_mean_from_csv_series(cond_series, val_series, mask_val):
    cond_df = cond_series.fillna("").str.split(",", expand=True).replace("", np.nan)
    val_df  = val_series.fillna("").str.split(",", expand=True).replace("", np.nan)

    cond_arr = cond_df.to_numpy(dtype=float)
    val_arr  = val_df.to_numpy(dtype=float)

    mask = (cond_arr == mask_val)
    with np.errstate(invalid="ignore"):
        sums = np.nansum(np.where(mask, val_arr, np.nan), axis=1)
        counts = np.sum(mask, axis=1)
        out = sums / np.where(counts==0, np.nan, counts)
    return pd.Series(out, index=cond_series.index)

#위와 비슷하지만 조건이 될 값이 두개 이상의 set
#mask_set에 하나라도 해당하면 True로 판단
def masked_mean_in_set_series(cond_series, val_series, mask_set):
    cond_df = cond_series.fillna("").str.split(",", expand=True).replace("", np.nan)
    val_df  = val_series.fillna("").str.split(",", expand=True).replace("", np.nan)

    cond_arr = cond_df.to_numpy(dtype=float)
    val_arr  = val_df.to_numpy(dtype=float)

    mask = np.isin(cond_arr, list(mask_set))
    with np.errstate(invalid="ignore"):
        sums = np.nansum(np.where(mask, val_arr, np.nan), axis=1)
        counts = np.sum(mask, axis=1)
        out = sums / np.where(counts == 0, np.nan, counts)
    return pd.Series(out, index=cond_series.index)

def preprocess_A(trainA):
    df = trainA.copy()

    #Age, TestDate 변환
    df["AgeNum"] = df["Age"].map(convert_age)
    ym = df["TestDate"].map(split_testdate) #(연도, 월)로 반환
    df["Year"] = [y for y, m in ym] #ym을 y,m으로 분리 후 y를 사용
    df["Month"] = [m for y, m in ym]

    feats = pd.DataFrame(index=df.index)

    # ---- A1 ----
    print("Step 2: A1 feature 생성...")
    feats["A1_resp_rate"] = seq_rate(df["A1-3"], "1")   #오반응 비율
    feats["A1_rt_mean"]   = seq_mean(df["A1-4"])    #rt의 평균
    feats["A1_rt_std"]    = seq_std(df["A1-4"])     #rt의 표준편차
    feats["A1_rt_left"]   = masked_mean_from_csv_series(df["A1-1"], df["A1-4"], 1)  #왼쪽일 때 rt 평균
    feats["A1_rt_right"]  = masked_mean_from_csv_series(df["A1-1"], df["A1-4"], 2)  #오른쪽일 때 rt 평균
    feats["A1_rt_side_diff"] = feats["A1_rt_left"] - feats["A1_rt_right"]   #왼쪽 오른쪽 rt 평균 차이
    feats["A1_rt_slow"]   = masked_mean_from_csv_series(df["A1-2"], df["A1-4"], 1)  #느릴 때 rt 평균
    feats["A1_rt_fast"]   = masked_mean_from_csv_series(df["A1-2"], df["A1-4"], 3) #빠를 때 rt 평균
    feats["A1_rt_normal"] = masked_mean_from_csv_series(df["A1-2"], df["A1-4"], 2)  #평범 속도일 때 rt 평균
    feats["A1_rt_speed_diff"] = feats["A1_rt_slow"] - feats["A1_rt_fast"]   #느릴 때와 빠를 때 rt 평균 차이
    feats["A1_resp_except_mean"] = masked_mean_from_csv_series(df["A1-3"], df["A1-4"], 0)   #오반응 제외 rt 평균
    feats["A1_rt_resp_diff"] = feats["A1_rt_mean"] - feats["A1_resp_except_mean"]   #오반응 있을 때와 없을 때 rt 평균 차이

    # ---- A2 ----
    print("Step 3: A2 feature 생성...")
    feats["A2_resp_rate"] = seq_rate(df["A2-3"], "1")   #오반응 비율
    feats["A2_rt_mean"]   = seq_mean(df["A2-4"])    #rt 평균
    feats["A2_rt_std"]    = seq_std(df["A2-4"]) #rt 표준편차
    feats["A2_rt_cond1_diff"] = masked_mean_from_csv_series(df["A2-1"], df["A2-4"], 1) - \
                                masked_mean_from_csv_series(df["A2-1"], df["A2-4"], 3)  #조건1 느릴 때와 빠를 때 rt 평균 차이
    feats["A2_rt_cond2_diff"] = masked_mean_from_csv_series(df["A2-2"], df["A2-4"], 1) - \
                                masked_mean_from_csv_series(df["A2-2"], df["A2-4"], 3)  #조건2 느릴 때와 빠를 때 rt 평균 차이
    #cond1과 cond2 하나는 차 이동속도(가속)이고 하나는 노란색??
    #조건 어떻게 되냐에 따라 더 늘려야 할듯

    # ---- A3 ----
    print("Step 4: A3 feature 생성...")
    s = df["A3-5"].fillna("")
    total   = s.apply(lambda x: len(x.split(",")) if x else 0)
    valid   = s.apply(lambda x: sum(v in {"1","2"} for v in x.split(",")) if x else 0)
    invalid = s.apply(lambda x: sum(v in {"3","4"} for v in x.split(",")) if x else 0)
    correct = s.apply(lambda x: sum(v in {"1","3"} for v in x.split(",")) if x else 0)
    feats["A3_valid_ratio"]   = (valid / total).replace([np.inf,-np.inf], np.nan)
    feats["A3_invalid_ratio"] = (invalid / total).replace([np.inf,-np.inf], np.nan)
    feats["A3_correct_ratio"] = (correct / total).replace([np.inf,-np.inf], np.nan)

    feats["A3_resp2_rate"] = seq_rate(df["A3-6"], "1")
    feats["A3_rt_mean"]    = seq_mean(df["A3-7"])
    feats["A3_rt_std"]     = seq_std(df["A3-7"])
    feats["A3_rt_size_diff"] = masked_mean_from_csv_series(df["A3-1"], df["A3-7"], 1) - \
                               masked_mean_from_csv_series(df["A3-1"], df["A3-7"], 2)
    feats["A3_rt_side_diff"] = masked_mean_from_csv_series(df["A3-3"], df["A3-7"], 1) - \
                               masked_mean_from_csv_series(df["A3-3"], df["A3-7"], 2)

     # ---- A4 ----
    print("Step 5: A4 feature 생성...")
    feats["A4_acc_rate"]   = seq_rate(df["A4-3"], "1")  #맞춘 비율
    feats["A4_resp2_rate"] = seq_rate(df["A4-4"], "1")  #오반응 비율
    feats["A4_rt_mean"]    = seq_mean(df["A4-5"])
    feats["A4_rt_std"]     = seq_std(df["A4-5"])
    feats["A4_stroop_diff"] = masked_mean_from_csv_series(df["A4-1"], df["A4-5"], 2) - \
                              masked_mean_from_csv_series(df["A4-1"], df["A4-5"], 1)
    feats["A4_rt_color_diff"] = masked_mean_from_csv_series(df["A4-2"], df["A4-5"], 1) - \
                                masked_mean_from_csv_series(df["A4-2"], df["A4-5"], 2)
    
    # ---- A5 ----
    print("Step 6: A5 feature 생성...")
    feats["A5_acc_rate"]   = seq_rate(df["A5-2"], "1")
    feats["A5_resp2_rate"] = seq_rate(df["A5-3"], "1")
    feats["A5_acc_nonchange"] = masked_mean_from_csv_series(df["A5-1"], df["A5-2"], 1)
    feats["A5_acc_change"]    = masked_mean_in_set_series(df["A5-1"], df["A5-2"], {2,3,4})

    # ---- Drop ----
    print("Step 7: 시퀀스 컬럼 drop & concat...")
    seq_cols = [
        "A1-1","A1-2","A1-3","A1-4",
        "A2-1","A2-2","A2-3","A2-4",
        "A3-1","A3-2","A3-3","A3-4","A3-5","A3-6","A3-7",
        "A4-1","A4-2","A4-3","A4-4","A4-5",
        "A5-1","A5-2","A5-3"
    ]
    print("A 검사 데이터 전처리 완료")
    return pd.concat([df.drop(columns=seq_cols, errors="ignore"), feats], axis=1)

preTainA = preprocess_A(trainA)
preTainA
preTainA.to_csv("preTiranA.csv", index=False)