import os, argparse, joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
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

def _to_array(x):
    if not x: return None
    arr = np.fromstring(x, sep=",", dtype=float)
    return arr if arr.size else None

def seq_mean(series):   #series의 평균값 반환 메소드
    return series.fillna("").progress_apply(
        lambda x: np.fromstring(x, sep=",").mean() if x else np.nan
    )   #series를 숫자 배열로 변환하여 평균값 반환, progress_apply는 진행률 표시

def seq_std(series):    #series의 표준편차 반환 메소드
    return series.fillna("").progress_apply(
        lambda x: np.fromstring(x, sep=",").std() if x else np.nan
    )

def seq_median(series: pd.Series):
    return series.fillna("").progress_apply(
        lambda x: (np.nanmedian(_to_array(x)) if _to_array(x) is not None else np.nan)
    )

def seq_quantile(series: pd.Series, q: float):
    return series.fillna("").progress_apply(
        lambda x: (np.nanpercentile(_to_array(x), q*100) if _to_array(x) is not None else np.nan)
    )

def seq_iqr(series: pd.Series):  #이거 쓰나?
    return seq_quantile(series, 0.75) - seq_quantile(series, 0.25)

def seq_mode(series: pd.Series):
    def _mode(x):
        if not x: return np.nan
        items = str(x).split(",")
        if not items: return np.nan
        cnt = Counter(items)
        maxc = max(cnt.values())
        cands = [k for k,v in cnt.items() if v==maxc]
        try:
            cands_num = [float(c) for c in cands]
            return min(cands_num)
        except:
            return min(cands)
    return series.fillna("").progress_apply(_mode)

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

def masked_quantile_from_csv_series(cond_series, val_series, mask_vals, q: float):
    cond_df = cond_series.fillna("").str.split(",", expand=True).replace("", np.nan)
    val_df  = val_series.fillna("").str.split(",", expand=True).replace("", np.nan)
    cond_arr = cond_df.to_numpy(dtype=float)
    val_arr  = val_df.to_numpy(dtype=float)

    if isinstance(mask_vals, (set, list, tuple)):
        mask = np.isin(cond_arr, list(mask_vals))
    else:
        mask = (cond_arr == float(mask_vals))

    masked_vals = np.where(mask, val_arr, np.nan)
    with np.errstate(all="ignore"):
        out = np.nanpercentile(masked_vals, q*100, axis=1)
    return pd.Series(out, index=cond_series.index)

def preprocess_A(trainA):
    df = trainA.copy()

    #Age, TestDate 변환
    df["AgeNum"] = df["Age"].map(convert_age)
    # ym = df["TestDate"].map(split_testdate) #(연도, 월)로 반환
    # df["Year"] = [y for y, m in ym] #ym을 y,m으로 분리 후 y를 사용
    # df["Month"] = [m for y, m in ym]

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
    feats["A1_rt_median"] = seq_median(df["A1-4"])
    feats["A1_rt_q25"]    = seq_quantile(df["A1-4"], 0.25)
    feats["A1_rt_q75"]    = seq_quantile(df["A1-4"], 0.75)
    feats["A1_rt_iqr"]    = feats["A1_rt_q75"] - feats["A1_rt_q25"]\
    # ---중앙값 통계값들 필요 유무 검토 필요---
    feats["A1_rt_left_med"]  = masked_quantile_from_csv_series(df["A1-1"], df["A1-4"], 1, 0.5)
    feats["A1_rt_right_med"] = masked_quantile_from_csv_series(df["A1-1"], df["A1-4"], 2, 0.5)
    feats["A1_rt_side_med_diff"] = feats["A1_rt_left_med"] - feats["A1_rt_right_med"]
    feats["A1_rt_slow_med"]  = masked_quantile_from_csv_series(df["A1-2"], df["A1-4"], 1, 0.5)
    feats["A1_rt_fast_med"]  = masked_quantile_from_csv_series(df["A1-2"], df["A1-4"], 3, 0.5)
    feats["A1_rt_speed_med_diff"] = feats["A1_rt_slow_med"] - feats["A1_rt_fast_med"]

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
    feats["A2_rt_median"] = seq_median(df["A2-4"])
    feats["A2_rt_q25"]    = seq_quantile(df["A2-4"], 0.25)
    feats["A2_rt_q75"]    = seq_quantile(df["A2-4"], 0.75)
    feats["A2_rt_iqr"]    = feats["A2_rt_q75"] - feats["A2_rt_q25"]
    # ---중앙값 접근 검토필요---
    c1_1_med = masked_quantile_from_csv_series(df["A2-1"], df["A2-4"], 1, 0.5)
    c1_3_med = masked_quantile_from_csv_series(df["A2-1"], df["A2-4"], 3, 0.5)
    feats["A2_rt_cond1_med_diff"] = c1_1_med - c1_3_med
    c2_1_med = masked_quantile_from_csv_series(df["A2-2"], df["A2-4"], 1, 0.5)
    c2_3_med = masked_quantile_from_csv_series(df["A2-2"], df["A2-4"], 3, 0.5)
    feats["A2_rt_cond2_med_diff"] = c2_1_med - c2_3_med

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
    
    feats["A3_rt_median"] = seq_median(df["A3-7"])
    feats["A3_rt_q25"]    = seq_quantile(df["A3-7"], 0.25)
    feats["A3_rt_q75"]    = seq_quantile(df["A3-7"], 0.75)
    feats["A3_rt_iqr"]    = feats["A3_rt_q75"] - feats["A3_rt_q25"]
    # --- 중앙값 접근 검토 필요---
    sz1_med = masked_quantile_from_csv_series(df["A3-1"], df["A3-7"], 1, 0.5)
    sz2_med = masked_quantile_from_csv_series(df["A3-1"], df["A3-7"], 2, 0.5)
    feats["A3_rt_size_med_diff"] = sz1_med - sz2_med
    side1_med = masked_quantile_from_csv_series(df["A3-3"], df["A3-7"], 1, 0.5)
    side2_med = masked_quantile_from_csv_series(df["A3-3"], df["A3-7"], 2, 0.5)
    feats["A3_rt_side_med_diff"] = side1_med - side2_med

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
    feats["A4_rt_median"] = seq_median(df["A4-5"])
    feats["A4_rt_q25"]    = seq_quantile(df["A4-5"], 0.25)
    feats["A4_rt_q75"]    = seq_quantile(df["A4-5"], 0.75)
    feats["A4_rt_iqr"]    = feats["A4_rt_q75"] - feats["A4_rt_q25"]
    # ---중앙값 접근 검토 필요---
    con_med  = masked_quantile_from_csv_series(df["A4-1"], df["A4-5"], 1, 0.5)
    incon_med= masked_quantile_from_csv_series(df["A4-1"], df["A4-5"], 2, 0.5)
    feats["A4_stroop_med_diff"] = incon_med - con_med
    red_med  = masked_quantile_from_csv_series(df["A4-2"], df["A4-5"], 1, 0.5)
    green_med= masked_quantile_from_csv_series(df["A4-2"], df["A4-5"], 2, 0.5)
    feats["A4_rt_color_med_diff"] = red_med - green_med
    
    # ---- A5 ----
    print("Step 6: A5 feature 생성...")
    feats["A5_acc_rate"]   = seq_rate(df["A5-2"], "1")
    feats["A5_resp2_rate"] = seq_rate(df["A5-3"], "1")
    feats["A5_acc_nonchange"] = masked_mean_from_csv_series(df["A5-1"], df["A5-2"], 1)
    feats["A5_acc_change"]    = masked_mean_in_set_series(df["A5-1"], df["A5-2"], {2,3,4})
    # ---최빈값 접근 검토 필요---
    feats["A1_resp_mode"] = seq_mode(df["A1-3"])
    feats["A3_resp1_mode"] = seq_mode(df["A3-5"])
    feats["A4_resp1_mode"] = seq_mode(df["A4-3"])

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

def preprocess_B(train_B):
    df = train_B.copy()
    print("Step 1: Age, TestDate 파생...")
    df["Age_num"] = df["Age"].map(convert_age)
    # ym = df["TestDate"].map(split_testdate)
    # df["Year"] = [y for y, m in ym]
    # df["Month"] = [m for y, m in ym]

    feats = pd.DataFrame(index=df.index)

    print("Step 2: B1 feature 생성...")
    feats["B1_acc_task1"] = seq_rate(df["B1-1"], "1")
    feats["B1_rt_mean"]   = seq_mean(df["B1-2"])
    feats["B1_rt_std"]    = seq_std(df["B1-2"])
    feats["B1_acc_task2"] = seq_rate(df["B1-3"], "1")
    feats["B1_rt_median"] = seq_median(df["B1-2"])
    feats["B1_rt_q25"]    = seq_quantile(df["B1-2"], 0.25)
    feats["B1_rt_q75"]    = seq_quantile(df["B1-2"], 0.75)
    feats["B1_rt_iqr"]    = feats["B1_rt_q75"] - feats["B1_rt_q25"]
    feats["B1_resp2_mode"] = seq_mode(df["B1-3"])

    print("Step 3: B2 feature 생성...")
    feats["B2_acc_task1"] = seq_rate(df["B2-1"], "1")
    feats["B2_rt_mean"]   = seq_mean(df["B2-2"])
    feats["B2_rt_std"]    = seq_std(df["B2-2"])
    feats["B2_acc_task2"] = seq_rate(df["B2-3"], "1")

    print("Step 4: B3 feature 생성...")
    feats["B3_acc_rate"] = seq_rate(df["B3-1"], "1")
    feats["B3_rt_mean"]  = seq_mean(df["B3-2"])
    feats["B3_rt_std"]   = seq_std(df["B3-2"])

    print("Step 5: B4 feature 생성...")
    feats["B4_acc_rate"] = seq_rate(df["B4-1"], "1")
    feats["B4_rt_mean"]  = seq_mean(df["B4-2"])
    feats["B4_rt_std"]   = seq_std(df["B4-2"])

    print("Step 6: B5 feature 생성...")
    feats["B5_acc_rate"] = seq_rate(df["B5-1"], "1")
    feats["B5_rt_mean"]  = seq_mean(df["B5-2"])
    feats["B5_rt_std"]   = seq_std(df["B5-2"])

    print("Step 7: B6~B8 feature 생성...")
    feats["B6_acc_rate"] = seq_rate(df["B6"], "1")
    feats["B7_acc_rate"] = seq_rate(df["B7"], "1")
    feats["B8_acc_rate"] = seq_rate(df["B8"], "1")
    for k in ["B2","B3","B4","B5"]:
        rt = f"{k}-2"
        feats[f"{k}_rt_median"] = seq_median(df[rt])
        feats[f"{k}_rt_q25"]    = seq_quantile(df[rt], 0.25)
        feats[f"{k}_rt_q75"]    = seq_quantile(df[rt], 0.75)
        feats[f"{k}_rt_iqr"]    = feats[f"{k}_rt_q75"] - feats[f"{k}_rt_q25"]

    print("Step 8: 시퀀스 컬럼 drop & concat...")
    seq_cols = [
        "B1-1","B1-2","B1-3",
        "B2-1","B2-2","B2-3",
        "B3-1","B3-2",
        "B4-1","B4-2",
        "B5-1","B5-2",
        "B6","B7","B8"
    ]
    print("B 검사 데이터 전처리 완료")
    out = pd.concat([df.drop(columns=seq_cols, errors="ignore"), feats], axis=1)
    out.replace([np.inf,-np.inf], np.nan, inplace=True)
    return out

# =======================
# 학습 때 사용한 파생 (그대로)
# =======================
def _has(df, cols):  return all(c in df.columns for c in cols)
def _safe_div(a, b, eps=1e-6): return a / (b + eps)

def add_features_A(df: pd.DataFrame) -> pd.DataFrame:
    feats = df.copy(); eps = 1e-6
    if _has(feats, ["Year","Month"]):
        feats["YearMonthIndex"] = feats["Year"] * 12 + feats["Month"]

    if _has(feats, ["A1_rt_mean","A1_resp_rate"]):
        feats["A1_speed_acc_tradeoff"] = _safe_div(feats["A1_rt_mean"], feats["A1_resp_rate"], eps)
    if _has(feats, ["A2_rt_mean","A2_resp_rate"]):
        feats["A2_speed_acc_tradeoff"] = _safe_div(feats["A2_rt_mean"], feats["A2_resp_rate"], eps)
    if _has(feats, ["A4_rt_mean","A4_acc_rate"]):
        feats["A4_speed_acc_tradeoff"] = _safe_div(feats["A4_rt_mean"], feats["A4_acc_rate"], eps)

    for k in ["A1","A2","A3","A4"]:
        m, s = f"{k}_rt_mean", f"{k}_rt_std"
        if _has(feats, [m, s]):
            feats[f"{k}_rt_cv"] = _safe_div(feats[s], feats[m], eps)

    for name, base in [
        ("A1_rt_side_gap_abs",  "A1_rt_side_diff"),
        ("A1_rt_speed_gap_abs", "A1_rt_speed_diff"),
        ("A2_rt_cond1_gap_abs", "A2_rt_cond1_diff"),
        ("A2_rt_cond2_gap_abs", "A2_rt_cond2_diff"),
        ("A4_stroop_gap_abs",   "A4_stroop_diff"),
        ("A4_color_gap_abs",    "A4_rt_color_diff"),
    ]:
        if base in feats.columns:
            feats[name] = feats[base].abs()

    if _has(feats, ["A3_valid_ratio","A3_invalid_ratio"]):
        feats["A3_valid_invalid_gap"] = feats["A3_valid_ratio"] - feats["A3_invalid_ratio"]
    if _has(feats, ["A3_correct_ratio","A3_invalid_ratio"]):
        feats["A3_correct_invalid_gap"] = feats["A3_correct_ratio"] - feats["A3_invalid_ratio"]
    if _has(feats, ["A5_acc_change","A5_acc_nonchange"]):
        feats["A5_change_nonchange_gap"] = feats["A5_acc_change"] - feats["A5_acc_nonchange"]

    if _has(feats, ["A1_rt_median","A1_resp_rate"]):
        feats["A1_speed_acc_tradeoff_med"] = _safe_div(feats["A1_rt_median"], feats["A1_resp_rate"])
    if _has(feats, ["A2_rt_median","A2_resp_rate"]):
        feats["A2_speed_acc_tradeoff_med"] = _safe_div(feats["A2_rt_median"], feats["A2_resp_rate"])
    if _has(feats, ["A4_rt_median","A4_acc_rate"]):
        feats["A4_speed_acc_tradeoff_med"] = _safe_div(feats["A4_rt_median"], feats["A4_acc_rate"])

    for k in ["A1","A2","A3","A4"]:
        iqr, med = f"{k}_rt_iqr", f"{k}_rt_median"
        if _has(feats, [iqr, med]):
            feats[f"{k}_rt_rcv"] = _safe_div(feats[iqr], feats[med])

    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats

def add_features_B(df: pd.DataFrame) -> pd.DataFrame:
    feats = df.copy(); eps = 1e-6
    if _has(feats, ["Year","Month"]):
        feats["YearMonthIndex"] = feats["Year"] * 12 + feats["Month"]

    for k, acc_col, rt_col in [
        ("B1", "B1_acc_task1", "B1_rt_mean"),
        ("B2", "B2_acc_task1", "B2_rt_mean"),
        ("B3", "B3_acc_rate",  "B3_rt_mean"),
        ("B4", "B4_acc_rate",  "B4_rt_mean"),
        ("B5", "B5_acc_rate",  "B5_rt_mean"),
    ]:
        if _has(feats, [rt_col, acc_col]):
            feats[f"{k}_speed_acc_tradeoff"] = _safe_div(feats[rt_col], feats[acc_col], eps)

    for k in ["B1","B2","B3","B4","B5"]:
        m, s = f"{k}_rt_mean", f"{k}_rt_std"
        if _has(feats, [m, s]):
            feats[f"{k}_rt_cv"] = _safe_div(feats[s], feats[m], eps)

    parts = []
    for k in ["B4","B5"]:
        if _has(feats, [f"{k}_rt_cv"]):
            parts.append(0.25 * feats[f"{k}_rt_cv"].fillna(0))
    for k in ["B3","B4","B5"]:
        acc = f"{k}_acc_rate" if k not in ["B1","B2"] else None
        if k in ["B1","B2"]:
            acc = f"{k}_acc_task1"
        if acc in feats:
            parts.append(0.25 * (1 - feats[acc].fillna(0)))
    for k in ["B1","B2"]:
        tcol = f"{k}_speed_acc_tradeoff"
        if tcol in feats:
            parts.append(0.25 * feats[tcol].fillna(0))
    for k, acc_col in [("B1","B1_acc_task1"), ("B2","B2_acc_task1"),
                       ("B3","B3_acc_rate"),  ("B4","B4_acc_rate"), ("B5","B5_acc_rate")]:
        med = f"{k}_rt_median"
        if _has(feats, [med, acc_col]):
            feats[f"{k}_speed_acc_tradeoff_med"] = _safe_div(feats[med], feats[acc_col])

    for k in ["B1","B2","B3","B4","B5"]:
        iqr, med = f"{k}_rt_iqr", f"{k}_rt_median"
        if _has(feats, [iqr, med]):
            feats[f"{k}_rt_rcv"] = _safe_div(feats[iqr], feats[med])

    if parts:
        feats["RiskScore_B"] = sum(parts)

    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats

# =======================
# 정렬/보정 (모델이 학습 때 본 피처 순서로)
# =======================
DROP_COLS = ["Test_id","Test","PrimaryKey","Age","TestDate", "YearMonthIndex", "Year", "Month"]

def align_to_model(X_df, model):
    # Booster / sklearn wrapper 호환적으로 feature name 추출
    feat_names = []
    try:
        if hasattr(model, "feature_name") and callable(model.feature_name):
            fn = model.feature_name()
            if fn is not None:
                feat_names = list(fn)
    except Exception:
        pass
    if (not feat_names) and hasattr(model, "feature_name_"):
        feat_names = list(getattr(model, "feature_name_", []) or [])

    if not feat_names:
        X = X_df.select_dtypes(include=[np.number]).copy()
        return X.fillna(0.0)

    X = X_df.drop(columns=[c for c in DROP_COLS if c in X_df.columns], errors="ignore").copy()
    for c in feat_names:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feat_names]
    return X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

# =======================
# 예측 래퍼 (Booster/Sklearn 공통)
# =======================
def _predict_proba_binary(model, X):
    if hasattr(model, "predict_proba"):   # sklearn API
        p = model.predict_proba(X)
        return p[:,1] if p.ndim == 2 else p
    # LightGBM Booster
    return model.predict(X, num_iteration=getattr(model, "best_iteration", None))

def main():
    # ---- 경로 변수 (필요에 따라 수정) ----
    TEST_DIR  = "./data"      # test.csv, test/A.csv, test/B.csv, sample_submission.csv 위치
    MODEL_DIR = "./model"     # lgbm_A.pkl, lgbm_B.pkl 위치
    OUT_DIR   = "./output"
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    OUT_PATH  = os.path.join(OUT_DIR, "submission.csv")

    # ---- 모델 로드 ----
    print("Load models...")
    model_A = joblib.load(os.path.join(MODEL_DIR, "lgbm_tdrop_A.pkl"))
    model_B = joblib.load(os.path.join(MODEL_DIR, "lgbm_tdrop_B.pkl"))
    print(" OK.")

    # ---- 테스트 데이터 로드 ----
    print("Load test data...")
    meta = pd.read_csv(os.path.join(TEST_DIR, "test.csv"))
    Araw = pd.read_csv(os.path.join(TEST_DIR, "test", "A.csv"))
    Braw = pd.read_csv(os.path.join(TEST_DIR, "test", "B.csv"))
    print(f" meta={len(meta)}, Araw={len(Araw)}, Braw={len(Braw)}")

    # ---- 매핑 ----
    A_df = meta.loc[meta["Test"] == "A", ["Test_id", "Test"]].merge(Araw, on="Test_id", how="left")
    B_df = meta.loc[meta["Test"] == "B", ["Test_id", "Test"]].merge(Braw, on="Test_id", how="left")
    print(f" mapped: A={len(A_df)}, B={len(B_df)}")

    # ---- 전처리 → 파생 (학습과 동일) ----
    A_feat = add_features_A(preprocess_A(A_df)) if len(A_df) else pd.DataFrame()
    B_feat = add_features_B(preprocess_B(B_df)) if len(B_df) else pd.DataFrame()

    # ---- 피처 정렬/보정 ----
    XA = align_to_model(A_feat, model_A) if len(A_feat) else pd.DataFrame()
    XB = align_to_model(B_feat, model_B) if len(B_feat) else pd.DataFrame()
    print(f" aligned: XA={XA.shape}, XB={XB.shape}")
    print(XA.columns)

    # ---- 예측 ----
    print("Inference Model...")
    predA = _predict_proba_binary(model_A, XA) if len(XA) else np.array([])
    predB = _predict_proba_binary(model_B, XB) if len(XB) else np.array([])

    # ---- Test_id와 합치기 ----
    subA = pd.DataFrame({"Test_id": A_df["Test_id"].values, "prob": predA})
    subB = pd.DataFrame({"Test_id": B_df["Test_id"].values, "prob": predB})
    probs = pd.concat([subA, subB], axis=0, ignore_index=True)

    # ---- sample_submission 기반 결과 생성 ----
    os.makedirs(OUT_DIR, exist_ok=True)
    sample = pd.read_csv(SAMPLE_SUB_PATH)
    out = sample.merge(probs, on="Test_id", how="left")
    out["Label"] = out["prob"].astype(float).fillna(0.0)
    out = out.drop(columns=["prob"])

    out.to_csv(OUT_PATH, index=False)
    print(f"✅ Saved: {OUT_PATH} (rows={len(out)})")

if __name__ == "__main__":
    main()

