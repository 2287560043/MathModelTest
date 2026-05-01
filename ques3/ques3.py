# -*- coding: utf-8 -*-
"""
问题3：去噪补齐 + 异常检测 + 关联建模 + 实验集表面位移预测
"""

import warnings
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, medfilt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# =========================
# 0. 参数设置
# =========================
DATA_FILE = "附件3：监测数据（训练集与实验集）-问题3.xlsx"
OUT_DIR = Path("./ques3")
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TRAIN_RATIO = 0.8

# 默认用线性回归生成实验集预测；如果论文最终写XGBoost，就改成 "XGBoost"
FINAL_MODEL_NAME = "LinearRegression"

VARIABLES = ["降雨量", "孔隙水压力", "微震事件数", "深部位移", "表面位移"]
TARGET = "表面位移"

LETTER_MAP = {
    "降雨量": "a",
    "孔隙水压力": "b",
    "微震事件数": "c",
    "深部位移": "d",
    "表面位移": "e",
}

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = [
    "SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"
]


# =========================
# 1. 工具函数
# =========================
def normalize_columns(df):
    """统一训练集和实验集列名"""
    rename_map = {
        "a:降雨量_mm": "降雨量",
        "b:孔隙水压力_kPa": "孔隙水压力",
        "c:微震事件数": "微震事件数",
        "d:深部位移_mm": "深部位移",
        "e:表面位移_mm": "表面位移",
        "降雨量_mm": "降雨量",
        "孔隙水压力_kPa": "孔隙水压力",
        "深部位移_mm": "深部位移",
        "表面位移_mm": "表面位移",
    }
    return df.rename(columns=rename_map).copy()


def denoise_and_fill(df, cols, smooth_target=True):
    """
    问题3.1：
    1. 线性插值补齐缺失值；
    2. 降雨量、孔隙水压力、深部位移、表面位移：S-G滤波 window=7, poly=3；
    3. 微震事件数：中值滤波 kernel=5。
    """
    out = df.copy()

    for col in cols:
        if col not in out.columns:
            continue

        raw = pd.to_numeric(out[col], errors="coerce")
        interp = raw.interpolate(method="linear", limit_direction="both")

        if col == "微震事件数":
            smooth = medfilt(interp.to_numpy(dtype=float), kernel_size=5)
        else:
            smooth = savgol_filter(
                interp.to_numpy(dtype=float),
                window_length=7,
                polyorder=3,
                mode="interp",
            )

        if col in ["降雨量", "微震事件数"]:
            smooth = np.clip(smooth, 0, None)

        if col == TARGET and not smooth_target:
            out[col] = raw
        else:
            out[col] = smooth

    return out


def make_features(df, has_target=True):
    """
    构造问题3.3建模特征：
    当前值 + 1~3阶滞后项 + 滚动均值 + 降雨强度分类
    """
    d = df.copy()

    for lag in [1, 2, 3]:
        for col in ["降雨量", "微震事件数", "孔隙水压力"]:
            d[f"{col}_lag{lag}"] = d[col].shift(lag)

    d["降雨量_roll3"] = d["降雨量"].rolling(3).mean()
    d["降雨量_roll7"] = d["降雨量"].rolling(7).mean()
    d["微震事件数_roll5"] = d["微震事件数"].rolling(5).mean()

    d["降雨强度"] = pd.cut(
        d["降雨量"],
        bins=[-np.inf, 0, 2, 10, np.inf],
        labels=["无雨", "小雨", "中雨", "大雨"],
    )

    feature_cols = [
        "降雨量", "孔隙水压力", "微震事件数", "深部位移",
        "降雨量_lag1", "微震事件数_lag1", "孔隙水压力_lag1",
        "降雨量_lag2", "微震事件数_lag2", "孔隙水压力_lag2",
        "降雨量_lag3", "微震事件数_lag3", "孔隙水压力_lag3",
        "降雨量_roll3", "降雨量_roll7", "微震事件数_roll5",
        "降雨强度",
    ]

    keep_cols = ["编号"] + feature_cols
    if has_target:
        keep_cols.append(TARGET)

    model_df = d[keep_cols].dropna().copy()
    return model_df, feature_cols


def encode_features(df_model, feature_cols, train_columns=None):
    """降雨强度独热编码，并和训练特征对齐"""
    X = df_model[feature_cols].copy()
    X = pd.get_dummies(X, columns=["降雨强度"], drop_first=True)

    if train_columns is not None:
        for col in train_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[train_columns]

    return X, list(X.columns)


def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((np.asarray(y_true) - y_pred) / np.asarray(y_true))) * 100
    return r2, mae, rmse, mape


def sliding_mad_flags(s, window=24, threshold=3.0, min_periods=12):
    """
    问题3.2：滑动MAD异常检测
    判别准则：
        |x_t - median_t| > threshold * MAD_t
    """
    x = pd.to_numeric(s, errors="coerce")
    med = x.rolling(window=window, center=True, min_periods=min_periods).median()

    def mad_func(arr):
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return np.nan
        m = np.median(arr)
        return np.median(np.abs(arr - m))

    mad = x.rolling(window=window, center=True, min_periods=min_periods).apply(
        mad_func, raw=True
    )

    eps = 1e-9
    flag = (x - med).abs() > threshold * mad.replace(0, eps)
    return flag.fillna(False)


# =========================
# 2. 读取附件3
# =========================
train_raw = pd.read_excel(DATA_FILE, sheet_name="训练集")
exp_raw = pd.read_excel(DATA_FILE, sheet_name="实验集")

train = normalize_columns(train_raw)
exp = normalize_columns(exp_raw)

print("=" * 70)
print("问题3：数据读取完成")
print("=" * 70)
print("训练集：", train.shape)
print("实验集：", exp.shape)
print("\n训练集缺失值：")
print(train[VARIABLES].isna().sum())


# =========================
# 3. 问题3.1 去噪补齐
# =========================
train_clean = denoise_and_fill(train, VARIABLES, smooth_target=True)
exp_clean = denoise_and_fill(
    exp,
    ["降雨量", "孔隙水压力", "微震事件数", "深部位移"],
    smooth_target=False,
)

train_clean[["编号"] + VARIABLES].to_csv(
    OUT_DIR / "附件3_去噪补齐.csv",
    index=False,
    encoding="utf-8-sig",
)

exp_clean.to_csv(
    OUT_DIR / "附件3_实验集_填充后.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n[3.1] 去噪补齐完成")
print("输出：附件3_去噪补齐.csv")

# 去噪对比图
fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
for ax, col in zip(axes, VARIABLES):
    ax.plot(train["编号"], train[col], alpha=0.35, linewidth=0.8, label="原始")
    ax.plot(train_clean["编号"], train_clean[col], linewidth=1.2, label="去噪补齐")
    ax.set_ylabel(col)
    ax.legend(loc="upper right", fontsize=8)

axes[-1].set_xlabel("编号")
fig.suptitle("问题3.1 各变量去噪与缺失补齐效果")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig3_4_denoising_comparison.png", dpi=300)
plt.close(fig)


# 相关性热力图
corr = train_clean[VARIABLES].corr()
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr, vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.index)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.index)

for i in range(len(corr.index)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)

fig.colorbar(im, ax=ax)
ax.set_title("附件3变量相关性热力图")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig3_3_correlation.png", dpi=300)
plt.close(fig)


# =========================
# 4. 问题3.2 异常值检测
# =========================
flags = {}
rows_31 = []

for_col = VARIABLES
for col in VARIABLES:
    flag = sliding_mad_flags(train_clean[col], window=24, threshold=3.0, min_periods=12)
    letter = LETTER_MAP[col]
    flags[letter] = flag.to_numpy(dtype=bool)

    rows_31.append({
        "变量": f"{letter}.{col}",
        "异常点数": int(flag.sum()),
        "方法": "滑动MAD(24h,±3)",
    })

table31 = pd.DataFrame(rows_31)
table31.loc[len(table31)] = ["合计", int(table31["异常点数"].sum()), ""]
table31.to_csv(
    OUT_DIR / "表3.1_单变量异常检测结果.csv",
    index=False,
    encoding="utf-8-sig",
)

flag_df = pd.DataFrame(flags)
common_mask = flag_df.sum(axis=1) >= 2

rows_32 = []
for idx in np.where(common_mask.to_numpy())[0]:
    abnormal_vars = "".join(
        [letter for letter in ["a", "b", "c", "d", "e"] if flag_df.loc[idx, letter]]
    )
    rows_32.append({
        "时间点对应编号": int(train_clean.loc[idx, "编号"]),
        "共同异常点处的异常变量": abnormal_vars,
    })

table32 = pd.DataFrame(rows_32)
table32.to_csv(
    OUT_DIR / "表3.2_多变量共同异常点清单.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n[3.2] 异常检测完成")
print(table31.to_string(index=False))
print("共同异常点数量：", len(table32))


# =========================
# 5. 问题3.3 建模
# =========================
model_df, feature_cols = make_features(train_clean, has_target=True)

X, feature_names = encode_features(model_df, feature_cols)
y = model_df[TARGET].copy()

split = int(len(X) * TRAIN_RATIO)

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "Lasso": Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=10000),
    "RandomForest": RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}

try:
    from xgboost import XGBRegressor

    models["XGBoost"] = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
except Exception:
    print("未安装xgboost，跳过XGBoost。需要时可运行：pip install xgboost")

results = []
fitted = {}

for name, model in models.items():
    t0 = perf_counter()

    if name in ["LinearRegression", "Ridge", "Lasso"]:
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

    r2, mae, rmse, mape = regression_metrics(y_test, pred)

    results.append({
        "模型": name,
        "R²": r2,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE(%)": mape,
        "训练时间(s)": perf_counter() - t0,
    })

    fitted[name] = model

results_df = pd.DataFrame(results).sort_values("R²", ascending=False)
results_df.to_csv(
    OUT_DIR / "问题3_模型对比.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n[3.3] 模型训练完成")
print("特征工程后样本数：", len(X))
print("特征数：", len(feature_names))
print("训练/测试：", len(X_train), "/", len(X_test))
print(results_df.to_string(index=False))

# 保存模型
joblib.dump(fitted["LinearRegression"], OUT_DIR / "model_linear_regression.pkl")
if "XGBoost" in fitted:
    joblib.dump(fitted["XGBoost"], OUT_DIR / "model_xgboost.pkl")

joblib.dump(scaler, OUT_DIR / "scaler.pkl")
joblib.dump(feature_names, OUT_DIR / "feature_names.pkl")


# 测试集预测散点图
plot_model_name = FINAL_MODEL_NAME if FINAL_MODEL_NAME in fitted else results_df.iloc[0]["模型"]
plot_model = fitted[plot_model_name]

if plot_model_name in ["LinearRegression", "Ridge", "Lasso"]:
    y_pred_test = plot_model.predict(X_test_scaled)
else:
    y_pred_test = plot_model.predict(X_test)

r2, mae, rmse, mape = regression_metrics(y_test, y_pred_test)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, y_pred_test, s=12, alpha=0.55)

lo = min(y_test.min(), y_pred_test.min())
hi = max(y_test.max(), y_pred_test.max())
ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)

ax.set_xlabel("真实表面位移 / mm")
ax.set_ylabel("预测表面位移 / mm")
ax.set_title(f"{plot_model_name}预测值 vs 真实值\nR²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

fig.tight_layout()
fig.savefig(OUT_DIR / "fig3_6_prediction_vs_actual.png", dpi=300)
plt.close(fig)


# 线性模型特征贡献
lr = fitted["LinearRegression"]

coef_df = pd.DataFrame({
    "特征": feature_names,
    "标准化系数": lr.coef_,
    "贡献度_abs": np.abs(lr.coef_),
}).sort_values("贡献度_abs", ascending=False)

coef_df.to_csv(
    OUT_DIR / "问题3_线性模型特征贡献.csv",
    index=False,
    encoding="utf-8-sig",
)

fig, ax = plt.subplots(figsize=(9, 5))
top_coef = coef_df.head(12).iloc[::-1]
ax.barh(top_coef["特征"], top_coef["贡献度_abs"])
ax.set_xlabel("|标准化系数|")
ax.set_title("表面位移影响因素贡献度 Top12")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig3_7_feature_importance.png", dpi=300)
plt.close(fig)


# =========================
# 6. 实验集预测
# =========================
exp_model_df, _ = make_features(exp_clean, has_target=False)
X_exp, _ = encode_features(exp_model_df, feature_cols, train_columns=feature_names)

final_model = fitted[FINAL_MODEL_NAME] if FINAL_MODEL_NAME in fitted else fitted[results_df.iloc[0]["模型"]]

if FINAL_MODEL_NAME in ["LinearRegression", "Ridge", "Lasso"]:
    y_exp_pred = final_model.predict(scaler.transform(X_exp))
else:
    y_exp_pred = final_model.predict(X_exp)

pred_df = pd.DataFrame({
    "样本序号": exp_model_df["编号"].astype(int).to_numpy(),
    "表面位移_预测值": y_exp_pred,
})

pred_df.to_csv(
    OUT_DIR / "实验集_表面位移预测.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n实验集预测完成")
print("预测样本数：", len(pred_df))
print("预测范围：[{:.2f}, {:.2f}] mm".format(
    pred_df["表面位移_预测值"].min(),
    pred_df["表面位移_预测值"].max(),
))
print("预测均值：{:.2f} mm".format(pred_df["表面位移_预测值"].mean()))


# 预测时序图
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(pred_df["样本序号"], pred_df["表面位移_预测值"], linewidth=1)
ax.set_xlabel("样本序号")
ax.set_ylabel("预测表面位移 / mm")
ax.set_title("实验集表面位移预测时序")
fig.tight_layout()
fig.savefig(OUT_DIR / "problem3_exp_prediction_timeseries.png", dpi=300)
plt.close(fig)


# 预测分布图
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(pred_df["表面位移_预测值"], bins=40, alpha=0.75)
ax.set_xlabel("预测表面位移 / mm")
ax.set_ylabel("频数")
ax.set_title("实验集表面位移预测值分布")
fig.tight_layout()
fig.savefig(OUT_DIR / "problem3_exp_prediction_hist.png", dpi=300)
plt.close(fig)


# 预测表面位移与关键因素散点图
merged = exp_model_df[["编号", "降雨量", "孔隙水压力", "微震事件数", "深部位移"]].copy()
merged["表面位移_预测值"] = y_exp_pred

scatter_specs = [
    ("深部位移", "problem3_exp_scatter_deep.png"),
    ("孔隙水压力", "problem3_exp_scatter_pore.png"),
    ("降雨量", "problem3_exp_scatter_rain.png"),
]

for x_col, file_name in scatter_specs:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(merged[x_col], merged["表面位移_预测值"], s=12, alpha=0.55)

    x = merged[x_col].to_numpy(dtype=float)
    y_plot = merged["表面位移_预测值"].to_numpy(dtype=float)

    if np.nanstd(x) > 0:
        k, b = np.polyfit(x, y_plot, 1)
        x_fit = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        y_fit = k * x_fit + b
        r = np.corrcoef(x, y_plot)[0, 1]

        ax.plot(x_fit, y_fit, linestyle="--", linewidth=1)
        ax.set_title(f"预测表面位移 vs {x_col}，r={r:.3f}")
    else:
        ax.set_title(f"预测表面位移 vs {x_col}")

    ax.set_xlabel(x_col)
    ax.set_ylabel("预测表面位移 / mm")
    fig.tight_layout()
    fig.savefig(OUT_DIR / file_name, dpi=300)
    plt.close(fig)


print("\n全部完成！输出目录：", OUT_DIR.resolve())