import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import itertools
import jenkspy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# 0. 环境设置与路径
# ==========================================
output_dir = 'Results_Great/Hazard_Unsupervised2'
os.makedirs(output_dir, exist_ok=True)

# 字体设置
LABEL_FONTSIZE = 26
TITLE_FONTSIZE = 28
TICK_FONTSIZE = 22
ANNOT_FONTSIZE = 18

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 核心计算函数
# ==========================================

def unsupervised_classification(df_sub, h_vars):
    scaler = StandardScaler()
    X = df_sub[h_vars].fillna(0)
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    df_temp = df_sub.copy()
    df_temp['cluster'] = labels
    norm_h = (df_sub[h_vars] - df_sub[h_vars].min()) / (df_sub[h_vars].max() - df_sub[h_vars].min())
    df_temp['h_score'] = norm_h.mean(axis=1).fillna(0)
    
    cluster_scores = df_temp.groupby('cluster')['h_score'].mean().sort_values()
    rank_map = {cluster: rank + 1 for rank, cluster in enumerate(cluster_scores.index)}
    return df_temp['cluster'].map(rank_map)

def adaptive_discretization(series, col_name, n_bins=5):
    if col_name in ['CL', 'SA', 'Altitude']: return series
    n_unique = series.nunique()
    if n_unique < 2: return series
    try:
        breaks = jenkspy.jenks_breaks(series.dropna().values, n_classes=min(n_bins, n_unique))
        return pd.cut(series, bins=breaks, labels=False, include_lowest=True)
    except:
        return pd.qcut(series.rank(method='first'), q=min(n_bins, n_unique), labels=False, duplicates='drop')

def factor_detector(df, feature_cols, target_col):
    results = {}
    y = df[target_col]
    sst = y.var(ddof=0) * len(y)
    if sst == 0: return {col: 0.0 for col in feature_cols}
    for col in feature_cols:
        grouped = df.groupby(col)[target_col]
        ssw = sum(group.var(ddof=0) * len(group) for _, group in grouped)
        results[col] = max(0.0, 1.0 - (ssw / sst))
    return results

def get_interaction_type(q1, q2, q12):
    if q12 < min(q1, q2): return "非线性削弱"
    if min(q1, q2) < q12 < max(q1, q2): return "单因子削弱"
    if q12 > (q1 + q2): return "非线性增强"
    if q12 > max(q1, q2): return "双因子增强"
    return "独立"

# ==========================================
# 2. 数据处理与地理探测器计算
# ==========================================
file_path = 'RiskHEV8700.xlsx'
df_raw = pd.read_excel(file_path)

h_vars = ['EQ', 'LS', 'WE', 'WAE', 'FT', 'SA', 'NDVI', 'HRF', 'CL', 'RND', 'GI']
ele_map = {1: 'ELER', 2: 'LER', 3: 'MER', 4: 'HER', 5: 'EHER'}
target = 'Hazard_Class'

df_raw[target] = 0
for val, name in ele_map.items():
    mask = df_raw['Altitude'] == val
    if mask.sum() >= 10:
        df_raw.loc[mask, target] = unsupervised_classification(df_raw[mask], h_vars).values

q_summary_list = []
interaction_list = []

# --- 因子排名绘图 ---
fig, axes = plt.subplots(2, 3, figsize=(22, 14)) # 略微增加宽度
axes = axes.flatten()

for i, (ele_val, ele_name) in enumerate(ele_map.items()):
    df_sub = df_raw[df_raw['Altitude'] == ele_val].copy()
    if len(df_sub) < 10: continue

    df_dis = df_sub.copy()
    for col in h_vars:
        df_dis[col] = adaptive_discretization(df_sub[col], col)

    current_q = factor_detector(df_dis, h_vars, target)
    rank_df = pd.DataFrame(list(current_q.items()), columns=['Feature', 'q']).sort_values('q', ascending=False)
    
    sns.barplot(x='q', y='Feature', data=rank_df, ax=axes[i], palette='magma')
    axes[i].set_title(f'Elevation: {ele_name}', fontsize=TITLE_FONTSIZE)
    axes[i].tick_params(labelsize=TICK_FONTSIZE)
    axes[i].set_xlabel('q value', fontsize=LABEL_FONTSIZE)
    axes[i].set_ylabel('', fontsize=LABEL_FONTSIZE)

    # --- 修正点 1: 拓宽 X 轴范围，防止数值标注溢出 ---
    max_q = rank_df['q'].max()
    axes[i].set_xlim(0, max_q * 1.3) 

    for p in axes[i].patches:
        width = p.get_width()
        axes[i].text(width + (max_q * 0.02), p.get_y() + p.get_height()/2, 
                     f'{width:.3f}', va='center', fontsize=ANNOT_FONTSIZE, fontweight='bold')

    q_dict = current_q.copy()
    q_dict['Elevation'] = ele_name
    q_summary_list.append(q_dict)

    top3 = rank_df['Feature'].head(3).tolist()
    sst_sub = df_dis[target].var(ddof=0) * len(df_dis)
    for p1, p2 in itertools.combinations(top3, 2):
        df_dis['inter'] = df_dis[p1].astype(str) + "_" + df_dis[p2].astype(str)
        ssw_i = sum(g[target].var(ddof=0)*len(g) for _, g in df_dis.groupby('inter'))
        q_inter = 1 - (ssw_i / sst_sub if sst_sub != 0 else 0)
        interaction_list.append({
            'Elevation': ele_name, 'Factor_A': p1, 'Factor_B': p2,
            'q_A': q_dict[p1], 'q_B': q_dict[p2], 'q_Inter': q_inter,
            'Type': get_interaction_type(q_dict[p1], q_dict[p2], q_inter)
        })

for j in range(i + 1, len(axes)): fig.delaxes(axes[j])
plt.tight_layout()
plt.savefig(f'{output_dir}/factor_ranking_by_elevation.png', dpi=300)

# ==========================================
# 3. 汇总表与交互热力图输出
# ==========================================
pd.DataFrame(q_summary_list).to_excel(f'{output_dir}/geodetector_q_summary.xlsx', index=False)
inter_df = pd.DataFrame(interaction_list)
inter_df.to_excel(f'{output_dir}/interaction_results.xlsx', index=False)

# --- 交互热力图绘图 ---
if not inter_df.empty:
    plt.figure(figsize=(18, 12)) # 调大画布
    pivot_inter = inter_df.pivot_table(index="Elevation", columns=["Factor_A", "Factor_B"], values="q_Inter")
    pivot_inter = pivot_inter.reindex([v for v in ele_map.values() if v in pivot_inter.index])
    
    sns.heatmap(pivot_inter, annot=True, cmap="YlOrRd", fmt=".3f", 
                annot_kws={'size': ANNOT_FONTSIZE, 'weight': 'bold'})
    
    plt.title('Interaction Detector Summary (q_Inter)', fontsize=TITLE_FONTSIZE, pad=30)
    plt.tick_params(labelsize=TICK_FONTSIZE)
    plt.ylabel('Elevation', fontsize=LABEL_FONTSIZE)
    plt.xlabel('Factor Pairs', fontsize=LABEL_FONTSIZE)

    # --- 修正点 2: X 轴标签斜放 45 度，防止重叠 ---
    plt.xticks(rotation=45, ha='right', fontsize=TICK_FONTSIZE) 
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/interaction_heatmap_summary.png', dpi=300)

print(f"✅ 处理完成！Ms.，图表布局已优化并保存至 {output_dir}")