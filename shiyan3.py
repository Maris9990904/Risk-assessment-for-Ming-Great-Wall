import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import itertools
import jenkspy  # pip install jenkspy

# ==========================================
# 0. 环境设置与路径
# ==========================================
output_dir = 'Results_Great/Hazard_Mechanism_Material'
os.makedirs(output_dir, exist_ok=True)

sns.set_context("paper", font_scale=1.8)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 核心计算函数
# ==========================================

def adaptive_discretization(series, col_name, n_bins=5):
    """自适应离散化：除了类别因子，全部使用自然断点法
    
    修复：当唯一值数量不足时，避免使用qcut强制分组产生虚假结果
    """
    if col_name in ['CL', 'SA', 'Altitude']:  # 这些通常已是类别或分组列
        return series
    
    # 修复：检查唯一值数量，避免虚假分组
    n_unique = series.nunique()
    if n_unique < 2:
        # 只有1个唯一值，无法离散化，保持原样（q值将为0）
        return series
    
    # 动态调整分箱数：不能超过唯一值数量
    actual_bins = min(n_bins, n_unique)
    
    try:
        breaks = jenkspy.jenks_breaks(series, n_classes=actual_bins)
        return pd.cut(series, bins=breaks, labels=False, include_lowest=True)
    except:
        # Fallback: 使用实际可行的分箱数进行等频分箱
        return pd.qcut(series.rank(method='first'), q=actual_bins, labels=False, duplicates='drop')

def factor_detector(df, feature_cols, target_col):
    """因子探测：计算解释力 q"""
    results = {}
    y = df[target_col]
    sst = y.var(ddof=0) * len(y)
    if sst == 0:
        return {col: 0.0 for col in feature_cols}
    for col in feature_cols:
        grouped = df.groupby(col)[target_col]
        # 使用总体方差(ddof=0)避免单样本组(var=0)被跳过导致 ssw 低估、q 被抬高
        ssw = sum(group.var(ddof=0) * len(group) for _, group in grouped)
        results[col] = max(0.0, 1.0 - (ssw / sst))
    return results

def get_interaction_type(q1, q2, q12):
    """判断交互类型"""
    if q12 < min(q1, q2): return "非线性削弱"
    if min(q1, q2) < q12 < max(q1, q2): return "单因子削弱"
    if q12 > (q1 + q2): return "非线性增强"
    if q12 > max(q1, q2): return "双因子增强"
    return "独立"
# ==========================================
# 2. 数据处理与计算
# ==========================================
file_path = 'RiskHEV8700.xlsx'
df_raw = pd.read_excel(file_path)

target = 'Hazard_Index'
group_col = 'Matirial'
features = ['EQ', 'LS', 'WE', 'RE', 'FT', 'SA', 'NDVI', 'HRF', 'CL', 'RND', 'GI']
# 材质名称映射
mat_map = {1: 'WBW', 2: 'EW', 3: 'BW', 4: 'SW', 5: 'MHW'}
mat_names = ['WBW', 'EW', 'BW', 'SW', 'MHW']

# --- A. 按材质类型绘制因子排名图 ---
fig, axes = plt.subplots(2, 3, figsize=(20, 14)) # 创建 2x3 的子图布局
axes = axes.flatten()

# 用于存储所有材质汇总数据的列表
all_material_q = []

for i, (mat_val, mat_name) in enumerate(mat_map.items()):
    # 1. 筛选特定材质数据
    df_sub = df_raw[df_raw[group_col] == mat_val].copy()
    if df_sub.empty:
        continue
        
    # 2. 离散化
    df_dis = df_sub.copy()
    for col in features:
        df_dis[col] = adaptive_discretization(df_sub[col], col)
    
    # 3. 计算因子探测器 q 值
    current_q = factor_detector(df_dis, features, target)
    
    # 4. 转换为 DataFrame 并排序
    rank_df = pd.DataFrame(list(current_q.items()), columns=['Feature', 'q']).sort_values('q', ascending=False)
    rank_df['Material'] = mat_name
    all_material_q.append(rank_df)
    
    # 5. 绘图
    sns.barplot(x='q', y='Feature', data=rank_df, ax=axes[i], palette='magma')
    axes[i].set_title(f'Material: {mat_name} (n={len(df_sub)})', fontsize=20)
    axes[i].set_xlabel('Factor Determinant (q)')
    axes[i].set_ylabel('')

# 移除多余的子图（如果有）
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(f'{output_dir}/factor_ranking_by_material.png', dpi=300)
plt.show()

# --- B. 原有的分组计算逻辑继续 (保持不变或复用上述计算结果) ---
# ... 后面接原有的 B 以后部分 ...

# --- B. 分组计算 (按材质分组) ---
q_summary_list = []
interaction_list = []
risk_trends = []

for mat_val, mat_name in mat_map.items():
    df_sub = df_raw[df_raw[group_col] == mat_val].copy()
    if len(df_sub) < 10: continue
    
    df_dis = df_sub.copy()
    for col in features:
        df_dis[col] = adaptive_discretization(df_sub[col], col)
    
    # 1. 因子探测 (存入 summary)
    q_dict = factor_detector(df_dis, features, target)
    q_dict['Material_Type'] = mat_name
    q_summary_list.append(q_dict)
    
    # 2. 风险探测 (存入 risk_trends)
    for col in features:
        mean_risk = df_dis.groupby(col)[target].mean()
        for strata, val in mean_risk.items():
            risk_trends.append({'Material': mat_name, 'Factor': col, 'Strata': strata, 'Mean_Risk': val})

    # 3. 交互探测 (选当前组前两名)
    top_factors = pd.Series(q_dict).drop('Material_Type').astype(float).sort_values(ascending=False).head(3).index.tolist()
    pairs = list(itertools.combinations(top_factors, 2))
    sst_sub = df_dis[target].var(ddof=0) * len(df_dis)
    
    for p1, p2 in pairs:
        df_dis['inter'] = df_dis[p1].astype(str) + "_" + df_dis[p2].astype(str)
        ssw_inter = sum(g.var(ddof=0) * len(g) for _, g in df_dis.groupby('inter')[target])
        q_inter = 1 - (ssw_inter / sst_sub if sst_sub != 0 else 0)
        interaction_list.append({
            'Material': mat_name, 'Factor_A': p1, 'Factor_B': p2,
            'q_A': q_dict[p1], 'q_B': q_dict[p2], 'q_Inter': q_inter,
            'Type': get_interaction_type(q_dict[p1], q_dict[p2], q_inter)
        })

# ==========================================
# 3. 结果汇总与图表输出
# ==========================================

# 1. 保存 Excel 结果
q_summary_df = pd.DataFrame(q_summary_list).set_index('Material_Type')
q_summary_df.to_excel(f'{output_dir}/geodetector_q_summary.xlsx')

risk_df = pd.DataFrame(risk_trends)
risk_df.to_excel(f'{output_dir}/risk_detector.xlsx', index=False)

inter_df = pd.DataFrame(interaction_list)
inter_df.to_excel(f'{output_dir}/interaction_results.xlsx', index=False)

# 2. 绘制 Interaction Heatmap (交互热力图汇总)
plt.figure(figsize=(14, 10))
pivot_inter = inter_df.pivot_table(index="Material", columns=["Factor_A", "Factor_B"], values="q_Inter")
material_order = list(mat_map.values())
pivot_inter = pivot_inter.reindex(material_order).dropna(how="all")
sns.heatmap(pivot_inter, annot=True, cmap="YlOrRd", fmt=".3f")
plt.title('Interaction Detector Summary by Material (q_Inter)')
plt.savefig(f'{output_dir}/interaction_heatmap_summary.png', dpi=300)


print(f"✅ 所有文件已生成并保存至: {output_dir}")