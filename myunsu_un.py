import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# 0. 环境设置与路径
# ==========================================
output_dir = 'Results_Great/Hazard_Unsupervised'
os.makedirs(output_dir, exist_ok=True)

# 绘图字体全局设置
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 定义 RGB 颜色并归一化为 0-1
rgb_colors = [
    (56/255, 168/255, 0/255),    # Very low - 深绿
    (139/255, 209/255, 0/255),   # Low - 浅绿
    (255/255, 255/255, 0/255),   # Medium - 黄色
    (255/255, 128/255, 0/255),   # High - 橙色
    (255/255, 0/255, 0/255)      # Very high - 红色
]

# ==========================================
# 1. 无监督聚类核心函数
# ==========================================

def unsupervised_classification(df_sub, h_vars, n_clusters=5):
    scaler = StandardScaler()
    X = df_sub[h_vars].fillna(0)
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(X_scaled)
    
    df_temp = df_sub.copy()
    df_temp['cluster'] = labels_kmeans
    norm_h = (df_sub[h_vars] - df_sub[h_vars].min()) / (df_sub[h_vars].max() - df_sub[h_vars].min())
    df_temp['hazard_score'] = norm_h.mean(axis=1).fillna(0)
    
    cluster_scores = df_temp.groupby('cluster')['hazard_score'].mean().sort_values()
    rank_map = {cluster: rank + 1 for rank, cluster in enumerate(cluster_scores.index)}
    
    hazard_class = df_temp['cluster'].map(rank_map)
    mapped_scores = {rank_map[c]: score for c, score in cluster_scores.items()}
    
    return hazard_class, mapped_scores

# ==========================================
# 2. 执行计算
# ==========================================
file_path = 'RiskHEV8700.xlsx'
df_raw = pd.read_excel(file_path)

h_vars = ['EQ', 'LS', 'WE', 'RE', 'FT', 'SA', 'NDVI', 'HRF', 'CL', 'RND', 'GI']
ele_map = {1: 'ELER', 2: 'LER', 3: 'MER', 4: 'HER', 5: 'EHER'}
target = 'Hazard_Class'

df_raw[target] = 0
all_elevation_data = {}

for ele_val, ele_name in ele_map.items():
    mask = df_raw['Altitude'] == ele_val
    df_sub = df_raw[mask].copy()
    if len(df_sub) < 5: continue
    
    h_class, m_scores = unsupervised_classification(df_sub, h_vars)
    df_raw.loc[mask, target] = h_class.values
    
    counts = h_class.value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    all_elevation_data[ele_name] = {'counts': counts, 'scores': m_scores}

df_raw.to_excel(f'{output_dir}/data_with_hazard_class.xlsx', index=False)

# ==========================================
# 3. 定制化绘图
# ==========================================
# 仅首单词首字母大写
labels = ['Very low', 'Low', 'Medium', 'High', 'Very high']
fig, axes = plt.subplots(1, 5, figsize=(32, 10))

for i, (ele_name, data) in enumerate(all_elevation_data.items()):
    ax = axes[i]
    counts = data['counts']
    scores = [data['scores'].get(c, 0) for c in range(1, 6)]
    
    # 1. 绘制柱状图 (左轴)
    bars = ax.bar(labels, counts.values, color=rgb_colors, edgecolor='black', alpha=0.8)
    ax.set_ylabel('Count (Sample Size)', fontsize=20)
    ax.set_ylim(0, counts.max() * 1.3)
    ax.tick_params(axis='y', labelsize=20)
    
    # 在柱子顶端添加数量标注
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        current_label = labels[idx]
        
        # --- 修改：MER 的 Very low 标注往下放一点 ---
        if ele_name == 'MER' and current_label == 'Very low':
            y_pos_bar = height - (counts.max() * 0.01) # 稍微向下偏移
            v_align_bar = 'top'
            color_bar = 'black' # 放在柱子里通常用白色更清晰，或者根据需要改回 black
        else:
            y_pos_bar = height
            v_align_bar = 'bottom'
            color_bar = 'black'

        ax.text(bar.get_x() + bar.get_width()/2., y_pos_bar,
                f'{int(height)}', ha='center', va=v_align_bar, 
                fontsize=18, fontweight='bold', color=color_bar)

    # 2. 绘制折线图 (右轴)
    ax2 = ax.twinx()
    ax2.plot(labels, scores, color='black', marker='D', markersize=10, linewidth=3, label='Mean Score')
    ax2.set_ylabel('Mean Hazard Score', fontsize=20)
    
    max_s = max(scores) if max(scores) > 0 else 1
    ax2.set_ylim(0, max_s * 1.4)
    ax2.tick_params(axis='y', labelsize=20)

    # 针对性调整折线标注位置
    for x_idx, s_val in enumerate(scores):
        current_label = labels[x_idx]
        h_align, v_align = 'center', 'bottom'
        x_pos, y_pos = x_idx, s_val + (max_s * 0.03)

        # LER / MER 的 High 左移
        if ele_name == 'LER' and current_label == 'High':
            h_align, x_pos, y_pos = 'right', x_idx - 0.03, s_val + 0.01
        elif ele_name == 'MER' and current_label == 'High':
            h_align, x_pos, y_pos = 'right', x_idx+0.01, s_val + 0.015
        # EHER 的部分标注下移
        elif ele_name == 'EHER' and current_label in ['Low', 'Very high']:
            v_align, y_pos = 'top', s_val - (max_s * 0.03)

        ax2.text(x_pos, y_pos, f'{s_val:.2f}', 
                 ha=h_align, va=v_align, 
                 fontsize=18, color='black', fontweight='bold')

    ax.set_title(f'Elevation: {ele_name}', fontsize=24, pad=20)
    ax.set_xticklabels(labels, fontsize=20, rotation=25) 
    
    if i == 0:
        ax2.legend(loc='upper left', fontsize=16)

plt.tight_layout()
plt.savefig(f'{output_dir}/hazard_class_distribution.png', dpi=300)
plt.show()

print(f"✅ 处理完成！Ms.，请检查 MER 的柱状图标注是否已调整。")