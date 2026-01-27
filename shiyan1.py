import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. 数据加载
# ==========================================
file_path = 'RiskHEV8700.xlsx'
if not os.path.exists(file_path):
    print(f"错误：找不到文件 {file_path}")
else:
    df = pd.read_excel(file_path)

    # 定义指标维度
    h_vars = ['EQ', 'LS', 'WE', 'RE', 'FT', 'SA', 'NDVI', 'HRF', 'CL', 'RND', 'GI']
    v_vars = ['MAT', 'PS']
    p_vars = ['PL']
    
    # 海拔分组映射
    ele_map = {1: 'ELAR', 2: 'LAR', 3: 'MAR', 4: 'HAR', 5: 'EHAR'}

    # ==========================================
    # 2. 权重计算函数定义
    # ==========================================
    def calculate_entropy_weights(df_norm):
        """熵权法计算权重"""
        n = len(df_norm)
        if n < 2:
            return pd.Series({col: 1/len(df_norm.columns) for col in df_norm.columns})
        weights = {}
        for col in df_norm.columns:
            p = df_norm[col] / df_norm[col].sum()
            p = p.replace(0, 1e-10)  # 避免log(0)
            e = -1 / np.log(n) * (p * np.log(p)).sum()  # 熵值
            weights[col] = 1 - e  # 信息效用值
        total = sum(weights.values())
        if total == 0:
            return pd.Series({col: 1/len(df_norm.columns) for col in df_norm.columns})
        return pd.Series({k: v/total for k, v in weights.items()})

    def calculate_critic_weights(df_norm):
        """CRITIC法计算权重"""
        std = df_norm.std()
        corr_matrix = df_norm.corr()
        # CRITIC 冲突性：sum(1 - r_ij)
        conflict = (1 - corr_matrix).sum()
        c_info = std * conflict
        if c_info.sum() == 0:
            return pd.Series({col: 1/len(df_norm.columns) for col in df_norm.columns})
        return c_info / c_info.sum()

    def calculate_combined_weights(df_norm):
        """组合权重法：计算熵权法、CRITIC及多种组合方式"""
        w_entropy = calculate_entropy_weights(df_norm)
        w_critic = calculate_critic_weights(df_norm)
        # 乘积组合
        w_product = w_entropy * w_critic
        if w_product.sum() == 0:
            w_product = pd.Series({col: 1/len(df_norm.columns) for col in df_norm.columns})
        else:
            w_product = w_product / w_product.sum()
        # 平均组合
        w_average = (w_entropy + w_critic) / 2
        w_average = w_average / w_average.sum()
        return w_entropy, w_critic, w_product, w_average

    # ==========================================
    # 3. 按海拔分组计算权重和Hazard_Index
    # ==========================================
    print("="*70)
    print("按海拔分组计算权重和Hazard_Index")
    print("="*70)
    
    # 存储所有海拔的权重
    all_weights = []
    
    # 初始化结果列
    df['Hazard_Index_byAlt'] = 0.0
    
    for ele_val, ele_name in ele_map.items():
        print(f"\n【{ele_name} (Altitude={ele_val})】")
        
        # 筛选该海拔数据
        mask = df['Altitude'] == ele_val
        df_sub = df[mask].copy()
        
        if len(df_sub) < 10:
            print(f"  样本数不足({len(df_sub)}), 跳过")
            continue
        
        # 对该海拔数据单独归一化
        norm_h_sub = (df_sub[h_vars] - df_sub[h_vars].min()) / (df_sub[h_vars].max() - df_sub[h_vars].min())
        norm_h_sub = norm_h_sub.fillna(0)
        
        # 计算该海拔的权重
        w_entropy, w_critic, w_product, w_average = calculate_combined_weights(norm_h_sub)
        
        # 保存权重
        weight_row = {'Altitude': ele_name, 'n': len(df_sub)}
        for col in h_vars:
            weight_row[f'{col}_Entropy'] = w_entropy[col]
            weight_row[f'{col}_CRITIC'] = w_critic[col]
            weight_row[f'{col}_Product'] = w_product[col]
        all_weights.append(weight_row)
        
        # 计算该海拔的Hazard_Index (使用乘积组合权重)
        hazard_index_sub = (norm_h_sub * w_product).sum(axis=1) * 5
        df.loc[mask, 'Hazard_Index_byAlt'] = hazard_index_sub.values
        
        # 打印该海拔的前3个权重最高的因子
        top3 = w_product.sort_values(ascending=False).head(3)
        print(f"  样本数: {len(df_sub)}")
        print(f"  Top3权重因子: {', '.join([f'{k}({v:.1%})' for k, v in top3.items()])}")
    
    # ==========================================
    # 4. 保存各海拔权重对比表
    # ==========================================
    weights_by_alt = pd.DataFrame(all_weights)
    weights_by_alt.to_excel('Weights_by_Altitude.xlsx', index=False)
    print(f"\n各海拔权重对比表已保存为: Weights_by_Altitude.xlsx")
    
    # ==========================================
    # 5. 同时计算全局权重（用于对比）
    # ==========================================
    norm_h_global = (df[h_vars] - df[h_vars].min()) / (df[h_vars].max() - df[h_vars].min())
    norm_h_global = norm_h_global.fillna(0)
    w_entropy_g, w_critic_g, w_product_g, w_average_g = calculate_combined_weights(norm_h_global)
    
    # 全局Hazard_Index
    df['Hazard_Index_Global'] = (norm_h_global * w_product_g).sum(axis=1) * 5
    
    # 保存全局权重
    weights_global = pd.DataFrame({
        'Factor': h_vars,
        'Entropy': [w_entropy_g[col] for col in h_vars],
        'CRITIC': [w_critic_g[col] for col in h_vars],
        'Product_Combined': [w_product_g[col] for col in h_vars],
        'Average_Combined': [w_average_g[col] for col in h_vars]
    }).set_index('Factor').sort_values(by='Product_Combined', ascending=False)
    weights_global.to_excel('Weights_Global.xlsx')
    print("全局权重已保存为: Weights_Global.xlsx")

    # ==========================================
    # 6. 绘制各海拔权重对比图
    # ==========================================
    plt.rcParams.update({'font.size': 14})
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (ele_val, ele_name) in enumerate(ele_map.items()):
        # 获取该海拔的权重
        weight_row = next((w for w in all_weights if w['Altitude'] == ele_name), None)
        if weight_row is None:
            continue
        
        # 提取Product权重
        weights = {col: weight_row[f'{col}_Product'] for col in h_vars}
        weights_series = pd.Series(weights).sort_values(ascending=False)
        
        # 绘图
        colors = sns.color_palette("flare_r", len(weights_series))
        axes[i].barh(weights_series.index, weights_series.values, color=colors)
        axes[i].set_title(f'{ele_name} (n={weight_row["n"]})', fontsize=16)
        axes[i].set_xlabel('Weight')
        axes[i].invert_yaxis()
    
    # 删除多余子图
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig('Weights_by_Altitude_Plot.png', dpi=300, bbox_inches='tight')
    print("各海拔权重对比图已保存为: Weights_by_Altitude_Plot.png")
    plt.show()

    # ==========================================
    # 7. 风险维度得分计算
    # ==========================================
    results = pd.DataFrame({'FID': df['FID']})
    results['Hazard_Index_byAlt'] = df['Hazard_Index_byAlt']
    results['Hazard_Index_Global'] = df['Hazard_Index_Global']
    
    # V 指数
    results['Sensitivity_Index'] = (df['MAT'] + df['PS'] / 6 * 5 )/2
    
    # P 指数
    results['Adaptive_Capacity_Index'] = df['PL'] 

    # ==========================================
    # 8. 最终风险计算 (使用按海拔计算的Hazard_Index)
    # ==========================================
    results['Final_Risk'] = (results['Hazard_Index_byAlt'] * results['Sensitivity_Index']) / \
                            results['Adaptive_Capacity_Index']

    # ==========================================
    # 9. 输出结果
    # ==========================================
    final_output = pd.merge(df, results[['FID', 'Hazard_Index_byAlt', 'Hazard_Index_Global', 
                                          'Sensitivity_Index', 'Adaptive_Capacity_Index', 'Final_Risk']], 
                           on='FID', suffixes=('', '_new'))
    final_output.to_excel('Great_Wall_Risk_Assessment_Final.xlsx', index=False)

    print("\n" + "="*70)
    print("处理完成！")
    print("="*70)
    print(f"Hazard_Index_byAlt 范围: {results['Hazard_Index_byAlt'].min():.2f} - {results['Hazard_Index_byAlt'].max():.2f}")
    print(f"Hazard_Index_Global 范围: {results['Hazard_Index_Global'].min():.2f} - {results['Hazard_Index_Global'].max():.2f}")
    print(f"Final Risk 范围: {results['Final_Risk'].min():.2f} - {results['Final_Risk'].max():.2f}")

print("\n任务运行结束。")