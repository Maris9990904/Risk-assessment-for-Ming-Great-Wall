import pandas as pd
import matplotlib.pyplot as plt
import mapclassify
import os

# 1. 读取数据
file_path = 'Great/2th/RiskHEV8700.xlsx'

if not os.path.exists(file_path):
    print(f"错误：未找到文件 {file_path}，请检查路径是否正确。")
else:
    df = pd.read_excel(file_path)
    
    # 需要分析的列名
    columns = ['Altitude', 'Matirial', 'Final_Risk']
    
    # 检查列是否存在
    existing_cols = [col for col in columns if col in df.columns]
    
    for col in existing_cols:
        # 2. 应用自然断点法分为5类
        # 注意：如果某列唯一值太少，自然断点法可能会自动减少分类数
        try:
            classifier = mapclassify.NaturalBreaks(df[col], k=5)
            # 获取分类结果（0到4）
            df[f'{col}_class'] = classifier.yb
            
            # 3. 计算各类别占比
            counts = df[f'{col}_class'].value_counts(normalize=True).sort_index()
            
            # 4. 绘图
            plt.figure(figsize=(10, 6))
            bars = plt.bar(counts.index, counts.values, color='steelblue', alpha=0.8, edgecolor='black')
            
            # 设置标题和标签
            plt.title(f'Proportion of {col} Categories (Natural Breaks)', fontsize=15)
            plt.xlabel('Category (1-5)', fontsize=12)
            plt.ylabel('Proportion', fontsize=12)
            plt.xticks(range(len(counts)), [f'Class {i+1}' for i in range(len(counts))])
            plt.ylim(0, max(counts.values) * 1.15)  # 留出顶部空间显示文字
            
            # 在柱状图上方添加百分比标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{height:.2%}', ha='center', va='bottom', fontsize=11)
            
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 输出图片
            output_name = f'outputs/{col}_Proportion.png'
            plt.savefig(output_name, dpi=300, bbox_inches='tight')
            print(f"已生成图表: {output_name}")
            plt.close()
            
        except Exception as e:
            print(f"处理列 {col} 时出错: {e}")

    print("\n所有处理完成。")