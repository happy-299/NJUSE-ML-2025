import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # 用黑体显示中文
rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 读取数据
with open('eval_result/para_return_mehod.json', encoding='utf-8') as f:
    Equivalence_java = json.load(f)
with open('eval_result/Name_overall_1.json', encoding='utf-8') as f:
    rq2_1 = json.load(f)
with open('eval_result/Name_overall_2.json', encoding='utf-8') as f:
    rq2_2 = json.load(f)
with open('eval_result/Name_overall_3.json', encoding='utf-8') as f:
    rq2_3 = json.load(f)
with open('eval_result/generalization.json', encoding='utf-8') as f:
    Equivalence_python = json.load(f)

# 创建图表目录
import os
os.makedirs('figures', exist_ok=True)

# ========== 图1: RQ1 Java准确率对比 ==========
fig, ax = plt.subplots(figsize=(10, 6))

methods = ['APIzator', 'Code2API']
params = [Equivalence_java['APIzator_Human_P'].count(1) / 2, 
          Equivalence_java['Code2API_Human_P'].count(1) / 2]
returns = [Equivalence_java['APIzator_Human_R'].count(1) / 2,
           Equivalence_java['Code2API_Human_R'].count(1) / 2]
methods_impl = [Equivalence_java['APIzator_Human'].count(1) / 2,
                Equivalence_java['Code2API_Human'].count(1) / 2]

x = np.arange(len(methods))
width = 0.25

bars1 = ax.bar(x - width, params, width, label='参数准确率 (E_P)', color='#3498db')
bars2 = ax.bar(x, returns, width, label='返回值准确率 (E_R)', color='#2ecc71')
bars3 = ax.bar(x + width, methods_impl, width, label='方法实现准确率 (E_M)', color='#e74c3c')

# 添加数值标签
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)

ax.set_xlabel('方法', fontsize=12)
ax.set_ylabel('准确率 (%)', fontsize=12)
ax.set_title('RQ1: Java API生成准确率对比', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.set_ylim(0, 80)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/rq1_java_accuracy.png', dpi=300, bbox_inches='tight')
print("✅ 已生成: figures/rq1_java_accuracy.png")
plt.close()

# ========== 图2: RQ2 方法名评分分布 ==========
method_name_APIzator = rq2_1['APIzator'] + rq2_2['APIzator'] + rq2_3['APIzator']
method_name_Human = rq2_1['Human'] + rq2_2['Human'] + rq2_3['Human']
method_name_Code2API = rq2_1['Code2API'] + rq2_2['Code2API'] + rq2_3['Code2API']

scores = ['1分', '2分', '3分', '4分']
apizator_dist = [method_name_APIzator.count(i) for i in range(1, 5)]
human_dist = [method_name_Human.count(i) for i in range(1, 5)]
code2api_dist = [method_name_Code2API.count(i) for i in range(1, 5)]

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(scores))
width = 0.25

bars1 = ax.bar(x - width, apizator_dist, width, label='APIzator', color='#95a5a6')
bars2 = ax.bar(x, human_dist, width, label='Human', color='#f39c12')
bars3 = ax.bar(x + width, code2api_dist, width, label='Code2API', color='#9b59b6')

# 添加百分比标签
for bars, dist in [(bars1, apizator_dist), (bars2, human_dist), (bars3, code2api_dist)]:
    for bar, count in zip(bars, dist):
        height = bar.get_height()
        percentage = count / 6  # 总共600个评分（200个API × 3个评估者）
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=9)

ax.set_xlabel('方法名质量评分', fontsize=12)
ax.set_ylabel('数量', fontsize=12)
ax.set_title('RQ2: 方法名质量评分分布', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(scores)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/rq2_method_name_distribution.png', dpi=300, bbox_inches='tight')
print("✅ 已生成: figures/rq2_method_name_distribution.png")
plt.close()

# ========== 图3: RQ2 平均方法名评分对比 ==========
fig, ax = plt.subplots(figsize=(8, 6))

methods = ['APIzator', 'Human', 'Code2API']
avg_scores = [
    np.mean(method_name_APIzator),
    np.mean(method_name_Human),
    np.mean(method_name_Code2API)
]
colors = ['#95a5a6', '#f39c12', '#9b59b6']

bars = ax.bar(methods, avg_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# 添加数值标签
for bar, score in zip(bars, avg_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.2f}',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

# 添加参考线
ax.axhline(y=3.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='良好水平 (3.0)')
ax.axhline(y=4.0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='完美水平 (4.0)')

ax.set_ylabel('平均评分', fontsize=12)
ax.set_title('RQ2: 平均方法名质量评分对比', fontsize=14, fontweight='bold')
ax.set_ylim(0, 4.5)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/rq2_avg_score.png', dpi=300, bbox_inches='tight')
print("✅ 已生成: figures/rq2_avg_score.png")
plt.close()

# ========== 图4: RQ2 最佳API投票结果饼图 ==========
from collections import Counter

best_APIs = [(i, j, k) for i, j, k in zip(rq2_1['best_API'], rq2_2['best_API'], rq2_3['best_API'])]
best_API_vote = []
for t in best_APIs:
    element_counts = Counter(t)
    most_common_elements = [element for element, count in element_counts.items() if
                            count == element_counts.most_common(1)[0][1]]
    if len(most_common_elements) == 1:
        best_API_vote.append(most_common_elements[0])

labels = ['APIzator', 'Human', 'Code2API']
sizes = [best_API_vote.count(1), best_API_vote.count(2), best_API_vote.count(3)]
colors = ['#95a5a6', '#f39c12', '#9b59b6']
explode = (0, 0, 0.1)  # 突出显示Code2API

fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                    autopct='%1.1f%%', startangle=90,
                                    textprops={'fontsize': 12})

# 美化百分比文字
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(14)

# 添加具体数量
for i, (label, size) in enumerate(zip(labels, sizes)):
    angle = (wedges[i].theta2 + wedges[i].theta1) / 2
    x = 1.2 * np.cos(np.radians(angle))
    y = 1.2 * np.sin(np.radians(angle))
    ax.text(x, y, f'{size}个', ha='center', va='center', fontsize=11, fontweight='bold')

ax.set_title('RQ2: 最佳API投票结果 (n=200)', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('figures/rq2_best_api_pie.png', dpi=300, bbox_inches='tight')
print("✅ 已生成: figures/rq2_best_api_pie.png")
plt.close()

# ========== 图5: RQ3 泛化能力对比（Java vs Python）==========
fig, ax = plt.subplots(figsize=(10, 6))

languages = ['Java', 'Python']
params_lang = [
    Equivalence_java['Code2API_Human_P'].count(1) / 2,
    Equivalence_python['Code2API_Human_P'].count(1) / 1
]
returns_lang = [
    Equivalence_java['Code2API_Human_R'].count(1) / 2,
    Equivalence_python['Code2API_Human_R'].count(1) / 1
]
methods_lang = [
    Equivalence_java['Code2API_Human'].count(1) / 2,
    Equivalence_python['Code2API_Human'].count(1) / 1
]

x = np.arange(len(languages))
width = 0.25

bars1 = ax.bar(x - width, params_lang, width, label='参数准确率 (E_P)', color='#3498db')
bars2 = ax.bar(x, returns_lang, width, label='返回值准确率 (E_R)', color='#2ecc71')
bars3 = ax.bar(x + width, methods_lang, width, label='方法实现准确率 (E_M)', color='#e74c3c')

# 添加数值标签
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xlabel('编程语言', fontsize=12)
ax.set_ylabel('准确率 (%)', fontsize=12)
ax.set_title('RQ3: Code2API泛化能力 (Java vs Python)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(languages)
ax.legend()
ax.set_ylim(0, 90)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/rq3_generalization.png', dpi=300, bbox_inches='tight')
print("✅ 已生成: figures/rq3_generalization.png")
plt.close()

# ========== 图6: 综合对比雷达图 ==========
from math import pi

categories = ['参数\n准确率', '返回值\n准确率', '方法实现\n准确率', '方法名\n质量', '整体API\n质量']
N = len(categories)

# 归一化数据到0-100
apizator_values = [
    Equivalence_java['APIzator_Human_P'].count(1) / 2,
    Equivalence_java['APIzator_Human_R'].count(1) / 2,
    Equivalence_java['APIzator_Human'].count(1) / 2,
    (np.mean(method_name_APIzator) / 4) * 100,
    (best_API_vote.count(1) / len(best_API_vote)) * 100
]

human_values = [
    100,  # 基准
    100,  # 基准
    100,  # 基准
    (np.mean(method_name_Human) / 4) * 100,
    (best_API_vote.count(2) / len(best_API_vote)) * 100
]

code2api_values = [
    Equivalence_java['Code2API_Human_P'].count(1) / 2,
    Equivalence_java['Code2API_Human_R'].count(1) / 2,
    Equivalence_java['Code2API_Human'].count(1) / 2,
    (np.mean(method_name_Code2API) / 4) * 100,
    (best_API_vote.count(3) / len(best_API_vote)) * 100
]

angles = [n / float(N) * 2 * pi for n in range(N)]
apizator_values += apizator_values[:1]
human_values += human_values[:1]
code2api_values += code2api_values[:1]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

ax.plot(angles, apizator_values, 'o-', linewidth=2, label='APIzator', color='#95a5a6')
ax.fill(angles, apizator_values, alpha=0.15, color='#95a5a6')

ax.plot(angles, human_values, 'o-', linewidth=2, label='Human', color='#f39c12')
ax.fill(angles, human_values, alpha=0.15, color='#f39c12')

ax.plot(angles, code2api_values, 'o-', linewidth=2, label='Code2API', color='#9b59b6')
ax.fill(angles, code2api_values, alpha=0.15, color='#9b59b6')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
ax.grid(True)

ax.set_title('综合性能对比雷达图', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.savefig('figures/comprehensive_radar.png', dpi=300, bbox_inches='tight')
print("✅ 已生成: figures/comprehensive_radar.png")
plt.close()

print("\n" + "="*50)
print("所有图表生成完成！保存在 figures/ 目录下")
print("="*50)
print("\n生成的图表：")
print("1. rq1_java_accuracy.png - RQ1 Java准确率对比")
print("2. rq2_method_name_distribution.png - 方法名评分分布")
print("3. rq2_avg_score.png - 平均方法名评分对比")
print("4. rq2_best_api_pie.png - 最佳API投票饼图")
print("5. rq3_generalization.png - 泛化能力对比")
print("6. comprehensive_radar.png - 综合性能雷达图")
