import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json


# ============== 常量定义 ==============

# Portable artifact locations.
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / 'comparison' / 'artifacts'
CATEGORY_CENTERS_DIR = ARTIFACTS_DIR / 'category_centers' / 'centers'
OUTPUT_DIR = ARTIFACTS_DIR / 'analysis' / 'category_centers'

# 道德维度配置
MORAL_DIMENSIONS = {
    'Care-Harm': ['care', 'harm'],
    'Fairness-Cheating': ['fairness', 'cheating'],
    'Loyalty-Betrayal': ['loyalty', 'betrayal'],
    'Authority-Subversion': ['authority', 'subversion'],
    'Sanctity-Degradation': ['sanctity', 'degradation']
}

# 可视化颜色
DIMENSION_COLORS = {
    'Care-Harm': '#E74C3C',
    'Fairness-Cheating': '#3498DB',
    'Loyalty-Betrayal': '#2ECC71',
    'Authority-Subversion': '#9B59B6',
    'Sanctity-Degradation': '#F39C12'
}


# ============== 工具函数 ==============

def ensure_dir(path):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)


def load_category_centers(model_name):
    """加载范畴中心数据"""
    centers_path = os.path.join(CATEGORY_CENTERS_DIR, model_name, 'category_centers.pt')
    if not os.path.exists(centers_path):
        return None
    return torch.load(centers_path, map_location='cpu', weights_only=False)


def cosine_similarity(v1, v2):
    """计算余弦相似度"""
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)


def compute_pairwise_cosine_matrix(vectors):
    """
    计算所有范畴中心之间的成对余弦相似度矩阵
    
    Args:
        vectors: dict, {category: vector}
    
    Returns:
        categories: list of category names
        sim_matrix: np.ndarray, 余弦相似度矩阵
    """
    categories = sorted(vectors.keys())
    n = len(categories)
    sim_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = cosine_similarity(vectors[categories[i]], vectors[categories[j]])
    
    return categories, sim_matrix


def compute_layer_statistics(centers, pooling_type='mean_pooling'):
    """
    计算每一层的统计信息
    
    Args:
        centers: 范畴中心数据
        pooling_type: 池化类型
    
    Returns:
        layer_stats: dict, {layer_idx: stats}
    """
    layers = centers['layer_indices']
    layer_stats = {}
    
    for layer_idx in layers:
        layer_data = centers[pooling_type][layer_idx]
        vectors = {cat: np.asarray(vec) for cat, vec in layer_data.items()}
        
        categories, sim_matrix = compute_pairwise_cosine_matrix(vectors)
        
        # 计算统计量（排除对角线）
        n = len(categories)
        off_diag_mask = ~np.eye(n, dtype=bool)
        off_diag_sims = sim_matrix[off_diag_mask]
        
        # 计算每个道德维度的virtue-vice相似度
        dimension_sims = {}
        for dim_name, (virtue, vice) in MORAL_DIMENSIONS.items():
            if virtue in vectors and vice in vectors:
                sim = cosine_similarity(vectors[virtue], vectors[vice])
                dimension_sims[dim_name] = sim
        
        layer_stats[layer_idx] = {
            'mean_sim': np.mean(off_diag_sims),
            'std_sim': np.std(off_diag_sims),
            'min_sim': np.min(off_diag_sims),
            'max_sim': np.max(off_diag_sims),
            'dimension_sims': dimension_sims,
            'sim_matrix': sim_matrix,
            'categories': categories,
            'vectors': vectors
        }
    
    return layer_stats


def find_optimal_layer(layer_stats, criterion='mean'):
    """
    找到最优层（范畴区分度最高的层）
    
    Args:
        layer_stats: 层统计信息
        criterion: 'mean' (平均相似度最低) 或 'virtue_vice' (virtue-vice相似度最低)
    
    Returns:
        optimal_layer: 最优层索引
    """
    if criterion == 'mean':
        # 找平均相似度最低的层
        optimal_layer = min(layer_stats.keys(), key=lambda x: layer_stats[x]['mean_sim'])
    elif criterion == 'virtue_vice':
        # 找virtue-vice平均相似度最低的层
        def avg_virtue_vice_sim(layer_idx):
            sims = layer_stats[layer_idx]['dimension_sims']
            return np.mean(list(sims.values())) if sims else float('inf')
        optimal_layer = min(layer_stats.keys(), key=avg_virtue_vice_sim)
    
    return optimal_layer


# ============== 可视化函数 ==============

def plot_layer_similarity_curve(layer_stats, model_name, save_path):
    """绘制各层平均余弦相似度曲线"""
    layers = sorted(layer_stats.keys())
    mean_sims = [layer_stats[l]['mean_sim'] for l in layers]
    std_sims = [layer_stats[l]['std_sim'] for l in layers]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制均值曲线
    ax.plot(layers, mean_sims, 'b-o', linewidth=2, markersize=6, label='Mean Cosine Similarity')
    
    # 绘制标准差区域
    mean_sims = np.array(mean_sims)
    std_sims = np.array(std_sims)
    ax.fill_between(layers, mean_sims - std_sims, mean_sims + std_sims, alpha=0.2)
    
    # 标记最优层
    optimal_layer = min(layers, key=lambda x: layer_stats[x]['mean_sim'])
    optimal_sim = layer_stats[optimal_layer]['mean_sim']
    ax.scatter([optimal_layer], [optimal_sim], color='red', s=200, zorder=5, marker='*', label=f'Optimal Layer {optimal_layer}')
    ax.axvline(x=optimal_layer, color='red', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Mean Pairwise Cosine Similarity', fontsize=12)
    ax.set_title(f'{model_name}\nPairwise Cosine Similarity Across Layers', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_dimension_similarity_curve(layer_stats, model_name, save_path):
    """绘制各层五个道德维度的virtue-vice相似度曲线"""
    layers = sorted(layer_stats.keys())
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for dim_name, color in DIMENSION_COLORS.items():
        sims = [layer_stats[l]['dimension_sims'].get(dim_name, np.nan) for l in layers]
        ax.plot(layers, sims, '-o', color=color, linewidth=2, markersize=5, label=dim_name)
    
    # 计算并绘制平均值
    avg_sims = []
    for l in layers:
        dim_sims = list(layer_stats[l]['dimension_sims'].values())
        avg_sims.append(np.mean(dim_sims) if dim_sims else np.nan)
    ax.plot(layers, avg_sims, 'k--', linewidth=3, markersize=0, label='Average', alpha=0.7)
    
    # 标记平均相似度最低的层
    valid_layers = [(l, s) for l, s in zip(layers, avg_sims) if not np.isnan(s)]
    if valid_layers:
        optimal_layer, optimal_sim = min(valid_layers, key=lambda x: x[1])
        ax.scatter([optimal_layer], [optimal_sim], color='black', s=200, zorder=5, marker='*')
        ax.axvline(x=optimal_layer, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Virtue-Vice Cosine Similarity', fontsize=12)
    ax.set_title(f'{model_name}\nVirtue-Vice Cosine Similarity per Dimension Across Layers', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_similarity_heatmap(sim_matrix, categories, model_name, layer_idx, save_path):
    """绘制余弦相似度热力图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(sim_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
    
    # 设置刻度
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(categories)))
    ax.set_xticklabels([c.capitalize() for c in categories], rotation=45, ha='right')
    ax.set_yticklabels([c.capitalize() for c in categories])
    
    # 添加数值标注
    for i in range(len(categories)):
        for j in range(len(categories)):
            text = ax.text(j, i, f'{sim_matrix[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=8)
    
    ax.set_title(f'{model_name} - Layer {layer_idx}\nPairwise Cosine Similarity', fontsize=14, fontweight='bold')
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Cosine Similarity', rotation=-90, va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cross_model_comparison(all_results, save_path):
    """绘制跨模型比较图：不同模型在最优层的五个道德维度相似度"""
    
    # 过滤有效结果
    valid_results = {k: v for k, v in all_results.items() if v['status'] == 'success'}
    
    if not valid_results:
        print("No valid results for cross-model comparison")
        return
    
    models = list(valid_results.keys())
    dimensions = list(MORAL_DIMENSIONS.keys())
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(models))
    width = 0.15
    
    for i, dim_name in enumerate(dimensions):
        sims = []
        for model in models:
            dim_sims = valid_results[model]['optimal_layer_stats']['dimension_sims']
            sims.append(dim_sims.get(dim_name, np.nan))
        
        offset = (i - len(dimensions)/2 + 0.5) * width
        bars = ax.bar(x + offset, sims, width, label=dim_name, color=DIMENSION_COLORS[dim_name])
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Virtue-Vice Cosine Similarity', fontsize=12)
    ax.set_title('Virtue-Vice Cosine Similarity at Optimal Layer\nAcross Models and Moral Dimensions', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    
    # 添加最优层标注
    for i, model in enumerate(models):
        optimal_layer = valid_results[model]['optimal_layer']
        ax.annotate(f'L{optimal_layer}', (i, ax.get_ylim()[1]), 
                   ha='center', va='bottom', fontsize=8, color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_optimal_layer_distribution(all_results, save_path):
    """绘制最优层在模型中的相对位置分布"""
    
    valid_results = {k: v for k, v in all_results.items() if v['status'] == 'success'}
    
    if not valid_results:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = []
    relative_positions = []
    optimal_layers = []
    total_layers = []
    
    for model, result in valid_results.items():
        models.append(model)
        rel_pos = result['optimal_layer'] / result['num_layers']
        relative_positions.append(rel_pos)
        optimal_layers.append(result['optimal_layer'])
        total_layers.append(result['num_layers'])
    
    # 按相对位置排序
    sorted_indices = np.argsort(relative_positions)
    models = [models[i] for i in sorted_indices]
    relative_positions = [relative_positions[i] for i in sorted_indices]
    optimal_layers = [optimal_layers[i] for i in sorted_indices]
    total_layers = [total_layers[i] for i in sorted_indices]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    
    bars = ax.barh(range(len(models)), relative_positions, color=colors)
    
    # 添加标注
    for i, (rel_pos, opt_l, total_l) in enumerate(zip(relative_positions, optimal_layers, total_layers)):
        ax.annotate(f'Layer {opt_l}/{total_l} ({rel_pos*100:.1f}%)', 
                   (rel_pos + 0.02, i), va='center', fontsize=9)
    
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel('Relative Position in Model (0=First Layer, 1=Last Layer)', fontsize=12)
    ax.set_title('Optimal Layer Position Across Models\n(Layer with Lowest Mean Pairwise Similarity)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.3)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_all_models_layer_curves(all_results, save_path):
    """绘制所有模型的层相似度曲线在同一张图"""
    
    valid_results = {k: v for k, v in all_results.items() if v['status'] == 'success'}
    
    if not valid_results:
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(valid_results)))
    
    for (model, result), color in zip(valid_results.items(), colors):
        layer_stats = result['layer_stats']
        layers = sorted(layer_stats.keys())
        num_layers = result['num_layers']
        
        # 归一化层索引到 [0, 1]
        normalized_layers = [l / num_layers for l in layers]
        mean_sims = [layer_stats[l]['mean_sim'] for l in layers]
        
        ax.plot(normalized_layers, mean_sims, '-', color=color, linewidth=1.5, 
                label=f'{model} ({num_layers}L)', alpha=0.8)
        
        # 标记最优层
        optimal_layer = result['optimal_layer']
        optimal_norm = optimal_layer / num_layers
        optimal_sim = layer_stats[optimal_layer]['mean_sim']
        ax.scatter([optimal_norm], [optimal_sim], color=color, s=100, marker='*', zorder=5)
    
    ax.set_xlabel('Relative Layer Position (0=First, 1=Last)', fontsize=12)
    ax.set_ylabel('Mean Pairwise Cosine Similarity', fontsize=12)
    ax.set_title('Mean Pairwise Similarity Across Layers\n(All Models, Normalized Layer Position)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============== 报告生成 ==============

def generate_markdown_report(all_results, output_path):
    """生成Markdown分析报告"""
    
    report = []
    report.append("# Layer-wise Cosine Similarity Analysis Report")
    report.append("")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("---")
    report.append("")
    
    # 概述
    report.append("## 1. Overview")
    report.append("")
    report.append("This report analyzes the pairwise cosine similarity between category center vectors across different layers of each model. The goal is to identify the **optimal layer** where categories are most distinguishable (i.e., lowest average pairwise similarity).")
    report.append("")
    
    # 方法说明
    report.append("## 2. Methodology")
    report.append("")
    report.append("### 2.1 Pairwise Cosine Similarity")
    report.append("")
    report.append("For each layer, we compute the cosine similarity between all pairs of category center vectors. The **mean pairwise similarity** indicates how similar the categories are on average:")
    report.append("")
    report.append("- **Lower similarity** → Better category separation → More distinguishable moral concepts")
    report.append("- **Higher similarity** → Categories are more similar → Less distinguishable")
    report.append("")
    
    report.append("### 2.2 Virtue-Vice Similarity")
    report.append("")
    report.append("For each moral dimension (e.g., Care-Harm), we also compute the cosine similarity between the virtue and vice categories. This indicates how well the model distinguishes between moral opposites.")
    report.append("")
    
    # 汇总表格
    report.append("## 3. Summary of Optimal Layers")
    report.append("")
    report.append("| Model | Total Layers | Optimal Layer | Relative Position | Mean Similarity |")
    report.append("|-------|--------------|---------------|-------------------|-----------------|")
    
    for model, result in all_results.items():
        if result['status'] != 'success':
            continue
        opt_layer = result['optimal_layer']
        num_layers = result['num_layers']
        rel_pos = opt_layer / num_layers
        mean_sim = result['optimal_layer_stats']['mean_sim']
        report.append(f"| {model} | {num_layers} | {opt_layer} | {rel_pos*100:.1f}% | {mean_sim:.4f} |")
    
    report.append("")
    
    # 可视化
    report.append("## 4. Cross-Model Visualizations")
    report.append("")
    
    report.append("### 4.1 Optimal Layer Position Distribution")
    report.append("")
    report.append("![Optimal Layer Distribution](layer_analysis/optimal_layer_distribution.png)")
    report.append("")
    report.append("This chart shows where the optimal layer is located relative to the model depth. A value of 0.5 means the optimal layer is at the middle of the model.")
    report.append("")
    
    report.append("### 4.2 Layer Similarity Curves (All Models)")
    report.append("")
    report.append("![All Models Layer Curves](layer_analysis/all_models_layer_curves.png)")
    report.append("")
    report.append("This chart overlays the mean pairwise similarity curves for all models, with layer positions normalized to [0, 1]. Stars indicate the optimal layer for each model.")
    report.append("")
    
    report.append("### 4.3 Virtue-Vice Similarity at Optimal Layer")
    report.append("")
    report.append("![Cross-Model Comparison](layer_analysis/cross_model_comparison.png)")
    report.append("")
    report.append("This chart compares the virtue-vice cosine similarity for each moral dimension at the optimal layer across all models.")
    report.append("")
    
    # 各模型详细结果
    report.append("## 5. Model-Specific Results")
    report.append("")
    
    for model, result in all_results.items():
        if result['status'] != 'success':
            continue
        
        report.append(f"### 5.{list(all_results.keys()).index(model)+1} {model}")
        report.append("")
        report.append(f"- **Total Layers:** {result['num_layers']}")
        report.append(f"- **Hidden Dimension:** {result['hidden_dim']}")
        report.append(f"- **Optimal Layer:** {result['optimal_layer']}")
        report.append(f"- **Mean Similarity at Optimal Layer:** {result['optimal_layer_stats']['mean_sim']:.4f}")
        report.append("")
        
        # 层相似度曲线
        report.append("#### Layer Similarity Curve")
        report.append("")
        report.append(f"![{model} Layer Curve](layer_analysis/{model}_layer_curve.png)")
        report.append("")
        
        # 维度相似度曲线
        report.append("#### Dimension-wise Virtue-Vice Similarity")
        report.append("")
        report.append(f"![{model} Dimension Curve](layer_analysis/{model}_dimension_curve.png)")
        report.append("")
        
        # 最优层热力图
        report.append("#### Similarity Heatmap at Optimal Layer")
        report.append("")
        report.append(f"![{model} Heatmap](layer_analysis/{model}_heatmap.png)")
        report.append("")
        
        # 最优层维度相似度表格
        report.append("#### Virtue-Vice Similarity at Optimal Layer")
        report.append("")
        report.append("| Dimension | Cosine Similarity |")
        report.append("|-----------|-------------------|")
        for dim_name, sim in result['optimal_layer_stats']['dimension_sims'].items():
            report.append(f"| {dim_name} | {sim:.4f} |")
        report.append("")
        
        report.append("---")
        report.append("")
    
    # 关键发现
    report.append("## 6. Key Findings")
    report.append("")
    
    # 计算一些统计信息
    valid_results = {k: v for k, v in all_results.items() if v['status'] == 'success'}
    if valid_results:
        rel_positions = [v['optimal_layer'] / v['num_layers'] for v in valid_results.values()]
        avg_rel_pos = np.mean(rel_positions)
        
        report.append(f"1. **Average Optimal Layer Position:** {avg_rel_pos*100:.1f}% of model depth")
        report.append("")
        
        if avg_rel_pos > 0.5:
            report.append("   The optimal layers tend to be in the **later** parts of the models, suggesting that higher-level representations provide better category separation.")
        else:
            report.append("   The optimal layers tend to be in the **earlier** parts of the models, suggesting that lower-level representations already encode distinct moral concepts.")
        report.append("")
        
        # 比较Instruct vs Base
        instruct_models = [k for k in valid_results.keys() if 'Base' not in k and 'Guard' not in k]
        base_models = [k for k in valid_results.keys() if 'Base' in k]
        
        if instruct_models and base_models:
            instruct_sims = [valid_results[k]['optimal_layer_stats']['mean_sim'] for k in instruct_models]
            base_sims = [valid_results[k]['optimal_layer_stats']['mean_sim'] for k in base_models]
            
            report.append(f"2. **Instruct vs Base Models:**")
            report.append(f"   - Instruct models average similarity: {np.mean(instruct_sims):.4f}")
            report.append(f"   - Base models average similarity: {np.mean(base_sims):.4f}")
            if np.mean(instruct_sims) < np.mean(base_sims):
                report.append("   - Instruct-tuned models show **better** category separation than base models.")
            else:
                report.append("   - Base models show comparable or better category separation.")
            report.append("")
    
    report.append("")
    report.append("---")
    report.append("")
    report.append("*This report was automatically generated by the layer analysis script.*")
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"Report saved to: {output_path}")


# ============== 主函数 ==============

def process_single_model(model_name):
    """处理单个模型"""
    print(f"\n处理模型: {model_name}")
    
    # 加载数据
    centers = load_category_centers(model_name)
    if centers is None:
        print(f"  未找到范畴中心文件，跳过")
        return {'status': 'skipped'}
    
    # 计算每层统计信息
    layer_stats = compute_layer_statistics(centers, 'mean_pooling')
    
    # 找到最优层
    optimal_layer = find_optimal_layer(layer_stats, 'mean')
    optimal_stats = layer_stats[optimal_layer]
    
    print(f"  总层数: {centers['num_layers']}, 隐藏维度: {centers['hidden_dim']}")
    print(f"  最优层: {optimal_layer} (相对位置: {optimal_layer/centers['num_layers']*100:.1f}%)")
    print(f"  最优层平均相似度: {optimal_stats['mean_sim']:.4f}")
    
    # 生成可视化
    ensure_dir(OUTPUT_DIR)
    
    # 1. 层相似度曲线
    curve_path = os.path.join(OUTPUT_DIR, f'{model_name}_layer_curve.png')
    plot_layer_similarity_curve(layer_stats, model_name, curve_path)
    print(f"  层相似度曲线: {curve_path}")
    
    # 2. 维度相似度曲线
    dim_curve_path = os.path.join(OUTPUT_DIR, f'{model_name}_dimension_curve.png')
    plot_dimension_similarity_curve(layer_stats, model_name, dim_curve_path)
    print(f"  维度相似度曲线: {dim_curve_path}")
    
    # 3. 最优层热力图
    heatmap_path = os.path.join(OUTPUT_DIR, f'{model_name}_heatmap.png')
    plot_similarity_heatmap(optimal_stats['sim_matrix'], optimal_stats['categories'], 
                           model_name, optimal_layer, heatmap_path)
    print(f"  相似度热力图: {heatmap_path}")
    
    return {
        'status': 'success',
        'num_layers': centers['num_layers'],
        'hidden_dim': centers['hidden_dim'],
        'optimal_layer': optimal_layer,
        'optimal_layer_stats': {
            'mean_sim': optimal_stats['mean_sim'],
            'std_sim': optimal_stats['std_sim'],
            'dimension_sims': optimal_stats['dimension_sims']
        },
        'layer_stats': layer_stats
    }


def main(model_names):
    """主函数"""
    print("=" * 60)
    print("范畴中心向量层级余弦相似度分析")
    print("=" * 60)
    
    ensure_dir(OUTPUT_DIR)
    
    all_results = {}
    
    # 处理每个模型
    for model_name in model_names:
        result = process_single_model(model_name)
        all_results[model_name] = result
    
    # 生成跨模型可视化
    print("\n生成跨模型可视化...")
    
    # 1. 跨模型比较图
    comparison_path = os.path.join(OUTPUT_DIR, 'cross_model_comparison.png')
    plot_cross_model_comparison(all_results, comparison_path)
    print(f"  跨模型比较: {comparison_path}")
    
    # 2. 最优层分布图
    distribution_path = os.path.join(OUTPUT_DIR, 'optimal_layer_distribution.png')
    plot_optimal_layer_distribution(all_results, distribution_path)
    print(f"  最优层分布: {distribution_path}")
    
    # 3. 所有模型层曲线
    all_curves_path = os.path.join(OUTPUT_DIR, 'all_models_layer_curves.png')
    plot_all_models_layer_curves(all_results, all_curves_path)
    print(f"  所有模型曲线: {all_curves_path}")
    
    # 生成Markdown报告
    report_path = os.path.join(OUTPUT_DIR, 'layer_analysis_report.md')
    generate_markdown_report(all_results, report_path)
    
    # 保存最优层数据
    optimal_data = {}
    for model, result in all_results.items():
        if result['status'] == 'success':
            optimal_layer = result['optimal_layer']
            optimal_data[model] = {
                'optimal_layer': optimal_layer,
                'num_layers': result['num_layers'],
                'relative_position': optimal_layer / result['num_layers'],
                'mean_similarity': result['optimal_layer_stats']['mean_sim'],
                'dimension_similarities': result['optimal_layer_stats']['dimension_sims']
            }
    
    optimal_data_path = os.path.join(OUTPUT_DIR, 'optimal_layer_data.json')
    with open(optimal_data_path, 'w', encoding='utf-8') as f:
        json.dump(optimal_data, f, indent=2, ensure_ascii=False)
    print(f"\n最优层数据已保存: {optimal_data_path}")
    
    # 打印总结
    success_count = sum(1 for r in all_results.values() if r['status'] == 'success')
    skip_count = sum(1 for r in all_results.values() if r['status'] == 'skipped')
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print(f"  成功: {success_count}")
    print(f"  跳过: {skip_count}")
    print(f"  报告: {report_path}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze and plot layer-wise moral category centers.'
    )
    parser.add_argument('--centers-dir', type=Path, default=CATEGORY_CENTERS_DIR)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--models', nargs='*', default=None)
    args = parser.parse_args()

    CATEGORY_CENTERS_DIR = args.centers_dir
    OUTPUT_DIR = args.output_dir
    models = args.models or sorted(
        path.name for path in args.centers_dir.iterdir()
        if path.is_dir() and (path / 'category_centers.pt').is_file()
    ) if args.centers_dir.exists() else []
    if not models:
        raise SystemExit(f'No category centers found under {args.centers_dir}')
    main(models)
