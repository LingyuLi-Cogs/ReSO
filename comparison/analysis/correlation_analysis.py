import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import seaborn as sns
from collections import defaultdict

# Portable artifact locations.
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / 'comparison' / 'artifacts'
base_path = ARTIFACTS_DIR / 'category_centers' / 'similarities'
output_path = ARTIFACTS_DIR / 'analysis' / 'correlations'

# 维度定义
dimensions = [
    'care-harm',
    'fairness-cheating', 
    'loyalty-betrayal',
    'authority-subversion',
    'sanctity-degradation'
]

dim_to_indices = {
    'care-harm': (0, 1),
    'fairness-cheating': (2, 3),
    'loyalty-betrayal': (4, 5),
    'authority-subversion': (6, 7),
    'sanctity-degradation': (8, 9)
}

# 范畴方向：10个范畴（5个维度 × 2个方向）
category_directions = []
for dim in dimensions:
    category_directions.append(f"{dim}_virtue")
    category_directions.append(f"{dim}_vice")

# 为10个范畴+overall定义颜色
CATEGORY_COLORS = {
    'overall': '#000000',  # 黑色，加粗
    'care-harm_virtue': '#2ecc71',      # 绿色系
    'care-harm_vice': '#e74c3c',        # 红色系
    'fairness-cheating_virtue': '#3498db',   # 蓝色系
    'fairness-cheating_vice': '#e67e22',     # 橙色系
    'loyalty-betrayal_virtue': '#9b59b6',    # 紫色系
    'loyalty-betrayal_vice': '#f39c12',      # 黄色系
    'authority-subversion_virtue': '#1abc9c', # 青色系
    'authority-subversion_vice': '#e91e63',   # 粉色系
    'sanctity-degradation_virtue': '#00bcd4', # 天蓝色
    'sanctity-degradation_vice': '#795548',   # 棕色系
}

# 简化的范畴标签（用于图例）
CATEGORY_LABELS = {
    'overall': 'Overall',
    'care-harm_virtue': 'Care',
    'care-harm_vice': 'Harm',
    'fairness-cheating_virtue': 'Fairness',
    'fairness-cheating_vice': 'Cheating',
    'loyalty-betrayal_virtue': 'Loyalty',
    'loyalty-betrayal_vice': 'Betrayal',
    'authority-subversion_virtue': 'Authority',
    'authority-subversion_vice': 'Subversion',
    'sanctity-degradation_virtue': 'Sanctity',
    'sanctity-degradation_vice': 'Degradation',
}


def load_similarities(model_name):
    """加载模型的相似度数据"""
    file_path = base_path / f"{model_name}_similarities.pt"
    data = torch.load(file_path, map_location='cpu', weights_only=False)
    return data


def get_human_label(sample, direction):
    """
    获取人类标注的归属度
    direction: 如 'care-harm_virtue' 或 'care-harm_vice'
    """
    moral_vector = sample['moral_vector']
    dim = direction.rsplit('_', 1)[0]
    pole = direction.rsplit('_', 1)[1]
    
    virtue_idx, vice_idx = dim_to_indices[dim]
    
    if pole == 'virtue':
        return moral_vector[virtue_idx]
    else:
        return moral_vector[vice_idx]


def get_model_similarity(sample, direction, layer, pooling_type='mean_pooling'):
    """获取模型的相似度"""
    sample_type = sample['sample_type']
    pole = direction.rsplit('_', 1)[1]
    
    if sample_type.endswith('_typical'):
        sample_type = sample_type.replace('_typical', '')
    
    if sample_type == 'neutral':
        key = f"{pooling_type}_cos_sim_{pole}"
        if key in sample and layer in sample[key]:
            return sample[key][layer]
        return None
    else:
        key = f"{pooling_type}_cos_sim"
        if key in sample and layer in sample[key]:
            return sample[key][layer]
        return None


def prepare_data_for_analysis(data, pooling_type='mean_pooling'):
    """准备分析数据，返回结构化的DataFrame"""
    records = []
    
    for sample in data:
        sample_id = sample['id']
        sampled_dim = sample['sampled_dimension']
        sample_type = sample['sample_type']
        moral_vector = sample['moral_vector']
        
        base_sample_type = sample_type.replace('_typical', '') if sample_type.endswith('_typical') else sample_type
        
        if base_sample_type == 'neutral':
            directions_to_analyze = [f"{sampled_dim}_virtue", f"{sampled_dim}_vice"]
        else:
            directions_to_analyze = [f"{sampled_dim}_{base_sample_type}"]
        
        if base_sample_type == 'neutral':
            layers = list(sample.get(f'{pooling_type}_cos_sim_virtue', {}).keys())
        else:
            layers = list(sample.get(f'{pooling_type}_cos_sim', {}).keys())
        
        for direction in directions_to_analyze:
            human_label = get_human_label(sample, direction)
            
            for layer in layers:
                model_sim = get_model_similarity(sample, direction, layer, pooling_type)
                
                if model_sim is not None:
                    records.append({
                        'id': sample_id,
                        'layer': layer,
                        'direction': direction,
                        'dimension': direction.rsplit('_', 1)[0],
                        'pole': direction.rsplit('_', 1)[1],
                        'sample_type': base_sample_type,
                        'human_label': human_label,
                        'model_similarity': model_sim
                    })
    
    return pd.DataFrame(records)


def compute_correlations(df):
    """计算Pearson和Spearman相关系数及其统计量"""
    if len(df) < 3:
        return {
            'pearson_r': np.nan, 'pearson_p': np.nan, 
            'pearson_ci_low': np.nan, 'pearson_ci_high': np.nan,
            'spearman_r': np.nan, 'spearman_p': np.nan,
            'n': len(df)
        }
    
    x = df['human_label'].values
    y = df['model_similarity'].values
    
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    
    if len(x) < 3:
        return {
            'pearson_r': np.nan, 'pearson_p': np.nan,
            'pearson_ci_low': np.nan, 'pearson_ci_high': np.nan,
            'spearman_r': np.nan, 'spearman_p': np.nan,
            'n': len(x)
        }
    
    # Pearson相关
    pearson_r, pearson_p = stats.pearsonr(x, y)
    
    # Pearson置信区间（Fisher z变换）
    n = len(x)
    z = np.arctanh(pearson_r)
    se = 1 / np.sqrt(n - 3)
    ci_low = np.tanh(z - 1.96 * se)
    ci_high = np.tanh(z + 1.96 * se)
    
    # Spearman相关
    spearman_r, spearman_p = stats.spearmanr(x, y)
    
    return {
        'pearson_r': pearson_r, 
        'pearson_p': pearson_p,
        'pearson_ci_low': ci_low, 
        'pearson_ci_high': ci_high,
        'spearman_r': spearman_r, 
        'spearman_p': spearman_p,
        'n': n
    }


def get_dynamic_ylim(results, layers, corr_type):
    """根据相关系数的取值动态调整纵轴范围"""
    r_key = f'{corr_type}_r'
    all_values = []
    
    # 收集overall的值
    for layer in layers:
        if layer in results['overall']:
            val = results['overall'][layer].get(r_key, np.nan)
            if not np.isnan(val):
                all_values.append(val)
    
    # 收集各范畴的值
    for category in category_directions:
        if category in results['by_category']:
            for layer in layers:
                if layer in results['by_category'][category]:
                    val = results['by_category'][category][layer].get(r_key, np.nan)
                    if not np.isnan(val):
                        all_values.append(val)
    
    if not all_values:
        return -1.05, 1.05
    
    min_val = min(all_values)
    max_val = max(all_values)
    
    # 添加一些边距
    margin = (max_val - min_val) * 0.15
    if margin < 0.05:
        margin = 0.05
    
    y_min = min_val - margin
    y_max = max_val + margin
    
    # 确保0在范围内（如果数据跨越0）
    if min_val < 0 < max_val:
        pass  # 保持原来的范围
    elif min_val >= 0:
        y_min = max(-0.05, y_min)  # 稍微留一点负数空间
    elif max_val <= 0:
        y_max = min(0.05, y_max)  # 稍微留一点正数空间
    
    # 限制在[-1.1, 1.1]范围内
    y_min = max(-1.1, y_min)
    y_max = min(1.1, y_max)
    
    return y_min, y_max


def find_overall_peak(results, layers, corr_type):
    """找到overall的最高相关系数及其对应的层"""
    r_key = f'{corr_type}_r'
    best_layer = None
    best_value = -np.inf
    
    for layer in layers:
        if layer in results['overall']:
            val = results['overall'][layer].get(r_key, np.nan)
            if not np.isnan(val) and val > best_value:
                best_value = val
                best_layer = layer
    
    if best_layer is None:
        return None, np.nan
    
    return best_layer, best_value


def plot_correlation_by_layer(results, layers, model_name, pooling_type, corr_type, output_dir):
    """
    绘制逐层相关性变化图
    corr_type: 'pearson' 或 'spearman'
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    r_key = f'{corr_type}_r'
    
    # 动态计算纵轴范围
    y_min, y_max = get_dynamic_ylim(results, layers, corr_type)
    
    # 找到overall的峰值
    peak_layer, peak_value = find_overall_peak(results, layers, corr_type)
    
    # 绘制顺序：先绘制各范畴，最后绘制overall（使其在最上层）
    plot_order = category_directions + ['overall']
    
    overall_x_vals = []
    overall_y_vals = []
    
    for category in plot_order:
        if category == 'overall':
            layer_data = results['overall']
        else:
            layer_data = results['by_category'].get(category, {})
        
        if not layer_data:
            continue
        
        x_vals = []
        y_vals = []
        
        for layer in layers:
            if layer in layer_data:
                r_val = layer_data[layer].get(r_key, np.nan)
                if not np.isnan(r_val):
                    x_vals.append(layer)
                    y_vals.append(r_val)
        
        if x_vals:
            color = CATEGORY_COLORS.get(category, '#888888')
            label = CATEGORY_LABELS.get(category, category)
            linewidth = 3 if category == 'overall' else 1.5
            alpha = 1.0 if category == 'overall' else 0.7
            zorder = 10 if category == 'overall' else 1
            
            ax.plot(x_vals, y_vals, 
                   color=color, 
                   linewidth=linewidth, 
                   alpha=alpha,
                   label=label,
                   marker='o' if category == 'overall' else None,
                   markersize=4,
                   zorder=zorder)
            
            if category == 'overall':
                overall_x_vals = x_vals
                overall_y_vals = y_vals
    
    # 在overall的最高值处加入标记
    if peak_layer is not None and not np.isnan(peak_value):
        # 绘制峰值标记点
        ax.scatter([peak_layer], [peak_value], 
                  color='red', s=150, zorder=20, 
                  edgecolors='white', linewidths=2,
                  marker='*')
        
        # 添加标注文本
        # 根据峰值位置决定文本位置，避免遮挡
        text_offset_y = (y_max - y_min) * 0.08
        if peak_value > (y_max + y_min) / 2:
            text_y = peak_value - text_offset_y
            va = 'top'
        else:
            text_y = peak_value + text_offset_y
            va = 'bottom'
        
        ax.annotate(f'Peak: {peak_value:.4f}\n(Layer {peak_layer})',
                   xy=(peak_layer, peak_value),
                   xytext=(peak_layer, text_y),
                   fontsize=11,
                   fontweight='bold',
                   color='red',
                   ha='center',
                   va=va,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.9),
                   zorder=25)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel(f'{corr_type.capitalize()} Correlation (r)', fontsize=12)
    ax.set_title(f'{model_name} - {pooling_type}\n{corr_type.capitalize()} Correlation by Layer', fontsize=14)
    
    ax.set_xticks(layers)
    ax.set_xticklabels(layers, rotation=45 if len(layers) > 20 else 0)
    
    # 设置动态纵轴范围
    ax.set_ylim(y_min, y_max)
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / f'{corr_type}_correlation_by_layer.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")
    
    return peak_layer, peak_value


def analyze_model(model_name, pooling_type='mean_pooling'):
    """对单个模型进行完整分析"""
    print(f"\n{'='*60}")
    print(f"Analyzing {model_name} ({pooling_type})")
    print('='*60)
    
    # 加载数据
    data = load_similarities(model_name)
    print(f"Loaded {len(data)} samples")
    
    # 准备分析数据
    df = prepare_data_for_analysis(data, pooling_type)
    print(f"Prepared {len(df)} records for analysis")
    
    # 获取所有层
    layers = sorted(df['layer'].unique())
    print(f"Layers: {layers}")
    
    # 创建输出目录
    model_output_path = output_path / model_name / pooling_type
    model_output_path.mkdir(parents=True, exist_ok=True)
    
    # 存储结果
    all_results = {
        'overall': {},
        'by_category': defaultdict(dict)
    }
    
    for layer in layers:
        layer_df = df[df['layer'] == layer]
        
        # 整体分析
        overall_corr = compute_correlations(layer_df)
        all_results['overall'][layer] = overall_corr
        
        # 各范畴分析
        for direction in category_directions:
            dir_df = layer_df[layer_df['direction'] == direction]
            
            if len(dir_df) > 0:
                dir_corr = compute_correlations(dir_df)
                all_results['by_category'][direction][layer] = dir_corr
    
    # 保存结果
    results_save_path = model_output_path / 'correlation_results.pt'
    torch.save(all_results, results_save_path)
    print(f"Results saved to {results_save_path}")
    
    # 生成两张图：Pearson和Spearman，并获取峰值信息
    peak_info = {}
    peak_layer_pearson, peak_value_pearson = plot_correlation_by_layer(
        all_results, layers, model_name, pooling_type, 'pearson', model_output_path)
    peak_layer_spearman, peak_value_spearman = plot_correlation_by_layer(
        all_results, layers, model_name, pooling_type, 'spearman', model_output_path)
    
    peak_info['pearson'] = {'layer': peak_layer_pearson, 'value': peak_value_pearson}
    peak_info['spearman'] = {'layer': peak_layer_spearman, 'value': peak_value_spearman}
    
    # 生成汇总CSV
    generate_summary_csv(all_results, layers, model_name, pooling_type, model_output_path)
    
    return all_results, peak_info


def generate_summary_csv(results, layers, model_name, pooling_type, output_dir):
    """生成汇总CSV"""
    rows = []
    
    # 整体结果
    for layer in layers:
        r = results['overall'].get(layer, {})
        rows.append({
            'model': model_name,
            'pooling_type': pooling_type,
            'layer': layer,
            'category': 'overall',
            'pearson_r': r.get('pearson_r', np.nan),
            'pearson_p': r.get('pearson_p', np.nan),
            'pearson_ci_low': r.get('pearson_ci_low', np.nan),
            'pearson_ci_high': r.get('pearson_ci_high', np.nan),
            'spearman_r': r.get('spearman_r', np.nan),
            'spearman_p': r.get('spearman_p', np.nan),
            'n': r.get('n', 0)
        })
    
    # 各范畴结果
    for direction in category_directions:
        for layer in layers:
            r = results['by_category'].get(direction, {}).get(layer, {})
            rows.append({
                'model': model_name,
                'pooling_type': pooling_type,
                'layer': layer,
                'category': direction,
                'pearson_r': r.get('pearson_r', np.nan),
                'pearson_p': r.get('pearson_p', np.nan),
                'pearson_ci_low': r.get('pearson_ci_low', np.nan),
                'pearson_ci_high': r.get('pearson_ci_high', np.nan),
                'spearman_r': r.get('spearman_r', np.nan),
                'spearman_p': r.get('spearman_p', np.nan),
                'n': r.get('n', 0)
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'correlation_summary.csv', index=False)
    print(f"Summary CSV saved to {output_dir / 'correlation_summary.csv'}")


def plot_peak_correlation_comparison(all_peak_info, corr_type, output_dir):
    """
    绘制所有模型overall最高相关性的柱状图
    corr_type: 'pearson' 或 'spearman'
    """
    # 准备数据
    models = []
    mean_pooling_values = []
    mean_pooling_layers = []
    last_token_values = []
    last_token_layers = []
    
    for model_name, pooling_info in all_peak_info.items():
        models.append(model_name)
        
        # mean_pooling
        mp_info = pooling_info.get('mean_pooling', {}).get(corr_type, {})
        mean_pooling_values.append(mp_info.get('value', np.nan))
        mean_pooling_layers.append(mp_info.get('layer', 'N/A'))
        
        # last_token
        lt_info = pooling_info.get('last_token', {}).get(corr_type, {})
        last_token_values.append(lt_info.get('value', np.nan))
        last_token_layers.append(lt_info.get('layer', 'N/A'))
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(max(14, len(models) * 1.2), 8))
    
    x = np.arange(len(models))
    width = 0.35
    
    # 绘制柱状图
    bars1 = ax.bar(x - width/2, mean_pooling_values, width, label='Mean Pooling', 
                   color='#3498db', edgecolor='white', linewidth=1)
    bars2 = ax.bar(x + width/2, last_token_values, width, label='Last Token', 
                   color='#e74c3c', edgecolor='white', linewidth=1)
    
    # 在柱子上添加数值和层数标注
    def add_labels(bars, values, layers):
        for bar, val, layer in zip(bars, values, layers):
            if not np.isnan(val):
                height = bar.get_height()
                # 数值标注
                ax.annotate(f'{val:.4f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=9, fontweight='bold')
                # 层数标注（在柱子内部）
                ax.annotate(f'L{layer}',
                           xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                           ha='center', va='center',
                           fontsize=8, color='white', fontweight='bold')
    
    add_labels(bars1, mean_pooling_values, mean_pooling_layers)
    add_labels(bars2, last_token_values, last_token_layers)
    
    # 设置图形属性
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(f'Peak {corr_type.capitalize()} Correlation', fontsize=12)
    ax.set_title(f'Peak {corr_type.capitalize()} Correlation (Overall) by Model', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 动态调整纵轴范围
    all_values = [v for v in mean_pooling_values + last_token_values if not np.isnan(v)]
    if all_values:
        max_val = max(all_values)
        min_val = min(all_values)
        margin = (max_val - min_val) * 0.2
        if margin < 0.05:
            margin = 0.1
        ax.set_ylim(max(0, min_val - margin), min(1.0, max_val + margin + 0.05))
    
    plt.tight_layout()
    save_path = output_dir / f'peak_{corr_type}_correlation_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")


def generate_peak_summary_csv(all_peak_info, output_dir):
    """生成峰值汇总CSV"""
    rows = []
    
    for model_name, pooling_info in all_peak_info.items():
        for pooling_type in ['mean_pooling', 'last_token']:
            if pooling_type in pooling_info:
                for corr_type in ['pearson', 'spearman']:
                    info = pooling_info[pooling_type].get(corr_type, {})
                    rows.append({
                        'model': model_name,
                        'pooling_type': pooling_type,
                        'correlation_type': corr_type,
                        'peak_layer': info.get('layer', np.nan),
                        'peak_value': info.get('value', np.nan)
                    })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'peak_correlation_summary.csv', index=False)
    print(f"Peak summary CSV saved to {output_dir / 'peak_correlation_summary.csv'}")


def analyze_all_models(model_names):
    """批量分析所有模型"""
    all_model_results = {}
    all_peak_info = {}
    
    for model_name in model_names:
        print(f"\n{'#'*60}")
        print(f"Processing: {model_name}")
        print('#'*60)
        
        try:
            results_mean, peak_info_mean = analyze_model(model_name, pooling_type='mean_pooling')
            results_last, peak_info_last = analyze_model(model_name, pooling_type='last_token')
            
            all_model_results[model_name] = {
                'mean_pooling': results_mean,
                'last_token': results_last
            }
            
            all_peak_info[model_name] = {
                'mean_pooling': peak_info_mean,
                'last_token': peak_info_last
            }
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 生成跨模型比较图
    if all_peak_info:
        print(f"\n{'='*60}")
        print("Generating cross-model comparison plots...")
        print('='*60)
        
        # Pearson峰值比较柱状图
        plot_peak_correlation_comparison(all_peak_info, 'pearson', output_path)
        
        # Spearman峰值比较柱状图
        plot_peak_correlation_comparison(all_peak_info, 'spearman', output_path)
        
        # 生成峰值汇总CSV
        generate_peak_summary_csv(all_peak_info, output_path)
    
    return all_model_results, all_peak_info


# 主函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Correlate model category similarity with human labels.'
    )
    parser.add_argument('--similarities-dir', type=Path, default=base_path)
    parser.add_argument('--output-dir', type=Path, default=output_path)
    parser.add_argument('--models', nargs='*', default=None,
                        help='Model names; discovers *_similarities.pt if omitted')
    args = parser.parse_args()

    base_path = args.similarities_dir
    output_path = args.output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    model_names = args.models or sorted(
        path.name.removesuffix('_similarities.pt')
        for path in base_path.glob('*_similarities.pt')
    )
    if not model_names:
        raise SystemExit(f'No similarity files found under {base_path}')

    all_results, all_peak_info = analyze_all_models(model_names)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    
    # 打印峰值汇总
    print("\n" + "="*60)
    print("Peak Correlation Summary:")
    print("="*60)
    for model_name, pooling_info in all_peak_info.items():
        print(f"\n{model_name}:")
        for pooling_type in ['mean_pooling', 'last_token']:
            if pooling_type in pooling_info:
                print(f"  {pooling_type}:")
                for corr_type in ['pearson', 'spearman']:
                    info = pooling_info[pooling_type].get(corr_type, {})
                    print(f"    {corr_type}: r={info.get('value', np.nan):.4f} @ Layer {info.get('layer', 'N/A')}")
                    
