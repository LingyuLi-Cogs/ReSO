import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ================= 全局绘图设置 =================

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# ================= 数据配置 =================

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANALYSIS_DIR = (
    REPO_ROOT / 'comparison' / 'artifacts' / 'analysis' / 'correlations'
)
ALL_MODELS = []

# 目标范畴顺序
TARGET_ORDER = [
    'care', 'harm', 
    'fairness', 'cheating', 
    'loyalty', 'betrayal', 
    'authority', 'subversion', 
    'sanctity', 'degradation'
]

# 范畴名称映射
CATEGORY_MAPPING = {
    'care-harm_virtue': 'care',
    'care-harm_vice':   'harm',
    'fairness-cheating_virtue': 'fairness',
    'fairness-cheating_vice':   'cheating',
    'loyalty-betrayal_virtue': 'loyalty',
    'loyalty-betrayal_vice':   'betrayal',
    'authority-subversion_virtue': 'authority',
    'authority-subversion_vice':   'subversion',
    'sanctity-degradation_virtue': 'sanctity',
    'sanctity-degradation_vice':   'degradation'
}

# ================= 功能函数 =================

def get_model_color_map():
    """Assign stable colors in the order supplied on the command line."""
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(ALL_MODELS), 1)))
    return dict(zip(ALL_MODELS, colors))

def load_data(base_path, model_names):
    all_data = []
    pooling = 'mean_pooling' 
    
    for model in model_names:
        csv_path = Path(base_path) / model / pooling / "correlation_summary.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                df['model'] = model
                all_data.append(df)
            except Exception:
                pass
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None

def clean_and_filter_categories(df):
    df = df.copy()
    df['clean_category'] = df['category'].map(CATEGORY_MAPPING)
    df = df.dropna(subset=['clean_category'])
    return df

def get_peak_spearman(df):
    peak_data = df.groupby(['model', 'clean_category'])['spearman_r'].max().reset_index()
    return peak_data

def plot_spearman_bars(peak_data, save_path=None):
    categories = TARGET_ORDER
    n_categories = len(categories)
    n_models = len(ALL_MODELS)
    color_map = get_model_color_map()
    
    fig, ax = plt.subplots(figsize=(24, 10))
    
    total_width = 0.8
    bar_width = total_width / n_models
    x_base = np.arange(n_categories)
    
    for i, model in enumerate(ALL_MODELS):
        model_data = peak_data[peak_data['model'] == model]
        
        values = []
        for cat in categories:
            val = model_data[model_data['clean_category'] == cat]['spearman_r'].values
            values.append(val[0] if len(val) > 0 else 0)
        
        positions = x_base + (i - n_models/2 + 0.5) * bar_width
        
        ax.bar(positions, values, bar_width, 
               label=model, 
               color=color_map.get(model, 'grey'), 
               edgecolor='white', linewidth=0.5)

    # 设置标签 (移除 bold, 使用默认字体)
    ax.set_ylabel('Peak Spearman Correlation', fontsize=22)
    ax.set_title('', fontsize=20, pad=20)
    
    # 设置 X 轴
    ax.set_xticks(x_base)
    xticklabels = [cat.capitalize() for cat in categories]
    ax.set_xticklabels(xticklabels, rotation=20, fontsize=22)
    
    # 设置 Y 轴范围 [0, 1]
    ax.set_ylim(0, 1)
    ax.tick_params(axis='y', labelsize=22)
    
    # 网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    # 图例
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10, 
              title="Models", title_fontsize=25, frameon=False, ncol=2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"图表已保存: {save_path}")
    
    plt.close()

# ================= 主程序 =================

def main():
    global ALL_MODELS
    parser = argparse.ArgumentParser(
        description='Plot peak human/model correlations across models.'
    )
    parser.add_argument('--analysis-dir', type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--models', nargs='*', default=None)
    args = parser.parse_args()

    base_path = args.analysis_dir
    output_dir = args.output_dir or (base_path / 'plots')
    output_dir.mkdir(exist_ok=True, parents=True)
    discovered = (
        [
            path.name for path in base_path.iterdir()
            if path.is_dir()
            and (path / 'mean_pooling' / 'correlation_summary.csv').is_file()
        ]
        if base_path.exists()
        else []
    )
    ALL_MODELS = args.models or sorted(discovered)
    if not ALL_MODELS:
        raise SystemExit(f'No correlation summaries found under {base_path}')
    
    print("正在加载数据...")
    raw_df = load_data(base_path, ALL_MODELS)
    
    if raw_df is None:
        print("错误：未找到数据")
        return

    clean_df = clean_and_filter_categories(raw_df)
    peak_df = get_peak_spearman(clean_df)
    
    print("正在绘图...")
    save_file = output_dir / "spearman_peak_final_style.png"
    plot_spearman_bars(peak_df, save_path=save_file)
    print("完成")

if __name__ == "__main__":
    main()
