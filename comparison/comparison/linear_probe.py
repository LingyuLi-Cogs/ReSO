#!/usr/bin/env python3
"""
Linear Probe Analysis for Moral Representation in LLMs
训练线性探针分析模型是否内在表征了道德区分
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Portable defaults. Both directories contain generated artifacts and are ignored.
REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = REPO_ROOT / 'comparison' / 'artifacts' / 'activations'
OUTPUT_BASE = REPO_ROOT / 'comparison' / 'artifacts' / 'linear_probe'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 训练超参数
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
WEIGHT_DECAY = 1e-4
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # 从训练集中划分
RANDOM_SEED = 42


class LinearProbe(nn.Module):
    """简单的线性探针模型"""
    def __init__(self, input_dim, output_dim=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)


def load_activations(model_path):
    """加载模型激活数据"""
    merged = model_path / 'activations_merged.pt'
    pt_files = [merged] if merged.exists() else sorted(model_path.glob('batch_*.pt'))
    if not pt_files:
        print(f"No .pt files found in {model_path}")
        return None
    
    all_data = []
    for pt_file in pt_files:
        data = torch.load(pt_file, map_location='cpu', weights_only=False)
        if isinstance(data, list):
            all_data.extend(data)
        else:
            all_data.append(data)
    
    return all_data


def prepare_data_for_layer(data, layer_idx, pooling_type='mean_pooling'):
    """为特定层准备训练数据"""
    X_list = []
    y_list = []
    
    for sample in data:
        # 获取moral_vector作为目标
        moral_vector = sample.get('moral_vector')
        if moral_vector is None:
            continue
        
        # 获取激活
        activations = sample.get(pooling_type)
        if activations is None or layer_idx not in activations:
            continue
        
        activation = activations[layer_idx]
        
        # 转换为numpy array
        if isinstance(activation, torch.Tensor):
            activation = activation.cpu().numpy()
        if isinstance(moral_vector, torch.Tensor):
            moral_vector = moral_vector.cpu().numpy()
        
        activation = np.array(activation).flatten()
        moral_vector = np.array(moral_vector).flatten()
        
        if len(moral_vector) != 10:
            continue
            
        X_list.append(activation)
        y_list.append(moral_vector)
    
    if len(X_list) == 0:
        return None, None
    
    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    
    return X, y


def train_probe(X_train, y_train, X_val, y_val, input_dim, device):
    """训练线性探针"""
    model = LinearProbe(input_dim, output_dim=10).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 转换为tensor
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        # 训练
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        train_loss /= len(train_dataset)
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                break
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model


def evaluate_probe(model, X_test, y_test, device):
    """评估探针性能，返回R2分数"""
    model.eval()
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        predictions = model(X_test_t).cpu().numpy()
    
    # 计算整体R2
    overall_r2 = r2_score(y_test, predictions)
    
    # 计算每个维度的R2
    dim_r2 = []
    dim_names = ['care', 'harm', 'fairness', 'cheating', 'loyalty', 
                 'betrayal', 'authority', 'subversion', 'sanctity', 'degradation']
    for i in range(10):
        if np.std(y_test[:, i]) > 1e-6:  # 避免常数列
            r2 = r2_score(y_test[:, i], predictions[:, i])
        else:
            r2 = 0.0
        dim_r2.append(r2)
    
    return overall_r2, dict(zip(dim_names, dim_r2))


def process_model(model_folder, model_name, device):
    """处理单个模型的所有层"""
    model_path = BASE_PATH / model_folder
    if not model_path.exists():
        print(f"Path not found: {model_path}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Processing: {model_name}")
    print(f"{'='*60}")
    
    # 加载数据
    data = load_activations(model_path)
    if data is None or len(data) == 0:
        print(f"No data loaded for {model_name}")
        return None
    
    print(f"Loaded {len(data)} samples")
    
    # 获取所有可用的层
    sample = data[0]
    available_layers_mean = sorted(sample.get('mean_pooling', {}).keys())
    available_layers_last = sorted(sample.get('last_token', {}).keys())
    
    print(f"Available layers (mean_pooling): {available_layers_mean}")
    print(f"Available layers (last_token): {available_layers_last}")
    
    results = {
        'model_name': model_name,
        'mean_pooling': {'layers': [], 'overall_r2': [], 'dim_r2': []},
        'last_token': {'layers': [], 'overall_r2': [], 'dim_r2': []}
    }
    
    # 处理两种pooling方式
    for pooling_type, available_layers in [('mean_pooling', available_layers_mean), 
                                            ('last_token', available_layers_last)]:
        print(f"\n--- Processing {pooling_type} ---")
        
        output_dir = OUTPUT_BASE / pooling_type / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for layer_idx in tqdm(available_layers, desc=f"{pooling_type}"):
            # 准备数据
            X, y = prepare_data_for_layer(data, layer_idx, pooling_type)
            if X is None or len(X) < 100:
                print(f"Insufficient data for layer {layer_idx}")
                continue
            
            # 划分训练/验证/测试集
            X_trainval, X_test, y_trainval, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval, y_trainval, test_size=VAL_SIZE/(1-TEST_SIZE),
                random_state=RANDOM_SEED
            )
            
            input_dim = X_train.shape[1]
            
            # 训练探针
            model = train_probe(X_train, y_train, X_val, y_val, input_dim, device)
            
            # 评估
            overall_r2, dim_r2 = evaluate_probe(model, X_test, y_test, device)
            
            # 保存结果
            results[pooling_type]['layers'].append(layer_idx)
            results[pooling_type]['overall_r2'].append(overall_r2)
            results[pooling_type]['dim_r2'].append(dim_r2)
            
            # 保存模型
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': input_dim,
                'layer_idx': layer_idx,
                'overall_r2': overall_r2,
                'dim_r2': dim_r2
            }, output_dir / f'probe_layer_{layer_idx}.pt')
    
    # 保存汇总结果
    results_path = OUTPUT_BASE / f'{model_name}_results.json'
    
    # 转换numpy类型为python原生类型以便JSON序列化
    results_serializable = {
        'model_name': results['model_name'],
        'mean_pooling': {
            'layers': [int(l) for l in results['mean_pooling']['layers']],
            'overall_r2': [float(r) for r in results['mean_pooling']['overall_r2']],
            'dim_r2': results['mean_pooling']['dim_r2']
        },
        'last_token': {
            'layers': [int(l) for l in results['last_token']['layers']],
            'overall_r2': [float(r) for r in results['last_token']['overall_r2']],
            'dim_r2': results['last_token']['dim_r2']
        }
    }
    
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    return results


def plot_results(results, model_name):
    """为单个模型绘制R2变化图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Linear Probe R² Analysis: {model_name}', fontsize=14, fontweight='bold')
    
    dim_names = ['care', 'harm', 'fairness', 'cheating', 'loyalty', 
                 'betrayal', 'authority', 'subversion', 'sanctity', 'degradation']
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for idx, pooling_type in enumerate(['mean_pooling', 'last_token']):
        data = results[pooling_type]
        if len(data['layers']) == 0:
            continue
        
        layers = data['layers']
        overall_r2 = data['overall_r2']
        dim_r2_list = data['dim_r2']
        
        # 左图：整体R2
        ax1 = axes[idx, 0]
        ax1.plot(layers, overall_r2, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Layer', fontsize=11)
        ax1.set_ylabel('Overall R²', fontsize=11)
        ax1.set_title(f'{pooling_type.replace("_", " ").title()} - Overall R²', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([min(0, min(overall_r2) - 0.05), max(overall_r2) + 0.05])
        
        # 标注最大值
        max_idx = np.argmax(overall_r2)
        ax1.annotate(f'Max: {overall_r2[max_idx]:.3f}\n(Layer {layers[max_idx]})',
                    xy=(layers[max_idx], overall_r2[max_idx]),
                    xytext=(10, -20), textcoords='offset points',
                    fontsize=9, ha='left',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        
        # 右图：各维度R2
        ax2 = axes[idx, 1]
        for i, dim_name in enumerate(dim_names):
            dim_values = [d[dim_name] for d in dim_r2_list]
            ax2.plot(layers, dim_values, color=colors[i], label=dim_name, 
                    linewidth=1.5, alpha=0.8)
        
        ax2.set_xlabel('Layer', fontsize=11)
        ax2.set_ylabel('R² per Dimension', fontsize=11)
        ax2.set_title(f'{pooling_type.replace("_", " ").title()} - Per-Dimension R²', fontsize=12)
        ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    fig_path = OUTPUT_BASE / f'{model_name}_r2_analysis.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {fig_path}")


def plot_summary(all_results):
    """绘制所有模型的对比汇总图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Linear Probe R² Summary Across Models', fontsize=14, fontweight='bold')
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_results)))
    
    for idx, pooling_type in enumerate(['mean_pooling', 'last_token']):
        ax = axes[idx]
        
        for i, results in enumerate(all_results):
            if results is None:
                continue
            
            model_name = results['model_name']
            data = results[pooling_type]
            
            if len(data['layers']) == 0:
                continue
            
            # 归一化层位置到[0, 1]以便跨模型比较
            layers = np.array(data['layers'])
            max_layer = max(layers) if len(layers) > 0 else 1
            normalized_layers = layers / max_layer
            
            ax.plot(normalized_layers, data['overall_r2'], 
                   color=colors[i], label=model_name, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Relative Layer Position', fontsize=11)
        ax.set_ylabel('Overall R²', fontsize=11)
        ax.set_title(f'{pooling_type.replace("_", " ").title()}', fontsize=12)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig_path = OUTPUT_BASE / 'summary_r2_comparison.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved summary figure to {fig_path}")


def discover_model_configs(base_path, specs):
    """Return ``(folder, display_name)`` pairs from CLI specs or the filesystem."""
    if specs:
        configs = []
        for spec in specs:
            if "=" in spec:
                model_name, folder = spec.split("=", 1)
            else:
                folder = spec
                model_name = Path(folder).name.removeprefix("activations_")
            configs.append((folder, model_name))
        return configs

    configs = []
    if base_path.exists():
        for folder in sorted(path for path in base_path.iterdir() if path.is_dir()):
            has_data = (folder / "activations_merged.pt").exists()
            has_data = has_data or any(folder.glob("batch_*.pt"))
            if has_data:
                configs.append(
                    (folder.name, folder.name.removeprefix("activations_"))
                )
    return configs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train layer-wise linear probes on extracted activations."
    )
    parser.add_argument("--activations-root", type=Path, default=BASE_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_BASE)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME=FOLDER",
        help="Model display name and folder under --activations-root; repeatable",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--val-size", type=float, default=VAL_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main():
    """Run probes for explicit models or every discovered activation folder."""
    global BASE_PATH, OUTPUT_BASE, DEVICE
    global BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE
    global WEIGHT_DECAY, TEST_SIZE, VAL_SIZE, RANDOM_SEED

    args = parse_args()
    BASE_PATH = args.activations_root
    OUTPUT_BASE = args.output_dir
    DEVICE = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    print(f"Using device: {DEVICE}")
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    NUM_EPOCHS = args.epochs
    EARLY_STOPPING_PATIENCE = args.patience
    WEIGHT_DECAY = args.weight_decay
    TEST_SIZE = args.test_size
    VAL_SIZE = args.val_size
    RANDOM_SEED = args.seed
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    model_configs = discover_model_configs(BASE_PATH, args.model)
    if not model_configs:
        raise SystemExit(
            f"No activation folders found under {BASE_PATH}; pass --model NAME=FOLDER"
        )

    # 创建输出目录
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    (OUTPUT_BASE / 'mean_pooling').mkdir(exist_ok=True)
    (OUTPUT_BASE / 'last_token').mkdir(exist_ok=True)
    
    all_results = []
    
    for model_folder, model_name in model_configs:
        try:
            results = process_model(model_folder, model_name, DEVICE)
            if results is not None:
                plot_results(results, model_name)
                all_results.append(results)
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 绘制汇总图
    if len(all_results) > 0:
        plot_summary(all_results)
    
    print("\n" + "="*60)
    print("Processing complete!")
    print(f"Results saved to {OUTPUT_BASE}")
    print("="*60)


if __name__ == '__main__':
    main()
