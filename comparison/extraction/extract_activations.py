import json
import argparse
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = (
    REPO_ROOT
    / "dataset"
    / "comparison"
    / "sampled-human-representation-vectors.jsonl"
)
DEFAULT_ACTIVATIONS_DIR = REPO_ROOT / "comparison" / "artifacts" / "activations"


@dataclass
class ActionSample:
    id: str
    action: str
    rot_moral_foundations: str
    moral_vector: List[float]
    m_virtue: float
    m_vice: float
    sampled_dimension: str
    sample_type: str


@dataclass
class ExtractionResult:
    id: str
    moral_vector: List[float]
    sampled_dimension: str
    sample_type: str
    mean_pooling: Dict[int, List[float]]
    last_token: Dict[int, List[float]]


# ============================================================================
# Dataset
# ============================================================================

class MoralActionDataset(Dataset):
    """Dataset for moral action samples."""
    
    PROMPT_TEMPLATE = "{action} is morally"
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: List[ActionSample] = []
        
        self._load_data(data_path)
    
    def _load_data(self, data_path: str):
        """Load data from JSONL file."""
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    sample = ActionSample(
                        id=data['id'],
                        action=data['action'],
                        rot_moral_foundations=data['rot-moral-foundations'],
                        moral_vector=data['moral_vector'],
                        m_virtue=data['m_virtue'],
                        m_vice=data['m_vice'],
                        sampled_dimension=data['sampled_dimension'],
                        sample_type=data['sample_type']
                    )
                    self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} samples from {data_path}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[ActionSample, str]:
        sample = self.samples[idx]
        prompt = self.PROMPT_TEMPLATE.format(action=sample.action)
        return sample, prompt


def collate_fn(batch, tokenizer, max_length):
    """Custom collate function for batching."""
    samples, prompts = zip(*batch)
    
    # Tokenize with padding
    encodings = tokenizer(
        list(prompts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    return samples, encodings


# ============================================================================
# Activation Extraction
# ============================================================================

class ActivationExtractor:
    """Extract hidden states from specified layers using forward hooks."""
    
    def __init__(
        self,
        model: nn.Module,
        layers_to_extract: List[int],
        model_type: str = "qwen"
    ):
        self.model = model
        self.layers_to_extract = set(layers_to_extract)
        self.model_type = model_type
        self.activations: Dict[int, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        
        self._register_hooks()
    
    def _get_layer_modules(self) -> List[Tuple[int, nn.Module]]:
        """Get decoder layer modules based on model architecture."""
        # Support different model architectures
        if hasattr(self.model, 'model'):
            base_model = self.model.model
        else:
            base_model = self.model
        
        # Try different layer attribute names
        if hasattr(base_model, 'layers'):
            layers = base_model.layers
        elif hasattr(base_model, 'h'):
            layers = base_model.h
        elif hasattr(base_model, 'decoder') and hasattr(base_model.decoder, 'layers'):
            layers = base_model.decoder.layers
        else:
            raise ValueError(f"Cannot find decoder layers in model architecture")
        
        return list(enumerate(layers))
    
    def _register_hooks(self):
        """Register forward hooks on specified layers."""
        layer_modules = self._get_layer_modules()
        
        for layer_idx, layer_module in layer_modules:
            if layer_idx in self.layers_to_extract:
                hook = layer_module.register_forward_hook(
                    self._create_hook(layer_idx)
                )
                self.hooks.append(hook)
        
        print(f"Registered hooks on {len(self.hooks)} layers: {sorted(self.layers_to_extract)}")
    
    def _create_hook(self, layer_idx: int):
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            # Handle different output formats
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Store activations (detached, on CPU to save GPU memory)
            self.activations[layer_idx] = hidden_states.detach().cpu()
        
        return hook
    
    def clear(self):
        """Clear stored activations."""
        self.activations.clear()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def get_activations(self) -> Dict[int, torch.Tensor]:
        """Return current activations."""
        return self.activations


def compute_representations(
    activations: Dict[int, torch.Tensor],
    attention_mask: torch.Tensor
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Compute mean pooling and last token representations.
    
    Args:
        activations: Dict mapping layer_idx to hidden states [batch, seq_len, hidden_dim]
        attention_mask: Attention mask [batch, seq_len]
    
    Returns:
        mean_pooling: Dict mapping layer_idx to [batch, hidden_dim]
        last_token: Dict mapping layer_idx to [batch, hidden_dim]
    """
    attention_mask = attention_mask.cpu()
    mean_pooling = {}
    last_token = {}
    
    for layer_idx, hidden_states in activations.items():
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Mean pooling: average over non-padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
        token_counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
        mean_pooling[layer_idx] = sum_hidden / token_counts
        
        # Last token for either left- or right-padded batches.
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        last_token_indices = positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
        batch_indices = torch.arange(batch_size)
        last_token[layer_idx] = hidden_states[batch_indices, last_token_indices]
    
    return mean_pooling, last_token


# ============================================================================
# Main Extraction Logic
# ============================================================================

def get_total_layers(model: nn.Module) -> int:
    """Get the total number of layers in the model."""
    if hasattr(model, 'model'):
        base_model = model.model
    else:
        base_model = model
    
    if hasattr(base_model, 'layers'):
        return len(base_model.layers)
    elif hasattr(base_model, 'h'):
        return len(base_model.h)
    elif hasattr(base_model, 'decoder') and hasattr(base_model.decoder, 'layers'):
        return len(base_model.decoder.layers)
    else:
        raise ValueError("Cannot determine number of layers")


def load_model_and_tokenizer(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_impl: str = "auto",
    local_files_only: bool = True,
) -> Tuple[nn.Module, AutoTokenizer]:
    """Load model with automatic device mapping for multi-GPU."""
    
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side='left',
        local_files_only=local_files_only,
    )
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Loading model from {model_path}...")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "device_map": "auto",
        "local_files_only": local_files_only,
    }
    if attn_impl != "auto":
        model_kwargs["attn_implementation"] = attn_impl
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    
    model.eval()
    
    # Print device map
    if hasattr(model, 'hf_device_map'):
        print(f"Device map: {model.hf_device_map}")
    
    return model, tokenizer


def save_batch_results(
    results: List[ExtractionResult],
    output_dir: Path,
    batch_id: int
):
    """Save extraction results to disk."""
    output_file = output_dir / f"batch_{batch_id:06d}.pt"
    
    # Convert to serializable format
    serializable_results = []
    for result in results:
        item = {
            'id': result.id,
            'moral_vector': result.moral_vector,
            'sampled_dimension': result.sampled_dimension,
            'sample_type': result.sample_type,
            'mean_pooling': {k: v.float().numpy() for k, v in result.mean_pooling.items()},
            'last_token': {k: v.float().numpy() for k, v in result.last_token.items()}
        }
        serializable_results.append(item)
    
    torch.save(serializable_results, output_file)


def load_checkpoint(output_dir: Path) -> int:
    """Load checkpoint to resume extraction."""
    checkpoint_file = output_dir / "checkpoint.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        return checkpoint.get('last_processed_idx', -1)
    return -1


def save_checkpoint(output_dir: Path, last_processed_idx: int):
    """Save checkpoint for resuming."""
    checkpoint_file = output_dir / "checkpoint.json"
    with open(checkpoint_file, 'w') as f:
        json.dump({'last_processed_idx': last_processed_idx}, f)


def extract_activations(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    dataset: MoralActionDataset,
    extractor: ActivationExtractor,
    output_dir: Path,
    batch_size: int = 8,
    save_every: int = 100,
    resume_from: int = -1,
    num_workers: int = 4,
):
    """Main extraction loop with batching and periodic saving."""
    
    # Create data loader with custom collate
    def collate_wrapper(batch):
        return collate_fn(batch, tokenizer, dataset.max_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_wrapper,
        pin_memory=True
    )
    
    all_results: List[ExtractionResult] = []
    batch_counter = 0
    file_batch_id = resume_from // save_every + 1 if resume_from >= 0 else 0
    
    # Calculate starting batch
    start_batch = (resume_from + 1) // batch_size if resume_from >= 0 else 0
    
    if hasattr(model, 'hf_device_map'):
        first_device = list(model.hf_device_map.values())[0]
        if isinstance(first_device, int):
            device = torch.device(f"cuda:{first_device}")
        else:
            device = torch.device(first_device)
    else:
        device = next(model.parameters()).device
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Extracting activations")
    
    for batch_idx, (samples, encodings) in pbar:
        # Skip already processed batches
        if batch_idx < start_batch:
            continue
        
        # Move inputs to device
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        # Clear previous activations
        extractor.clear()
        
        # Forward pass (no gradient computation)
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get activations and compute representations
        activations = extractor.get_activations()
        mean_pooling, last_token = compute_representations(activations, attention_mask)
        
        # Create results for each sample in batch
        for i, sample in enumerate(samples):
            result = ExtractionResult(
                id=sample.id,
                moral_vector=sample.moral_vector,
                sampled_dimension=sample.sampled_dimension,
                sample_type=sample.sample_type,
                mean_pooling={k: v[i] for k, v in mean_pooling.items()},
                last_token={k: v[i] for k, v in last_token.items()}
            )
            all_results.append(result)
        
        batch_counter += 1
        
        # Periodic saving
        if batch_counter % save_every == 0:
            save_batch_results(all_results, output_dir, file_batch_id)
            current_idx = batch_idx * batch_size + len(samples) - 1
            save_checkpoint(output_dir, current_idx)
            
            pbar.set_postfix({
                'saved': file_batch_id,
                'samples': len(all_results)
            })
            
            all_results = []
            file_batch_id += 1
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
    
    # Save remaining results
    if all_results:
        save_batch_results(all_results, output_dir, file_batch_id)
        total_samples = len(dataset)
        save_checkpoint(output_dir, total_samples - 1)
    
    print(f"Extraction complete. Results saved to {output_dir}")


def merge_batch_files(output_dir: Path, delete_batches: bool = False):
    """Merge all batch files into a single file."""
    batch_files = sorted(output_dir.glob("batch_*.pt"))
    
    if not batch_files:
        print("No batch files found to merge.")
        return
    
    print(f"Merging {len(batch_files)} batch files...")
    
    all_results = []
    for batch_file in tqdm(batch_files, desc="Loading batches"):
        batch_results = torch.load(batch_file, weights_only=False)
        all_results.extend(batch_results)
    
    # Save merged file
    merged_file = output_dir / "activations_merged.pt"
    torch.save(all_results, merged_file)
    print(f"Merged {len(all_results)} samples to {merged_file}")
    
    # Optionally delete batch files
    if delete_batches:
        for batch_file in batch_files:
            batch_file.unlink()
        print("Deleted individual batch files.")

# ============================================================================
# Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract LLM hidden states for moral actions."
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Local checkpoint path or Hugging Face model identifier"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Display/output name (default: basename of --model_path)"
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Moral-action JSONL (default: dataset/comparison sample)"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Tensor directory (default: comparison/artifacts/activations/<model>)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Inference batch size"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Max sequence length"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=200,
        help="Save to disk every N batches"
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--all_layers",
        action="store_true",
        help="Extract all layers (overrides --layers_to_extract)"
    )
    parser.add_argument(
        "--layers_to_extract",
        type=int,
        nargs='+',
        default=[-1],
        help="List of layer indices to extract (e.g., 0 12 24). Use -1 for last layer."
    )
    parser.add_argument(
        "--merge_batches",
        action="store_true",
        help="Merge batch files after extraction"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Model precision"
    )
    parser.add_argument(
        "--attn_impl",
        choices=("auto", "flash_attention_2", "sdpa", "eager"),
        default="auto",
    )
    parser.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable network model/tokenizer lookups (default: enabled)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.data_path.is_file():
        raise FileNotFoundError(f"Data file does not exist: {args.data_path}")

    if args.model_name is None:
        args.model_name = Path(args.model_path.rstrip("/")).name
    
    # Set up output directory (使用 model_name 命名)
    if args.output_dir is None:
        # 清理 model_name 中的特殊字符
        safe_model_name = args.model_name.replace('/', '_').replace('\\', '_')
        args.output_dir = DEFAULT_ACTIVATIONS_DIR / safe_model_name
    
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save extraction config
    config_file = output_dir / "extraction_config.json"
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)
    
    # Set dtype
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32
    }
    torch_dtype = dtype_map[args.dtype]
    
    # Load model and tokenizer (直接使用 model_path)
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        torch_dtype,
        args.attn_impl,
        args.local_files_only,
    )
    
    # Determine layers to extract
    total_layers = get_total_layers(model)
    print(f"Model has {total_layers} layers")
    
    if args.all_layers:
        layers_to_extract = list(range(total_layers))
    else:
        # Handle negative indices
        layers_to_extract = []
        for layer_idx in args.layers_to_extract:
            if layer_idx < 0:
                layer_idx = total_layers + layer_idx
            if 0 <= layer_idx < total_layers:
                layers_to_extract.append(layer_idx)
            else:
                print(f"Warning: Layer {layer_idx} out of range, skipping")
        layers_to_extract = list(set(layers_to_extract))
    
    print(f"Extracting layers: {sorted(layers_to_extract)}")
    
    # Load dataset
    dataset = MoralActionDataset(
        str(args.data_path),
        tokenizer,
        args.max_length
    )
    
    # Set up extractor
    extractor = ActivationExtractor(
        model,
        layers_to_extract
    )
    
    # Check for resume
    resume_from = -1
    if args.resume:
        resume_from = load_checkpoint(output_dir)
        if resume_from >= 0:
            print(f"Resuming from sample index {resume_from + 1}")
    
    # Run extraction
    try:
        extract_activations(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            extractor=extractor,
            output_dir=output_dir,
            batch_size=args.batch_size,
            save_every=args.save_every,
            resume_from=resume_from,
            num_workers=args.num_workers,
        )
    finally:
        extractor.remove_hooks()
    
    # Merge batch files if requested
    if args.merge_batches:
        merge_batch_files(output_dir, delete_batches=True)
    
    print("Done!")


if __name__ == "__main__":
    main()
