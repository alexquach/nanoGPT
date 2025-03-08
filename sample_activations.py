"""
Sample MLP activations from a trained model using training set inputs
"""
import os
import pickle
import argparse
import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import einops
import tiktoken

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values
out_dir = 'out_8x_sparse'  # output directory
tag = '_'.join(out_dir.split('_')[1:])
init_from = 'resume'  # 'resume' from checkpoint in out_dir
dataset = 'openwebtext'  # dataset name
num_samples = 10  # number of training samples to analyze
max_tokens = 1024  # max tokens per sample
device = "cpu" #'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
proportion_neurons_to_sample = 0.125  # proportion of neurons to randomly sample
output_file = f'mlp_activations_{tag}_posttopk_{proportion_neurons_to_sample}.csv'  # output file for the activation data
seed = 1337
# -----------------------------------------------------------------------------

@dataclass
class NeuronIdentifier:
    """Class to identify a specific neuron in the model."""
    layer_idx: int
    neuron_idx: int

    def __str__(self):
        return f"{self.layer_idx}:{self.neuron_idx}"

class ActivationHook:
    """Hook to capture activations from MLP layers."""
    def __init__(self):
        self.activations = {}
        self.hooks = []

    def register_hook(self, module, layer_idx, hook_point='post_gelu'):
        """Register a forward hook on a module.
        
        Args:
            module: The MLP module to hook into
            layer_idx: Layer index for identification
            hook_point: Where to capture activations - 'post_gelu' or 'post_topk'
        """
        if hook_point == 'post_gelu':
            # Hook to capture activations after GELU
            hook = module.gelu.register_forward_hook(
                lambda mod, inp, out, layer=layer_idx:
                self.save_activations(out, layer)
            )
            self.hooks.append(hook)
        elif hook_point == 'post_topk':
            # We need to register a pre-hook on c_proj to capture 
            # the activations after topk but before projection
            hook = module.c_proj.register_forward_pre_hook(
                lambda mod, inp, layer=layer_idx:
                self.save_activations(inp[0], layer)  # inp is a tuple, we want the first element
            )
            self.hooks.append(hook)

    def save_activations(self, activations, layer_idx):
        """Save activations from a specific layer."""
        # Clone and detach to avoid memory leaks
        self.activations[layer_idx] = activations.detach().float().cpu()

    def clear(self):
        """Clear stored activations."""
        self.activations = {}

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def get_first_samples(model, data_dir, num_samples, max_tokens, device):
    """Get the first num_samples from the training data with token info."""
    # Load training data
    data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    
    # Load meta.pkl to get encoder/decoder
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        decode = lambda l: ''.join([meta['itos'][i] for i in l])
    else:
        # Fallback to GPT-2 encodings
        enc = tiktoken.get_encoding("gpt2")
        decode = lambda l: enc.decode(l)
    
    # Take first num_samples sequences of max_tokens
    samples = []
    prompts = []
    prompts_tokenized = []
    
    for i in range(num_samples):
        # Make sure we don't exceed data length
        start_idx = i * max_tokens
        if start_idx + max_tokens > len(data):
            break
            
        # Get sample tokens
        x_tokens = data[start_idx:start_idx + max_tokens].astype(np.int64)
        x = torch.tensor(x_tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        # Store prompt text for reference
        prompt = decode(x_tokens)
        prompts.append(prompt[:100] + "...")  # Store first 100 chars + ellipsis
        
        # Store tokenized prompt for token-level analysis
        if os.path.exists(meta_path):
            tokenized = [meta['itos'][t] for t in x_tokens]
        else:
            tokenized = [enc.decode([t]) for t in x_tokens]  # Decode each token individually
        
        prompts_tokenized.append(tokenized)
        samples.append(x)
    
    return samples, prompts, prompts_tokenized
def sample_mlp_neurons(model, proportion_neurons_to_sample):
    """Create a list of sampled MLP neurons in the model.
    
    Args:
        model: The GPT model
        proportion_neurons_to_sample: Fraction of neurons to sample
    
    Returns:
        List of NeuronIdentifier objects for the sampled neurons
    """
    all_neurons = []
    for layer_idx, block in enumerate(model.transformer.h):
        mlp = block.mlp
        # Get hidden dimension size - accounting for sparse MLP if enabled
        hidden_size = mlp.hidden_size
        
        # Determine how many neurons to sample from this layer
        num_to_sample = max(1, int(hidden_size * proportion_neurons_to_sample))
        
        # Randomly sample neuron indices
        neuron_indices = torch.randperm(hidden_size)[:num_to_sample].tolist()
        print(neuron_indices)
        
        # Add the sampled neurons to our list
        for neuron_idx in neuron_indices:
            all_neurons.append(NeuronIdentifier(layer_idx, neuron_idx))
        
        print(f"Sampled {num_to_sample} neurons from layer {layer_idx} (out of {hidden_size})")
    
    return all_neurons

def sample_neuron_activations(model, samples, neurons_to_sample, hook_point='post_gelu'):
    """Collect activations for the specified neurons across all samples.
    
    Args:
        model: The GPT model
        samples: List of input samples
        neurons_to_sample: List of neurons to sample
        hook_point: Where to capture activations - 'post_gelu' or 'post_topk'
    """
    activation_hook = ActivationHook()
    
    # Register hooks on each layer's MLP
    for layer_idx, block in enumerate(model.transformer.h):
        activation_hook.register_hook(block.mlp, layer_idx, hook_point=hook_point)
    
    # Store activations for each sample
    all_activations = []
    
    for sample_idx, x in enumerate(samples):
        print(f"Processing sample {sample_idx+1}/{len(samples)}")
        
        # Clear previous activations
        activation_hook.clear()
        
        # Forward pass through the model to generate activations
        with torch.no_grad():
            # Use autocast for mixed precision if needed
            ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16 if dtype == 'bfloat16' else torch.float16)
            with ctx:
                model(x)
        
        # Extract activations for our sampled neurons
        sample_activations = {}
        for neuron in neurons_to_sample:
            layer_activations = activation_hook.activations[neuron.layer_idx]
            # Store per-token activations instead of averaging
            # Shape: [sequence_length]
            token_activations = layer_activations[0, :, neuron.neuron_idx].cpu().numpy()
            sample_activations[str(neuron)] = token_activations
        
        all_activations.append(sample_activations)
    
    # Clean up hooks
    activation_hook.remove_hooks()
    
    return all_activations

def main():
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Set up device
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Load the model
    print(f"Loading model from {out_dir}...")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    gptconf.is_sparse_mlp = True
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    # Data directory
    data_dir = os.path.join('data', dataset)
    
    # Get samples from training data
    print(f"Getting {num_samples} samples from training data...")
    samples, prompts, prompts_tokenized = get_first_samples(model, data_dir, num_samples, max_tokens, device)
    
    # Sample a random subset of neurons
    print(f"Sampling {proportion_neurons_to_sample} of neurons...")
    sampled_neurons = sample_mlp_neurons(model, proportion_neurons_to_sample)

    # Collect activations for sampled neurons
    print(f"Collecting neuron activations (hook point: {hook_point})...")
    all_activations = sample_neuron_activations(model, samples, sampled_neurons, hook_point=hook_point)
    
    # Create DataFrame with results
    print("Creating output DataFrame...")
    df_data = []
    for idx, (activations, x) in enumerate(zip(all_activations, samples)):
        # Get the decoded tokens for this sample
        tokens = prompts_tokenized[idx]  # We'll need to add this to get_first_samples
        
        # For each position in the sequence
        for pos in range(len(tokens)):
            row = {
                'sample_idx': idx,
                'position': pos,
                'token': tokens[pos],
                'prompt_prefix': prompts[idx][:pos+1]  # The prompt up to this token
            }
            
            # Add activation for each neuron at this position
            for neuron_id, token_activations in activations.items():
                row[f'neuron_{neuron_id}'] = token_activations[pos]
            
            df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Save to CSV
    output_path = os.path.join(out_dir, output_file)
    print(f"Saving activations to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Sample MLP activations from a trained model')
    parser.add_argument('--out_dir', type=str, default=out_dir)
    parser.add_argument('--dataset', type=str, default=dataset)
    parser.add_argument('--num_samples', type=int, default=num_samples)
    parser.add_argument('--max_tokens', type=int, default=max_tokens)
    parser.add_argument('--proportion_neurons_to_sample', type=float, default=proportion_neurons_to_sample)
    parser.add_argument('--output_file', type=str, default=output_file)
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--hook_point', type=str, default='post_topk', choices=['post_gelu', 'post_topk'],
                        help='Where to capture activations - after GELU or after TopK')
    args = parser.parse_args()
    
    # Update config with command-line arguments
    out_dir = args.out_dir
    dataset = args.dataset
    num_samples = args.num_samples
    max_tokens = args.max_tokens
    proportion_neurons_to_sample = args.proportion_neurons_to_sample
    output_file = args.output_file
    seed = args.seed
    hook_point = args.hook_point
    
    main()
