import requests
import numpy as np
import random
import time
from typing import List, Dict, Tuple, Any, Optional
import pandas as pd
from einops import rearrange
from sklearn.metrics import balanced_accuracy_score
import json
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = "qwen/qwq-32b:free"
DEFAULT_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "HTTP-Referer": "https://your-app-or-website.com",
    "X-Title": "Neuron Auto-Interpretation",
    "Content-Type": "application/json"
}
API_URL = "https://openrouter.ai/api/v1/chat/completions"

def call_openrouter(prompt: str, model: str = DEFAULT_MODEL, 
                   temperature: float = 0.2, max_retries: int = 3) -> str:
    """Make an API call to OpenRouter."""
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature
    }
    
    for attempt in range(max_retries):
        try:
            print(f"Calling API with model: {model}, attempt {attempt+1}/{max_retries}")
            response = requests.post(API_URL, headers=DEFAULT_HEADERS, json=data)
            response.raise_for_status()
            
            # Print full response for debugging
            response_json = response.json()
            print(f"Full API response: {response_json}")
            
            # More robust parsing with detailed error messages
            if "choices" not in response_json:
                raise KeyError(f"'choices' not found in response: {response_json}")
            if not response_json["choices"]:
                raise ValueError(f"Empty 'choices' array in response: {response_json}")
            if "message" not in response_json["choices"][0]:
                raise KeyError(f"'message' not found in first choice: {response_json['choices'][0]}")
            if "content" not in response_json["choices"][0]["message"]:
                raise KeyError(f"'content' not found in message: {response_json['choices'][0]['message']}")
                
            content = response_json["choices"][0]["message"]["content"]
            if not content:
                print(f"Warning: Empty content received from API")
            
            return content
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"API call failed: {str(e)}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise Exception(f"Failed after {max_retries} attempts: {str(e)}")

def get_neuron_explanation(
    activating_examples: List[Dict[str, Any]], 
    top_logits: Optional[List[str]] = None,
    model: str = DEFAULT_MODEL,
    neuron_name: str = None
) -> str:
    """
    Generate an explanation for what a neuron detects based on activating examples.
    
    Args:
        activating_examples: List of dicts with 'text' and 'activation' keys
        top_logits: Optional list of top tokens promoted by this neuron
        model: Which LLM to use
        neuron_name: Name of the neuron for saving the prompt
    
    Returns:
        explanation: Text explanation of what the neuron detects
    """
    # Format activating examples with dual-level marking
    formatted_examples = []
    for i, example in enumerate(activating_examples):
        text = example["text"]  # This is the focused context
        activation = example["activation"]
        
        # Mark tokens with different levels of activation
        if "token_activations" in example:
            token_acts = example["token_activations"]
            threshold = 0.1 * max(token_acts)
            high_threshold = 0.7 * max(token_acts)
            marked_text = ""
            tokens = example["tokens"]
            
            # If we have the high_activating field, use it for more precise marking
            if "high_activating" in example:
                high_activating = example["high_activating"]
                
                for j, (token, act, is_high) in enumerate(zip(tokens, token_acts, high_activating)):
                    if is_high:
                        marked_text += f"<<{token}>>"  # Double brackets for highly activating tokens
                    elif act >= threshold:
                        marked_text += f"[[{token}]]"    # Single brackets for moderately activating tokens
                    else:
                        marked_text += f"{token}"      # No marking for non-activating tokens
            else:
                # Fallback to threshold-based marking
                for j, (token, act) in enumerate(zip(tokens, token_acts)):
                    if act >= high_threshold:
                        marked_text += f"<<{token}>>"
                    elif act >= threshold:
                        marked_text += f"[[{token}]]"
                    else:
                        marked_text += f"{token}"
            
            formatted_examples.append(f"Example {i+1}: {marked_text.strip()} [Activation: {activation:.4f}]")
        else:
            # If token-level activations aren't available, just show the text
            formatted_examples.append(f"Example {i+1}: {text} [Activation: {activation:.4f}]")
    
    # Create the prompt
    prompt = "You are helping me interpret what activates a specific neuron in a neural network.\n\n"
    prompt += "Here are examples of text that strongly activate this neuron. "
    prompt += "The tokens that activate the neuron are marked with different symbols:\n"
    prompt += "- Tokens with <<double brackets>> are highly activating (70%+ of max activation)\n"
    prompt += "- Tokens with [[double brackets]] are moderately activating (10%+ of max activation)\n\n"
    
    prompt += "\n".join(formatted_examples) + "\n\n"
    
    if top_logits:
        prompt += f"The neuron most strongly promotes these tokens in the next token prediction: {', '.join(top_logits)}\n\n"
    
    prompt += "Based on these examples, write a concise and precise explanation of what this neuron detects. "
    prompt += "Focus especially on the highly activating tokens (in double brackets) and their patterns. "
    prompt += "Consider both the token itself and its context (surrounding tokens). "
    prompt += "Your explanation should be specific enough to distinguish this pattern from other similar patterns. "
    prompt += "Your explanation should be 2-3 sentences long and extremely specific about the pattern. Don't reference the double brackets in your explanation."
    
    # Call the API
    print(f"Prompt ready")
    filename_prefix = f"neuron_{neuron_name}_" if neuron_name else "neuron_"
    with open(f"{filename_prefix}explanation_prompt_{time.strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
        f.write(prompt)
    response = call_openrouter(prompt, model=model)
    print(f"Response Returned")
    return response

def prepare_neuron_data(
    activations, texts, neuron_idx, 
    top_k=40, num_non_activating=20, 
    context_window=15
):
    """
    Prepare data for a specific neuron.
    
    Args:
        activations: Array of shape [num_examples, num_neurons]
        texts: List of text examples corresponding to activations
        neuron_idx: Index of the neuron to analyze
        top_k: Number of top activating examples to use
        num_non_activating: Number of non-activating examples to sample
        context_window: Number of tokens to include around activating regions
    
    Returns:
        dict with activating_examples and non_activating_examples
    """
    # Get activations for the specific neuron
    neuron_acts = activations[:, neuron_idx]
    
    # Sort by activation (descending)
    sorted_indices = np.argsort(-neuron_acts)  # Descending
    
    # Get top activating examples
    activating_examples = []
    for i in range(min(top_k, len(sorted_indices))):
        idx = sorted_indices[i]
        act = neuron_acts[idx]
        full_text = texts[idx]
        
        if act <= 0:  # Skip if not activating
            continue
            
        # Calculate activation decile (1-10, with 10 being highest)
        decile = 10 - int(i * 10 / top_k)
        
        # Create shorter sample by splitting into tokens and taking a central window
        tokens = full_text.split()  # Approximation of tokens
        
        # If the sequence is short enough, use as is
        if len(tokens) <= context_window * 2:
            short_text = full_text
        else:
            # Otherwise, take a window from the middle
            mid_point = len(tokens) // 2
            start_idx = max(0, mid_point - context_window)
            end_idx = min(len(tokens), mid_point + context_window)
            short_text = " ".join(tokens[start_idx:end_idx])
        
        activating_examples.append({
            "text": short_text,
            "full_text": full_text,
            "activation": float(act),
            "activation_decile": decile
        })
    
    # Sample non-activating examples
    non_activating_indices = [i for i, act in enumerate(neuron_acts) if act <= 0]
    if len(non_activating_indices) > num_non_activating:
        non_activating_indices = random.sample(non_activating_indices, num_non_activating)
    
    non_activating_examples = []
    for idx in non_activating_indices:
        full_text = texts[idx]
        
        # Create shorter sample similar to activating examples
        tokens = full_text.split()
        
        # If the sequence is short enough, use as is
        if len(tokens) <= context_window * 2:
            short_text = full_text
        else:
            # Otherwise, take a random window of comparable length
            if len(tokens) > context_window * 2:
                start_idx = random.randint(0, len(tokens) - context_window * 2)
                end_idx = start_idx + context_window * 2
                short_text = " ".join(tokens[start_idx:end_idx])
            else:
                short_text = full_text
            
        non_activating_examples.append({
            "text": short_text, 
            "full_text": full_text,
            "activation": float(neuron_acts[idx])
        })
    
    return {
        "activating_examples": activating_examples,
        "non_activating_examples": non_activating_examples
    }

def detection_scoring(
    explanation: str,
    activating_examples: List[Dict[str, Any]],
    non_activating_examples: List[Dict[str, Any]],
    num_samples: int = 25,
    model: str = DEFAULT_MODEL,
    neuron_name: str = None
) -> Dict[str, float]:
    """
    Score explanation using detection - asking LLM to predict which examples activate the neuron.
    
    Args:
        explanation: Text explanation of what the neuron detects
        activating_examples: List of dicts with 'text' and 'activation' keys
        non_activating_examples: List of dicts with 'text' key
        num_samples: Number of total examples to use (will be balanced)
        model: Which LLM to use
        neuron_name: Name of the neuron for saving the prompt
    
    Returns:
        scores: Dict with balanced_accuracy and other metrics
    """
    # Sample examples to use
    n_activating = min(num_samples // 2, len(activating_examples))
    n_non_activating = min(num_samples - n_activating, len(non_activating_examples))
    
    sampled_activating = random.sample(activating_examples, n_activating)
    sampled_non_activating = random.sample(non_activating_examples, n_non_activating)
    
    # Combine and shuffle
    all_examples = sampled_activating + sampled_non_activating
    random.shuffle(all_examples)
    
    # Create the prompt
    prompt = "You are helping me evaluate whether an explanation correctly describes what a neuron detects.\n\n"
    prompt += f"Neuron explanation: \"{explanation}\"\n\n"
    prompt += "For each of the following examples, determine if the neuron would activate based on the explanation (Yes/No).\n\n"
    
    for i, example in enumerate(all_examples):
        # Using the shortened text version
        prompt += f"Example {i+1}: {example['text']}\n"
    
    prompt += "\nRespond with Yes or No for each example, one per line, numbered."
    
    # Call the API
    print(f"Detection Prompt ready")
    filename_prefix = f"neuron_{neuron_name}_" if neuron_name else "neuron_"
    with open(f"{filename_prefix}detection_prompt_{time.strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
        f.write(prompt)
    response = call_openrouter(prompt, model=model)
    print(f"Detection response received")
    
    # Parse the response
    predictions = []
    lines = response.strip().split('\n')
    for line in lines:
        if "yes" in line.lower():
            predictions.append(1)
        elif "no" in line.lower():
            predictions.append(0)
    
    # If we didn't get enough predictions, fill with zeros (conservative)
    while len(predictions) < len(all_examples):
        predictions.append(0)
    
    # Truncate if we somehow got too many
    predictions = predictions[:len(all_examples)]
    
    # Create ground truth labels
    ground_truth = [1 if i < n_activating else 0 for i in range(len(all_examples))]
    
    # Calculate metrics
    balanced_acc = balanced_accuracy_score(ground_truth, predictions)
    
    # Calculate precision and recall
    true_positives = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 1 and pred == 1)
    false_positives = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 0 and pred == 1)
    false_negatives = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 1 and pred == 0)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def fuzzing_scoring(
    explanation: str,
    activating_examples: List[Dict[str, Any]],
    num_samples: int = 35,
    model: str = DEFAULT_MODEL
) -> Dict[str, float]:
    """
    Score explanation using fuzzing - asking LLM if tokens are correctly marked.
    
    Args:
        explanation: Text explanation of what the neuron detects
        activating_examples: List of dicts with 'text', 'activation', and 'token_activations' keys
        num_samples: Number of total examples to use
        model: Which LLM to use
    
    Returns:
        scores: Dict with balanced_accuracy and other metrics
    """
    # Prepare correctly and incorrectly marked examples
    marked_examples = []
    ground_truth = []
    
    # Sample from different activation deciles
    deciles = {}
    for example in activating_examples:
        decile = int(example["activation_decile"]) if "activation_decile" in example else 10
        if decile not in deciles:
            deciles[decile] = []
        deciles[decile].append(example)
    
    # For each decile, create correctly and incorrectly marked examples
    for decile, examples in deciles.items():
        if not examples:
            continue
            
        # How many examples to take from this decile
        n_from_decile = max(1, int(num_samples * 0.1))  # Distribute across deciles
        if len(examples) < n_from_decile:
            sampled = examples
        else:
            sampled = random.sample(examples, n_from_decile)
        
        for example in sampled:
            text = example["text"]
            token_acts = example.get("token_activations", [1.0] * len(text.split()))
            threshold = 0.7 * max(token_acts) if token_acts else 0.5
            
            # Correctly mark
            tokens = text.split()
            correct_text = ""
            for token, act in zip(tokens, token_acts):
                if act >= threshold:
                    correct_text += f"<<{token}>> "
                else:
                    correct_text += f"{token} "
            
            marked_examples.append(correct_text.strip())
            ground_truth.append(1)  # Correctly marked
            
            # Incorrectly mark (pick random tokens)
            if len(tokens) > 3:  # Only create incorrect examples if we have enough tokens
                incorrect_text = ""
                # Count how many tokens should be marked
                correct_marks = sum(1 for act in token_acts if act >= threshold)
                
                # Randomly select that many tokens, but NOT the actually activating ones
                non_activating_indices = [i for i, act in enumerate(token_acts) if act < threshold]
                if len(non_activating_indices) >= correct_marks:
                    to_mark = random.sample(non_activating_indices, correct_marks)
                    
                    for i, token in enumerate(tokens):
                        if i in to_mark:
                            incorrect_text += f"<<{token}>> "
                        else:
                            incorrect_text += f"{token} "
                    
                    marked_examples.append(incorrect_text.strip())
                    ground_truth.append(0)  # Incorrectly marked
    
    # Shuffle examples and ground truth together
    combined = list(zip(marked_examples, ground_truth))
    random.shuffle(combined)
    marked_examples, ground_truth = zip(*combined)
    
    # Create the prompt
    prompt = "You are helping me evaluate whether an explanation correctly describes what a neuron detects.\n\n"
    prompt += f"Neuron explanation: \"{explanation}\"\n\n"
    prompt += "For each example, the tokens that might activate the neuron are marked with << >> delimiters.\n"
    prompt += "Determine if the markings correctly identify the tokens that would activate the neuron (Yes/No).\n\n"
    
    for i, example in enumerate(marked_examples):
        prompt += f"Example {i+1}: {example}\n"
    
    prompt += "\nRespond with Yes or No for each example, one per line, numbered."
    
    # Call the API
    response = call_openrouter(prompt, model=model)
    
    # Parse the response
    predictions = []
    lines = response.strip().split('\n')
    for line in lines:
        if "yes" in line.lower():
            predictions.append(1)
        elif "no" in line.lower():
            predictions.append(0)
    
    # If we didn't get enough predictions, fill with zeros (conservative)
    while len(predictions) < len(marked_examples):
        predictions.append(0)
    
    # Truncate if we somehow got too many
    predictions = predictions[:len(marked_examples)]
    
    # Calculate metrics
    balanced_acc = balanced_accuracy_score(ground_truth, predictions)
    
    # Calculate precision and recall
    true_positives = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 1 and pred == 1)
    false_positives = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 0 and pred == 1)
    false_negatives = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 1 and pred == 0)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def fuzzing_scoring_for_sequences(
    explanation: str,
    activating_examples: List[Dict[str, Any]],
    num_samples: int = 35,
    model: str = DEFAULT_MODEL
) -> Dict[str, float]:
    """
    Score explanation using fuzzing for sequence data.
    
    Args:
        explanation: Text explanation of what the neuron detects
        activating_examples: List of dicts with 'text', 'tokens', 'token_activations'
        num_samples: Number of total examples to use
        model: Which LLM to use
    
    Returns:
        scores: Dict with balanced_accuracy and other metrics
    """
    # Prepare correctly and incorrectly marked examples
    marked_examples = []
    ground_truth = []
    
    # Sample from different activation deciles
    deciles = {}
    for example in activating_examples:
        decile = int(example["activation_decile"]) if "activation_decile" in example else 10
        if decile not in deciles:
            deciles[decile] = []
        deciles[decile].append(example)
    
    # For each decile, create correctly and incorrectly marked examples
    for decile, examples in deciles.items():
        if not examples:
            continue
            
        # How many examples to take from this decile
        n_from_decile = max(1, int(num_samples * 0.1))  # Distribute across deciles
        if len(examples) < n_from_decile:
            sampled = examples
        else:
            sampled = random.sample(examples, n_from_decile)
        
        for example in sampled:
            tokens = example["tokens"]
            token_acts = example["token_activations"]
            threshold = 0.7 * max(token_acts) if token_acts else 0.5
            
            # Correctly mark
            correct_text = ""
            for token, act in zip(tokens, token_acts):
                if act >= threshold:
                    correct_text += f"<<{token}>>"
                else:
                    correct_text += token
            
            marked_examples.append(correct_text)
            ground_truth.append(1)  # Correctly marked
            
            # Incorrectly mark (pick random tokens)
            if len(tokens) > 3:  # Only create incorrect examples if we have enough tokens
                incorrect_text = ""
                # Count how many tokens should be marked
                correct_marks = sum(1 for act in token_acts if act >= threshold)
                
                # Randomly select that many tokens, but NOT the actually activating ones
                non_activating_indices = [i for i, act in enumerate(token_acts) if act < threshold]
                if len(non_activating_indices) >= correct_marks and correct_marks > 0:
                    to_mark = random.sample(non_activating_indices, correct_marks)
                    
                    for i, token in enumerate(tokens):
                        if i in to_mark:
                            incorrect_text += f"<<{token}>>"
                        else:
                            incorrect_text += token
                    
                    marked_examples.append(incorrect_text)
                    ground_truth.append(0)  # Incorrectly marked
    
    # Only proceed if we have examples
    if not marked_examples:
        return {
            "balanced_accuracy": 0.5,  # Default to random chance
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }
    
    # Shuffle examples
    combined = list(zip(marked_examples, ground_truth))
    random.shuffle(combined)
    marked_examples, ground_truth = zip(*combined)
    
    # Create prompt
    prompt = "You are helping me evaluate whether text examples have correctly marked activating tokens.\n\n"
    prompt += f"Explanation of what activates a neuron: \"{explanation}\"\n\n"
    prompt += "For each example below, the most relevant tokens are marked with << >>. "
    prompt += "Determine if the markings correctly identify the tokens that match the explanation (Yes/No).\n\n"
    
    for i, example in enumerate(marked_examples[:num_samples]):  # Limit to num_samples
        prompt += f"Example {i+1}: {example}\n"
    
    prompt += "\nRespond with Yes or No for each example, one per line, numbered."
    
    # Call the API
    response = call_openrouter(prompt, model=model)
    
    # Parse the response
    predictions = []
    lines = response.strip().split('\n')
    for line in lines:
        if "yes" in line.lower():
            predictions.append(1)
        elif "no" in line.lower():
            predictions.append(0)
    
    # If we didn't get enough predictions, fill with zeros (conservative)
    while len(predictions) < min(len(ground_truth), num_samples):
        predictions.append(0)
    
    # Truncate to match num_samples
    ground_truth = ground_truth[:num_samples]
    predictions = predictions[:len(ground_truth)]
    
    # Calculate metrics
    balanced_acc = balanced_accuracy_score(ground_truth, predictions)
    
    # Calculate precision and recall
    true_positives = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 1 and pred == 1)
    false_positives = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 0 and pred == 1)
    false_negatives = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 1 and pred == 0)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def analyze_neuron(activations, texts, neuron_idx, model=DEFAULT_MODEL):
    """
    Complete pipeline to analyze a single neuron: generate explanation and score it.
    
    Args:
        activations: Array of shape [num_examples, num_neurons]
        texts: List of text examples corresponding to activations
        neuron_idx: Index of the neuron to analyze
        model: Which LLM to use
    
    Returns:
        dict with explanation and scores
    """
    # Prepare data
    data = prepare_neuron_data(activations, texts, neuron_idx)
    
    # Generate explanation
    explanation = get_neuron_explanation(
        data["activating_examples"], 
        model=model
    )
    
    # Score explanation
    detection_scores = detection_scoring(
        explanation,
        data["activating_examples"],
        data["non_activating_examples"],
        model=model
    )
    
    fuzzing_scores = fuzzing_scoring(
        explanation,
        data["activating_examples"],
        model=model
    )
    
    # Calculate monosemanticity score (average of detection and fuzzing balanced accuracy)
    monosemanticity = (detection_scores["balanced_accuracy"] + fuzzing_scores["balanced_accuracy"]) / 2
    
    return {
        "neuron_idx": neuron_idx,
        "explanation": explanation,
        "detection_scores": detection_scores,
        "fuzzing_scores": fuzzing_scores,
        "monosemanticity": monosemanticity
    }

def analyze_neurons_batch(activations, texts, neuron_indices, model=DEFAULT_MODEL):
    """
    Analyze multiple neurons in batch.
    
    Args:
        activations: Array of shape [num_examples, num_neurons]
        texts: List of text examples corresponding to activations
        neuron_indices: List of neuron indices to analyze
        model: Which LLM to use
    
    Returns:
        List of analysis results
    """
    results = []
    for idx in neuron_indices:
        result = analyze_neuron(activations, texts, idx, model=model)
        results.append(result)
    return results

def compare_models(base_results, sparse_results):
    """
    Compare monosemanticity scores between BaseGPT and SparseGPT.
    
    Args:
        base_results: Results from analyze_neurons_batch for BaseGPT
        sparse_results: Results from analyze_neurons_batch for SparseGPT
    
    Returns:
        Dict with comparison metrics
    """
    base_scores = [r["monosemanticity"] for r in base_results]
    sparse_scores = [r["monosemanticity"] for r in sparse_results]
    
    return {
        "base_mean": np.mean(base_scores),
        "sparse_mean": np.mean(sparse_scores),
        "base_median": np.median(base_scores),
        "sparse_median": np.median(sparse_scores),
        "base_percent_above_0.8": np.mean([s > 0.8 for s in base_scores]),
        "sparse_percent_above_0.8": np.mean([s > 0.8 for s in sparse_scores])
    }

def prepare_data_from_df(df, neuron_col_prefix="neuron_", top_k=40, num_non_activating=20, context_window=15):
    """
    Process DataFrame activation data into format needed for auto-interpretation.
    
    Args:
        df: DataFrame with token activations
        neuron_col_prefix: Prefix for neuron column names (e.g., "neuron_")
        top_k: Number of top activating examples to use
        num_non_activating: Number of non-activating examples to sample
        context_window: Number of tokens to include around activating regions
    
    Returns:
        dict with:
            - activations: Array of shape [num_examples, num_neurons]
            - texts: List of text examples
            - neuron_cols: List of neuron column names
    """
    # Get neuron columns
    neuron_cols = [col for col in df.columns if col.startswith(neuron_col_prefix)]
    
    # Group by sample_idx to get full sequences
    grouped = df.groupby('sample_idx')
    
    # Extract full texts and max activations for each sequence
    texts = []
    activations = []
    
    for sample_idx, group in grouped:
        # Sort by position to get tokens in correct order
        group = group.sort_values('position')
        
        # Reconstruct full text - convert tokens to strings first
        tokens = [str(token) for token in group['token'].tolist()]
        full_text = ' '.join(tokens)  # Join with spaces for better readability in detection
        texts.append(full_text)
        
        # Get max activation for each neuron across the sequence
        max_activations = group[neuron_cols].max().values
        activations.append(max_activations)
    
    # Convert to numpy array
    activations = np.array(activations)
    
    return {
        "activations": activations,
        "texts": texts,
        "neuron_cols": neuron_cols
    }

def prepare_token_level_data_from_df(df, neuron_idx, neuron_col_prefix="neuron_", threshold=0.7, high_threshold=0.9, context_window=15):
    """
    Prepare token-level activations for a specific neuron from DataFrame.
    
    Args:
        df: DataFrame with token activations
        neuron_idx: Index or column name of the neuron to analyze
        neuron_col_prefix: Prefix for neuron column names
        threshold: Threshold for marking tokens as activating (fraction of max activation)
        high_threshold: Threshold for marking tokens as highly activating (fraction of max activation)
        context_window: Number of tokens to include before and after activating tokens
    
    Returns:
        List of dicts with text, token_activations, and activation keys
    """
    # Convert neuron_idx to column name if it's an index
    if isinstance(neuron_idx, int):
        neuron_cols = [col for col in df.columns if col.startswith(neuron_col_prefix)]
        neuron_col = neuron_cols[neuron_idx]
    else:
        neuron_col = neuron_idx
    
    # Group by sample_idx
    grouped = df.groupby('sample_idx')
    
    examples = []
    for sample_idx, group in grouped:
        # Sort by position
        group = group.sort_values('position')
        
        # Get tokens and their activations
        tokens = [str(token) for token in group['token'].tolist()]  # Convert to strings
        token_acts = group[neuron_col].tolist()
        
        # Skip if all activations are zero or negative
        if max(token_acts) <= 0:
            continue
        
        # Find activating tokens (those above threshold)
        max_act = max(token_acts)
        threshold_value = threshold * max_act
        high_threshold_value = high_threshold * max_act
        
        # Store both activation levels
        activating_indices = [i for i, act in enumerate(token_acts) if act >= threshold_value]
        high_activating_indices = [i for i, act in enumerate(token_acts) if act >= high_threshold_value]
        
        if not activating_indices:
            continue
            
        # Create context-focused examples around activating regions
        # For each activating region, create a separate example
        current_region = []
        regions = []
        
        for i, idx in enumerate(activating_indices):
            # If this index is far from the previous one, start a new region
            if i > 0 and idx > activating_indices[i-1] + 2:  # Gap of at least 2 tokens
                regions.append(current_region)
                current_region = [idx]
            else:
                current_region.append(idx)
                
        # Add the last region
        if current_region:
            regions.append(current_region)
            
        # Create examples for each activating region with surrounding context
        for region in regions:
            # Define window around the region
            start_idx = max(0, region[0] - context_window)
            end_idx = min(len(tokens), region[-1] + context_window + 1)
            
            # Extract tokens and activations for this window
            window_tokens = tokens[start_idx:end_idx]
            window_acts = token_acts[start_idx:end_idx]
            
            # Save original indices relative to the full sequence
            window_indices = list(range(start_idx, end_idx))
            window_high_activating = [idx in high_activating_indices for idx in window_indices]
            
            # Prepare example
            example = {
                "text": ' '.join(window_tokens),  # Use spaces between tokens for readability
                "tokens": window_tokens,
                "token_activations": window_acts,
                "high_activating": window_high_activating,  # New field for highly activating tokens
                "activation": max(window_acts),
                "full_text": ''.join(tokens)  # Keep the full text for reference
            }
            
            examples.append(example)
    
    # Sort by activation (descending)
    examples = sorted(examples, key=lambda x: x["activation"], reverse=True)
    
    # Add activation decile
    for i, example in enumerate(examples):
        example["activation_decile"] = 10 - min(int(i * 10 / len(examples)), 9)
    
    return examples

def prepare_token_level_data(csv_path, neuron_idx, neuron_col_prefix="neuron_", threshold=0.7):
    """
    Prepare token-level activations for a specific neuron.
    
    Args:
        csv_path: Path to the CSV file with token activations
        neuron_idx: Index or column name of the neuron to analyze
        neuron_col_prefix: Prefix for neuron column names
        threshold: Threshold for marking tokens (fraction of max activation)
    
    Returns:
        List of dicts with text, token_activations, and activation keys
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    return prepare_token_level_data_from_df(df, neuron_idx, neuron_col_prefix, threshold)

def analyze_neuron_from_df(df, neuron_idx, model=DEFAULT_MODEL):
    """
    Analyze a neuron from DataFrame activation data.
    
    Args:
        df: DataFrame with token activations
        neuron_idx: Index of the neuron to analyze (relative to neuron columns)
        model: Which LLM to use
    
    Returns:
        dict with explanation and scores
    """
    # Get neuron column name
    neuron_cols = [col for col in df.columns if col.startswith("neuron_")]
    neuron_col = neuron_cols[neuron_idx]
    
    # Prepare sequence-level data
    data = prepare_data_from_df(df)
    
    # Prepare token-level data
    token_level_examples = prepare_token_level_data_from_df(df, neuron_idx)
    
    # Generate explanation
    explanation = get_neuron_explanation(
        token_level_examples[:10],  # Top 10 examples
        model=model,
        neuron_name=neuron_col
    )

    print(f"Explanation for {neuron_col}: {explanation}")
    
    print(f"Preparing Detection Data")
    # Standard detection scoring with sequence data
    detection_data = prepare_neuron_data(
        data["activations"], 
        data["texts"], 
        neuron_idx
    )
    print(f"Detection Data Prepared")
    detection_scores = detection_scoring(
        explanation,
        detection_data["activating_examples"],
        detection_data["non_activating_examples"],
        model=model,
        neuron_name=neuron_col
    )
    print(f"Detection Scores Calculated")
    
    # Fuzzing scoring with token-level data
    # fuzzing_scores = fuzzing_scoring_for_sequences(
    #     explanation,
    #     token_level_examples,
    #     model=model
    # )
    print(f"Fuzzing Scores Calculated")
    # Calculate monosemanticity score
    fuzzing_scores = detection_scores
    monosemanticity = (detection_scores["balanced_accuracy"] + fuzzing_scores["balanced_accuracy"]) / 2
    
    return {
        "neuron_col": neuron_col,
        "explanation": explanation,
        "detection_scores": detection_scores,
        "fuzzing_scores": fuzzing_scores,
        "monosemanticity": monosemanticity
    }

def analyze_neuron_from_csv(csv_path, neuron_idx, model=DEFAULT_MODEL):
    """
    Analyze a neuron from CSV activation data.
    
    Args:
        csv_path: Path to the CSV file with token activations
        neuron_idx: Index of the neuron to analyze (relative to neuron columns)
        model: Which LLM to use
    
    Returns:
        dict with explanation and scores
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    return analyze_neuron_from_df(df, neuron_idx, model)

def save_neuron_result(result, model_type, output_dir="results"):
    """
    Save a single neuron analysis result to JSON file.
    
    Args:
        result: Result dict from analyze_neuron_from_df
        model_type: String indicating "base" or "sparse"
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename using neuron and timestamp
    neuron_col = result["neuron_col"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/{model_type}_{neuron_col}_{timestamp}.json"
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Saved {model_type} neuron result to {filename}")
    return filename

def save_comparison_results(results, output_dir="results"):
    """
    Save the full comparison results to JSON file.
    
    Args:
        results: Dict with base_results, sparse_results, and comparison
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    neuron_count = len(results["base_results"])
    filename = f"{output_dir}/comparison_n{neuron_count}_{timestamp}.json"
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved full comparison results to {filename}")
    return filename

def compare_neurons_base_vs_sparse(base_csv, sparse_csv, neuron_indices, model=DEFAULT_MODEL, output_dir="results", base_df=None, sparse_df=None):
    """
    Compare neurons between base and sparse models.
    
    Args:
        base_csv: Path to the CSV file with base model activations
        sparse_csv: Path to the CSV file with sparse model activations
        neuron_indices: List of neuron indices to analyze
        model: Which LLM to use
        output_dir: Directory to save result files
        base_df: Optional pre-loaded base DataFrame to avoid reloading
        sparse_df: Optional pre-loaded sparse DataFrame to avoid reloading
        
    Returns:
        Dict with comparison results
    """
    # Load CSV files if not provided
    if base_df is None:
        print(f"Loading base CSV: {base_csv}")
        base_df = pd.read_csv(base_csv)
    else:
        print(f"Using provided base DataFrame")
        
    if sparse_df is None:
        print(f"Loading sparse CSV: {sparse_csv}")
        sparse_df = pd.read_csv(sparse_csv)
    else:
        print(f"Using provided sparse DataFrame")
    
    base_results = []
    sparse_results = []
    
    for idx in neuron_indices:
        print(f"Analyzing neuron {idx}")
        base_result = analyze_neuron_from_df(base_df, idx, model=model)
        sparse_result = analyze_neuron_from_df(sparse_df, idx, model=model)
        print(f"Base result monosemanticity: {base_result['monosemanticity']:.3f}")
        print(f"Sparse result monosemanticity: {sparse_result['monosemanticity']:.3f}")
        
        # Save intermediate results
        save_neuron_result(base_result, "base", output_dir)
        save_neuron_result(sparse_result, "sparse", output_dir)
        
        base_results.append(base_result)
        sparse_results.append(sparse_result)
    
    # Compare results
    comparison = compare_models(base_results, sparse_results)
    
    results = {
        "base_results": base_results,
        "sparse_results": sparse_results,
        "comparison": comparison
    }
    
    # Save complete results
    save_comparison_results(results, output_dir)
    
    return results

def check_neuron_activations(df, neuron_idx, neuron_col_prefix="neuron_", min_activations=5):
    """
    Check if a neuron has sufficient activating samples in the dataset.
    
    Args:
        df: DataFrame with token activations
        neuron_idx: Index of the neuron to analyze
        neuron_col_prefix: Prefix for neuron column names
        min_activations: Minimum number of activating examples required
    
    Returns:
        bool: True if neuron has sufficient activations, False otherwise
    """
    # Get neuron column name
    neuron_cols = [col for col in df.columns if col.startswith(neuron_col_prefix)]
    neuron_col = neuron_cols[neuron_idx]
    
    # Group by sample_idx
    grouped = df.groupby('sample_idx')
    
    # Count samples with positive activations
    activation_count = 0
    for sample_idx, group in grouped:
        # Get max activation for this neuron in this sample
        max_act = group[neuron_col].max()
        if max_act > 0:
            activation_count += 1
            
        # Early exit if we've found enough
        if activation_count >= min_activations:
            return True
            
    return activation_count >= min_activations

def select_valid_neurons(base_df, sparse_df, num_neurons=10, max_attempts=100, neuron_col_prefix="neuron_"):
    """
    Select random neurons that have activations in both base and sparse models.
    
    Args:
        base_df: DataFrame with base model activations
        sparse_df: DataFrame with sparse model activations
        num_neurons: Number of neurons to select
        max_attempts: Maximum number of attempts to find valid neurons
        neuron_col_prefix: Prefix for neuron column names
    
    Returns:
        list: Indices of valid neurons
    """
    # Get total number of neurons
    base_neuron_cols = [col for col in base_df.columns if col.startswith(neuron_col_prefix)]
    total_neurons = len(base_neuron_cols)
    
    valid_neurons = []
    attempted_neurons = set()
    
    print(f"Selecting {num_neurons} valid neurons (with activations in both models)...")
    
    while len(valid_neurons) < num_neurons and len(attempted_neurons) < max_attempts:
        # Generate a random neuron index not yet attempted
        available_indices = [i for i in range(total_neurons) if i not in attempted_neurons]
        
        if not available_indices:
            print(f"Ran out of neurons to try after {len(attempted_neurons)} attempts")
            break
            
        neuron_idx = random.choice(available_indices)
        attempted_neurons.add(neuron_idx)
        
        # Check if this neuron has activations in both models
        base_valid = check_neuron_activations(base_df, neuron_idx)
        sparse_valid = check_neuron_activations(sparse_df, neuron_idx)
        
        if base_valid and sparse_valid:
            valid_neurons.append(neuron_idx)
            print(f"  Found valid neuron: {neuron_idx} ({len(valid_neurons)}/{num_neurons})")
    
    if len(valid_neurons) < num_neurons:
        print(f"Warning: Could only find {len(valid_neurons)} valid neurons after {len(attempted_neurons)} attempts")
    
    return valid_neurons

if __name__ == "__main__":
    # Paths to your CSV files
    base_csv = "out_base/mlp_activations_base_posttopk_0.0013020833333333333.csv"
    # sparse_csv = "out/mlp_activations__posttopk_0.0013020833333333333.csv"
    sparse_csv = "out_8x_sparse/mlp_activations_8x_sparse_posttopk_0.00016276041666666666.csv"
    
    # Create output directory for results
    output_dir = "neuron_analysis_results"
    os.makedirs(output_dir, exist_ok=True)

    # Load CSV files once for neuron selection
    print("Loading CSVs to check for valid neurons...")
    base_df = pd.read_csv(base_csv)
    sparse_df = pd.read_csv(sparse_csv)
    
    # Select valid neurons with activations in both models
    num_neurons_to_analyze = 10
    neuron_indices = select_valid_neurons(base_df, sparse_df, num_neurons=num_neurons_to_analyze)
    print(f"Selected neurons for analysis: {neuron_indices}")
    
    # Analyze the selected neurons, passing the already-loaded DataFrames
    results = compare_neurons_base_vs_sparse(
        base_csv, 
        sparse_csv, 
        neuron_indices, 
        output_dir=output_dir,
        base_df=base_df,
        sparse_df=sparse_df
    )

    # Print results
    for i, (base, sparse) in enumerate(zip(results["base_results"], results["sparse_results"])):
        print(f"Neuron {neuron_indices[i]}:")
        print(f"  Base explanation: {base['explanation'][:100]}...")
        print(f"  Base monosemanticity: {base['monosemanticity']:.3f}")
        print(f"  Sparse explanation: {sparse['explanation'][:100]}...")
        print(f"  Sparse monosemanticity: {sparse['monosemanticity']:.3f}")
        print()

    print("Overall comparison:")
    print(f"  Base mean monosemanticity: {results['comparison']['base_mean']:.3f}")
    print(f"  Sparse mean monosemanticity: {results['comparison']['sparse_mean']:.3f}")
    print(f"  Percent of Base neurons above 0.8: {results['comparison']['base_percent_above_0.8']*100:.1f}%")
    print(f"  Percent of Sparse neurons above 0.8: {results['comparison']['sparse_percent_above_0.8']*100:.1f}%")