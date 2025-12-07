# Crosscoder Training Runner

This script trains crosscoders on language models, similar to `llm_sae_training_runner.py` but adapted for multi-layer mappings.

## Key Differences from SAE Runner

### Architecture
- **SAE**: Maps single layer activations back to themselves (autoencoder)
- **Crosscoder**: Maps concatenated activations from multiple input layers to concatenated activations at multiple output layers

### Configuration Changes

1. **Multiple Hook Points** (instead of single `hook_name`):
   ```python
   hook_names_in = ["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"]
   hook_names_out = ["blocks.3.hook_resid_post", "blocks.4.hook_resid_post"]
   ```

2. **Dimensions**:
   - `d_model`: Base model dimension (e.g., 512 for pythia-70m)
   - `d_in`: Automatically calculated as `d_model * len(hook_names_in)`
   - `d_out`: Automatically calculated as `d_model * len(hook_names_out)`

3. **Architecture Options**:
   - `"crosscoder"`: Standard crosscoder
   - `"skip_crosscoder"`: Adds learnable skip connection
   - `"jumprelu_crosscoder"`: Uses JumpReLU activation

## Usage

### Basic Example
```python
from llm_crosscoder_training_runner import (
    LanguageModelCrosscoderRunnerConfig,
    LanguageModelCrosscoderTrainingRunner,
)

cfg = LanguageModelCrosscoderRunnerConfig(
    model_name="EleutherAI/pythia-70m",
    hook_names_in=["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"],
    hook_names_out=["blocks.3.hook_resid_post", "blocks.4.hook_resid_post"],
    d_model=512,
    d_sae=4096,
    training_tokens=10_000_000,
    output_path="./my_crosscoder",
)

runner = LanguageModelCrosscoderTrainingRunner(cfg)
crosscoder = runner.run()
```

### Command Line
```bash
python llm_crosscoder_training_runner.py \
    --model_name EleutherAI/pythia-70m \
    --hook_names_in blocks.0.hook_resid_post blocks.1.hook_resid_post \
    --hook_names_out blocks.3.hook_resid_post blocks.4.hook_resid_post \
    --d_model 512 \
    --d_sae 4096 \
    --training_tokens 10000000 \
    --output_path ./my_crosscoder
```

## Crosscoder Types

### Causal Crosscoder
Maps earlier layers to later layers, predicting future activations:
```python
hook_names_in=["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"]
hook_names_out=["blocks.3.hook_resid_post", "blocks.4.hook_resid_post"]
```

### Acausal Crosscoder
Maps later layers to earlier layers (less common):
```python
hook_names_in=["blocks.3.hook_resid_post", "blocks.4.hook_resid_post"]
hook_names_out=["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"]
```

### Mixed Crosscoder
Maps across non-contiguous layers:
```python
hook_names_in=["blocks.0.hook_resid_post", "blocks.3.hook_resid_post"]
hook_names_out=["blocks.1.hook_resid_post", "blocks.4.hook_resid_post"]
```

## Maintained Similarities with SAE Runner

To keep troubleshooting easier, the following structure is identical:
- Training loop via `SAETrainer`
- Checkpoint saving/loading
- WandB logging
- Evaluation metrics
- Signal handling for interruptions
- Compilation options

## Architecture Details

The crosscoder forward pass:
```
Input:  [batch, d_model * n_input_layers]
Encode: [batch, d_sae] (sparse features)
Decode: [batch, d_model * n_output_layers]
```

Loss compares reconstructed output to actual `acts_out` from the output layers, not to the input.

## Requirements

Your existing infrastructure already supports crosscoders:
- `ActivationsStore` detects crosscoder mode via `hook_names_in`/`hook_names_out`
- Returns `(acts_in, acts_out)` tuples for training
- `cache_activations_runner.py` handles crosscoder caching

## Troubleshooting

**Issue**: `ValueError` about dimension mismatch
- **Fix**: Ensure `d_model` matches your model's actual dimension
- For pythia-70m: `d_model=512`
- For pythia-160m: `d_model=768`

**Issue**: Out of memory
- **Fix**: Reduce `n_batches_in_buffer`, `store_batch_size_prompts`, or `d_sae`

**Issue**: Loss not decreasing
- **Fix**: Try lower `l1_coefficient` or higher `lr`

## Example: Pythia-70m on TinyStories

See `example_train_pythia70m_crosscoder.py` for a complete working example that trains a causal crosscoder mapping layers 0-2 to layers 3-5.
