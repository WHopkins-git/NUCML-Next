"""Update notebook cells 17 (markdown) and 18 (code) for NeuralNetEvaluator."""
import json

with open('notebooks/00_Baselines_and_Limitations.ipynb') as f:
    nb = json.load(f)

# ============================================================================
# Cell 17: Updated markdown
# ============================================================================
nb['cells'][17]['source'] = [
    "## Baseline 3 -- Simple Neural Network\n",
    "\n",
    "A small feedforward neural network trained on the same features as the tree\n",
    "models.  Unlike trees, a neural network can produce smooth (continuous)\n",
    "predictions because it composes differentiable activation functions.\n",
    "\n",
    "The `NeuralNetEvaluator` class (in `nucml_next.baselines`) wraps the\n",
    "network with CPU-friendly training defaults:\n",
    "\n",
    "| Setting | Default | Rationale |\n",
    "|---------|---------|----------|\n",
    "| **OneCycleLR** | warmup 30%, cosine anneal | 3-5x faster than flat Adam |\n",
    "| **Kaiming He init** | fan_in, ReLU | Correct variance for ReLU networks |\n",
    "| **AdamW** | weight_decay=1e-5 | Decoupled L2 regularisation |\n",
    "| **Early stopping** | patience=8 | Saves CPU time, restores best weights |\n",
    "| **Gradient clipping** | max_norm=1.0 | Prevents exploding gradients |\n",
    "\n",
    "Four loss functions are available:\n",
    "\n",
    "| Loss | Formula | Use case |\n",
    "|------|---------|----------|\n",
    "| **MSE** | $\\frac{1}{N}\\sum(y_i - \\hat{y}_i)^2$ | Standard regression baseline |\n",
    "| **Chi-squared** | $\\frac{1}{N}\\sum\\frac{(y_i - \\hat{y}_i)^2}{\\sigma_i^2}$ | Weights by measurement precision |\n",
    "| **Physics-informed** | MSE + $\\lambda\\|\\nabla\\hat{\\sigma}\\|$ | Penalises unphysical oscillations |\n",
    "| **Resonance-informed** | $\\chi^2$ + 1/v + threshold + curvature | Research-level nuclear physics priors |\n",
]

# ============================================================================
# Cell 18: Rewritten to use NeuralNetEvaluator
# ============================================================================
nb['cells'][18]['source'] = [
    "# ============================================================================\n",
    "# BASELINE 3: SIMPLE NEURAL NETWORK\n",
    "# ============================================================================\n",
    "# Uses the NeuralNetEvaluator class from nucml_next.baselines.\n",
    "# CPU-friendly defaults: OneCycleLR, Kaiming He init, AdamW, early stopping.\n",
    "#\n",
    "# LOSS FUNCTION OPTIONS:\n",
    "#   'mse'                -- Standard mean squared error\n",
    "#   'chi_squared'        -- Weighted MSE using inverse uncertainty (chi^2/N)\n",
    "#   'physics_informed'   -- MSE + smoothness penalty on d-sigma/dE\n",
    "#   'resonance_informed' -- Chi^2 + 1/v law + threshold rise + curvature bound\n",
    "\n",
    "from nucml_next.baselines import NeuralNetEvaluator\n",
    "\n",
    "nn_evaluator = NeuralNetEvaluator(\n",
    "    loss_function='chi_squared',      # change to try other losses\n",
    "    # hidden_sizes=(256, 128),         # wider than old [128, 64]\n",
    "    # epochs=50,                       # upper bound (early stopping fires sooner)\n",
    "    # batch_size=512,                  # smaller than old 4096\n",
    "    # learning_rate=3e-3,              # OneCycleLR peak LR\n",
    ")\n",
    "\n",
    "nn_metrics = nn_evaluator.train(\n",
    "    df_tier,\n",
    "    transformation_config=TRANSFORMATION_CONFIG,  # same config as tree baselines\n",
    "    verbose=True,\n",
    ")\n",
    "\n",
    "# nn_evaluator has .predict(df) -- works directly with IsotopePlotter\n",
    "nn_model = nn_evaluator\n",
]

# Clear any existing outputs from cell 18
nb['cells'][18]['outputs'] = []
nb['cells'][18]['execution_count'] = None

with open('notebooks/00_Baselines_and_Limitations.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Cells 17 and 18 updated successfully.")
