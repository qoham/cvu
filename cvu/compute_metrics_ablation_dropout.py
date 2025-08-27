# %%
import os
if True:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
# `if` block is used to prevent formatting tools (such as autopep8) from reordering these lines.
if True:
    sys.path.append('..')

from utils.evaluate_utils import compute_metrics_for_concept

concepts = [
    ('frog', 0.0),
    ('frog', 0.2),
    ('frog', 0.4),
    ('frog', 0.6),
    ('frog', 0.8),
    ('frog', 1.0),
]

method = "cvu"
Df_image_num = 1000
batch_size = 100
print(f"=== Method: {method} ===")
for concept, dropout in concepts:
    compute_metrics_for_concept(method, concept, Df_image_num, batch_size, suffix=f"dropout_{dropout}")
    print("\n\n")

# %%
