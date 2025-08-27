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
    ('frog', 0.1),
    ('frog', 0.3),
    ('frog', 0.5),
    ('frog', 0.7),
    ('frog', 0.9),
]

method = "cvu"
Df_image_num = 1000
batch_size = 100
print(f"=== Method: {method} ===")
for concept, threshold_scale in concepts:
    compute_metrics_for_concept(method, concept, Df_image_num, batch_size, suffix=f"beta_{threshold_scale}")
    print("\n\n")

# %%
