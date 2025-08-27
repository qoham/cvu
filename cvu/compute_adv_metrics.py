# %%
import os
if True:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
# `if` block is used to prevent formatting tools (such as autopep8) from reordering these lines.
if True:
    sys.path.append('..')

from utils.evaluate_utils import compute_adv_metrics_for_concept

concept_list = [
    'frog',
    'van_gogh',
    'nudity',
    'angelina_jolie',
]

method = "cvu"
Df_image_num = 1000
batch_size = 100
print(f"=== Method: {method} ===")
for concept in concept_list:
    compute_adv_metrics_for_concept(method, concept, Df_image_num, batch_size)
    print("\n\n")

# %%
