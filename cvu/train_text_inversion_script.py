# %%
import os
if True:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
# `if` block is used to prevent formatting tools (such as autopep8) from reordering these lines.
if True:
    sys.path.append('..')

from utils.utils import dict_to_argv, get_ti_learnable_property, get_ti_initializer_token, get_ti_validation_prompt, get_ti_max_train_steps, get_ti_sample_num
from train_textual_inversion import train_main, parse_args

lr = 5e-4
concepts = [
    ('frog', 1900),
    ('van_gogh', 2000),
    # ('nudity', 2000),
    # ('angelina_jolie', 2000),
]

method = 'cvu'
for concept, checkpoint in concepts:
    lora_dir = f"./output/{method}/{concept}/checkpoint-{checkpoint}"
    params = {
        "method": method,
        "pretrained_model_name_or_path": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "lora_dir": lora_dir,
        "train_data_dir": os.path.dirname(os.getcwd()) + f"/data/{concept}",
        "sample_num": get_ti_sample_num(concept),
        "learnable_property": get_ti_learnable_property(concept),
        "placeholder_token": f"<{concept}>",
        "initializer_token": get_ti_initializer_token(concept),
        "resolution": 512,
        "repeats": 100,
        "save_steps": 100,
        "num_vectors": 1,
        "train_batch_size": 10,
        "max_train_steps": get_ti_max_train_steps(concept),
        "learning_rate": lr,
        "output_dir": f"ti_output/{concept}",
        # "num_train_epochs": 10,
        "validation_prompt": get_ti_validation_prompt(concept),
        "validation_steps": 100,
        "num_validation_images": 6,
        "seed": 100,
    }
    args = parse_args(dict_to_argv(params))
    train_main(args)

# %%
