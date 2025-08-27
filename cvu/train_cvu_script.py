# %%
import os
if True:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys

if True:
    sys.path.append('..')

import argparse
from utils.utils import dict_to_argv, get_val_unlearn_prompt, get_val_retain_prompt
from train_cvu import train_main, parse_args

# lr = 4e-4
# concepts = [
#     ('frog', 2500, 0.18),
#     ('van_gogh', 2500, 0.18),
#     ('nudity', 2500, 0.18),
#     ('angelina_jolie', 2500, 0.18),
# ]

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run CVU unlearning for specific concepts')
    parser.add_argument('--concept', type=str, required=True, help='Concept to unlearn')
    parser.add_argument('--steps', type=int, default=2500, help='Number of training steps')
    parser.add_argument('--threshold_scale', type=float, default=0.18, help='Threshold scale')
    parser.add_argument('--lr', type=float, default=4e-4, help='Learning rate')
    cli_args = parser.parse_args()

    concept = cli_args.concept
    steps = cli_args.steps
    threshold_scale = cli_args.threshold_scale
    lr = cli_args.lr

    method = "cvu"
    print(f"↓=============开始遗忘{concept}=============↓")
    params = {
        "pretrained_model_name_or_path": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "variant": "fp16",
        "unlearn_data_dirs": f"../data/{concept};",
        "unlearn_concepts": f"{concept};",
        "retain_data_dir": "../data/retain",
        "unlearn_batch_size": 5,
        "retain_batch_size": 5,
        "unlearn_data_copy_multiple": 5,
        "output_dir": f"./output/{method}/{concept}",
        "resolution": 512,
        "unlearn_condition_dropout": 0.5,
        "retain_condition_dropout": 0.5,
        "alpha": 0.2,  # intermediate loss (unlearn and retain)
        "num_train_epochs": 1,  # 一般取1
        "max_train_steps": steps,
        "learning_rate": lr,
        "threshold_scale": threshold_scale,
        "lr_scheduler": "constant",
        # "lr_warmup_steps": 50,
        "validation_prompts": f"{get_val_unlearn_prompt(concept)}; {get_val_retain_prompt(concept)}",
        "validation_steps": 100,
        "checkpointing_steps": 100,
        "num_validation_images": 6,
        "train_text_encoder": False,
        "rank": 8,
        "seed": 100,
        "report_to": "wandb",
        # "wandb_mode": "offline",
        # "wandb_mode": "online",
        # "wandb_run_name": f"{method}",
        "mixed_precision": "fp16",
    }
    args = parse_args(dict_to_argv(params))
    train_main(args)
    print(f"↑=============结束遗忘{concept}=============↑\n\n")
    # torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
