# %%
import os
if True:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
# `if` block is used to prevent formatting tools (such as autopep8) from reordering these lines.
if True:
    sys.path.append('..')

from utils.evaluate_utils import calculate_nd_and_fd, get_retain_prompts_by_num, get_sd_15_pipeline
import torch

concepts = [
    ('frog', 2000, 0.0),
    ('frog', 2000, 0.2),
    ('frog', 2000, 0.4),
    ('frog', 2000, 0.6),
    ('frog', 2000, 0.8),
    ('frog', 2000, 1.0),
]

seed = 100
image_num = 3000
prompts = get_retain_prompts_by_num(image_num, seed)


method = "cvu"
img_dir = f"../data/sd5_Dr"
batch_size = 10
for concept, checkpoint, dropout in concepts:
    unlearn_pipeline = get_sd_15_pipeline(seed=seed, torch_dtype=torch.float32)
    origin_pileline = get_sd_15_pipeline(seed=seed, torch_dtype=torch.float32)

    unlearn_pipeline.load_lora_weights(f"./output/{method}/{concept}_dropout_{dropout}/checkpoint-{checkpoint}", adapter_name=concept)
    unlearn_pipeline.set_adapters(concept)
    # 验证适配器是否正确加载
    assert concept in unlearn_pipeline.get_active_adapters(), f"加载 {concept} 的适配器失败"

    noise_distance, feature_distance = calculate_nd_and_fd(
        unlearn_pipeline=unlearn_pipeline,
        origin_pileline=origin_pileline,
        prompts=prompts,
        img_dir=img_dir,
        batch_size=batch_size,
        mixed_precision="fp16",
    )
    print(f"==> Method: {method}, {concept} FD 为: {feature_distance}, PND 为: {noise_distance}")
    # 删除当前适配器
    unlearn_pipeline.delete_adapters(concept)
    # 验证清理是否成功
    assert len(unlearn_pipeline.get_active_adapters()) == 0, f"清理 {concept} 的适配器失败"

    del unlearn_pipeline, origin_pileline
    torch.cuda.empty_cache()

# %%
