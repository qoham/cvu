# %%
import os
if True:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
# `if` block is used to prevent formatting tools (such as autopep8) from reordering these lines.
if True:
    sys.path.append('..')

from utils.evaluate_utils import get_sd_15_pipeline, generate_adv_Df_from_pipeline_by_prompts
from utils.utils import get_adv_validation_prompts

seed = 100

concepts = [
    ('frog', 2000, 0.1),
    ('frog', 2000, 0.3),
    ('frog', 2000, 0.5),
    ('frog', 2000, 0.7),
    ('frog', 2000, 0.9),
]

method = "cvu"
Df_image_num = 1000
batch_size = 25
for concept, checkpoint, threshold_scale in concepts:
    sd_pipeline = get_sd_15_pipeline(seed=seed)
    sd_pipeline.load_lora_weights(f"./output/{method}/{concept}_beta_{threshold_scale}/checkpoint-{checkpoint}", adapter_name=concept)
    sd_pipeline.set_adapters(concept)

    # 验证适配器是否正确加载
    assert concept in sd_pipeline.get_active_adapters(), f"加载 {concept} 的适配器失败"

    prompts = get_adv_validation_prompts(concept)
    generate_adv_Df_from_pipeline_by_prompts(sd_pipeline, method, concept, prompts, Df_image_num, seed, batch_size, suffix=f"beta_{threshold_scale}")

    # 删除当前适配器
    sd_pipeline.delete_adapters(concept)

    # 验证清理是否成功
    assert len(sd_pipeline.get_active_adapters()) == 0, f"清理 {concept} 的适配器失败"
    del sd_pipeline


# %%
