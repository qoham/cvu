# # %%
# import os
# if True:
#     os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# import sys
# # `if` block is used to prevent formatting tools (such as autopep8) from reordering these lines.
# if True:
#     sys.path.append('..')

# from utils.evaluate_utils import get_sd_15_pipeline, generate_Df_from_pipeline, generate_Dr_from_pipeline

# seed = 100
# sd_pipeline = get_sd_15_pipeline(seed=seed)

# concepts = [
#     ('frog', 2000, 0.0),
#     ('frog', 2000, 0.2),
#     ('frog', 2000, 0.4),
#     ('frog', 2000, 0.6),
#     ('frog', 2000, 0.8),
#     ('frog', 2000, 1.0),
# ]

# method = "cvu"
# Df_image_num = 1000
# Dr_image_num = 3000
# batch_size = 25
# for concept, checkpoint, dropout in concepts:
#     sd_pipeline.load_lora_weights(f"./output/{method}/{concept}_dropout_{dropout}/checkpoint-{checkpoint}", adapter_name=concept)
#     sd_pipeline.set_adapters(concept)

#     # 验证适配器是否正确加载
#     assert concept in sd_pipeline.get_active_adapters(), f"加载 {concept} 的适配器失败"

#     generate_Df_from_pipeline(sd_pipeline, method, concept, Df_image_num, seed, batch_size, suffix=f"dropout_{dropout}")
#     generate_Dr_from_pipeline(sd_pipeline, method, concept, Dr_image_num, seed, batch_size, suffix=f"dropout_{dropout}")

#     # 删除当前适配器
#     sd_pipeline.delete_adapters(concept)

#     # 验证清理是否成功
#     assert len(sd_pipeline.get_active_adapters()) == 0, f"清理 {concept} 的适配器失败"

# # %%

# %%
import os
import sys
import argparse

if True:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# `if` block is used to prevent formatting tools (such as autopep8) from reordering these lines.
if True:
    sys.path.append('..')

from utils.evaluate_utils import get_sd_15_pipeline, generate_Df_from_pipeline, generate_Dr_from_pipeline

concepts = [
    ('frog', 2000, 0.0),
    ('frog', 2000, 0.2),
    ('frog', 2000, 0.4),
    ('frog', 2000, 0.6),
    ('frog', 2000, 0.8),
    ('frog', 2000, 1.0),
]
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate images from trained LoRA adapters')
    parser.add_argument('--concept', type=str, required=True, help='Concept to generate images for')
    parser.add_argument('--checkpoint', type=int, default=2000, help='Checkpoint step to load')
    parser.add_argument('--dropout', type=float, required=True, help='Dropout value used in training')
    parser.add_argument('--method', type=str, default="cvu", help='Training method used')
    parser.add_argument('--seed', type=int, default=100, help='Random seed')
    parser.add_argument('--df_images', type=int, default=1000, help='Number of forget dataset images to generate')
    parser.add_argument('--dr_images', type=int, default=3000, help='Number of retain dataset images to generate')
    parser.add_argument('--batch_size', type=int, default=25, help='Batch size for generation')
    cli_args = parser.parse_args()
    
    concept = cli_args.concept
    checkpoint = cli_args.checkpoint
    dropout = cli_args.dropout
    method = cli_args.method
    seed = cli_args.seed
    Df_image_num = cli_args.df_images
    Dr_image_num = cli_args.dr_images
    batch_size = cli_args.batch_size
    
    print(f"Loading SD pipeline with seed {seed}...")
    sd_pipeline = get_sd_15_pipeline(seed=seed)
    
    print(f"Loading LoRA weights for {concept} with dropout {dropout}...")
    adapter_path = f"./output/{method}/{concept}_dropout_{dropout}/checkpoint-{checkpoint}"
    print(f"Loading from: {adapter_path}")
    
    sd_pipeline.load_lora_weights(adapter_path, adapter_name=concept)
    sd_pipeline.set_adapters(concept)

    # 验证适配器是否正确加载
    assert concept in sd_pipeline.get_active_adapters(), f"加载 {concept} 的适配器失败"
    print(f"Adapter loaded successfully: {sd_pipeline.get_active_adapters()}")

    print(f"Generating {Df_image_num} images for forget dataset...")
    generate_Df_from_pipeline(sd_pipeline, method, concept, Df_image_num, seed, batch_size, suffix=f"dropout_{dropout}")
    
    print(f"Generating {Dr_image_num} images for retain dataset...")
    generate_Dr_from_pipeline(sd_pipeline, method, concept, Dr_image_num, seed, batch_size, suffix=f"dropout_{dropout}")

    # 删除当前适配器
    print("Cleaning up adapters...")
    sd_pipeline.delete_adapters(concept)

    # 验证清理是否成功
    assert len(sd_pipeline.get_active_adapters()) == 0, f"清理 {concept} 的适配器失败"
    print("Adapter cleanup successful")
    
    print("Image generation completed successfully!")

if __name__ == "__main__":
    main()
