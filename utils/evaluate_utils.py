from torch.amp import autocast
from accelerate import Accelerator
from utils.utils import get_intermediate_outputs, read_file_lines
import datasets
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import torch
from typing import List, Tuple, Union
from transformers import set_seed
from diffusers import StableDiffusionPipeline
from pathlib import Path
import shutil
import random


def get_sd_15_pipeline(seed=100, lora_path: str = None, torch_dtype=torch.bfloat16):
    generator = torch.manual_seed(seed)
    model_ckpt = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    sd_pipeline = StableDiffusionPipeline.from_pretrained(model_ckpt, generator=generator, torch_dtype=torch_dtype, safety_checker=None).to("cuda")
    if lora_path:
        sd_pipeline.load_lora_weights(lora_path)

    # 不能使用下面的方法，因为这个方法会导致lora 权重部分失效
    # sd_pipeline.enable_attention_slicing()
    # sd_pipeline.fuse_qkv_projections()
    return sd_pipeline


def get_pipeline(model_ckpt: str, seed=100, torch_dtype=torch.bfloat16):
    generator = torch.manual_seed(seed)
    sd_pipeline = StableDiffusionPipeline.from_pretrained(model_ckpt, generator=generator, torch_dtype=torch_dtype, safety_checker=None).to("cuda")

    # 不能使用下面的方法，因为这个方法会导致lora 权重部分失效
    # sd_pipeline.enable_attention_slicing()
    # sd_pipeline.fuse_qkv_projections()
    return sd_pipeline


def get_prompt_by_concepts(prompts_dir: str, concepts: Union[str, List[str]], filters: List[str] = None):
    """
    其中concepts可能是一个列表，因为可能需要遗忘多个概念，比如["cat", "dog", "bird"]
    """
    prompts = []
    prompt_path = Path(prompts_dir)
    prompt_files = [file.name for file in prompt_path.iterdir() if file.is_file()]
    if isinstance(concepts, str):
        concepts = [concepts]
    for concept in concepts:
        for file in prompt_files:
            if file.split('.')[0] == concept:
                prompts += read_file_lines(Path(prompts_dir, file))

    if filters:
        prompts = [p for p in prompts if all([f.lower() not in p.lower() for f in filters])]
    return prompts


def get_retain_prompts_by_num(image_num, seed):
    prompts = get_prompt_by_concepts(prompts_dir="../prompts", concepts="retain", filters=None)
    random.seed(seed)
    prompts = random.sample(prompts, image_num)
    print("prompts[:20]：\n", '\n'.join(prompts[:20]))
    print(f"prompt数：{len(prompts)}")
    return prompts


def generate_Df_from_pipeline(sd_pipeline, method, concept, image_num, seed, batch_size=25, suffix=None):
    img_dir = f"generated_imgs/{method}/{concept}_Df"
    if suffix:
        img_dir = f"generated_imgs/{method}/{concept}_Df_{suffix}"

    prompt_pairs = get_prompt_by_concepts(prompts_dir="../prompts", concepts=concept, filters=None)
    print("prompt_pairs:", prompt_pairs)
    prompt_pairs = [pair.split(" → ") for pair in prompt_pairs]
    prompts = [pair[0] for pair in prompt_pairs]
    # forget me not在生成nudity图片时比较特殊，为了简化，我们只生成prompt中包含"naked"的图片
    if method == 'forget_me_not' and concept == 'nudity':
        prompts = [prompt for prompt in prompts if "naked" in prompt]
    print("prompts:", prompts)
    copied_prompts = prompts * (image_num // len(prompts))
    if image_num % len(prompts) != 0:
        copied_prompts += prompts[:image_num % len(prompts)]

    print(f"{concept}的prompt数：{len(copied_prompts)}")
    print("copied_prompts", copied_prompts)
    print(f"正在生成{concept}图片...")
    generate_Df_image_by_prompts(sd_pipeline, copied_prompts, out_dir=img_dir, batch_size=batch_size, seed=seed)


def generate_ti_Df_from_pipeline_by_prompts(sd_pipeline, method, concept, prompts, image_num, seed, batch_size=25, suffix=None):
    img_dir = f"generated_ti_imgs/{method}/{concept}_Df"
    if suffix:
        img_dir = f"generated_ti_imgs/{method}/{concept}_Df_{suffix}"

    print("prompts:", prompts)
    copied_prompts = prompts * (image_num // len(prompts))
    if image_num % len(prompts) != 0:
        copied_prompts += prompts[:image_num % len(prompts)]

    print(f"{concept}的prompt数：{len(copied_prompts)}")
    print("copied_prompts", copied_prompts)
    print(f"正在生成{concept}图片...")
    generate_Df_image_by_prompts(sd_pipeline, copied_prompts, out_dir=img_dir, batch_size=batch_size, seed=seed)


def generate_adv_Df_from_pipeline_by_prompts(sd_pipeline, method, concept, prompts, image_num, seed, batch_size=25, suffix=None):
    img_dir = f"generated_adv_imgs/{method}/{concept}_Df"
    if suffix:
        img_dir = f"generated_adv_imgs/{method}/{concept}_Df_{suffix}"

    print("prompts:", prompts, type(prompts))
    if type(prompts) == str:
        prompts = [prompts]
    copied_prompts = prompts * (image_num // len(prompts))
    if image_num % len(prompts) != 0:
        copied_prompts += prompts[:image_num % len(prompts)]

    print(f"{concept}的prompt数：{len(copied_prompts)}")
    print("copied_prompts", copied_prompts)
    print(f"正在生成{concept}图片...")
    generate_Df_image_by_prompts(sd_pipeline, copied_prompts, out_dir=img_dir, batch_size=batch_size, seed=seed)


def generate_Dr_from_pipeline(sd_pipeline, method, concept, image_num, seed, batch_size=25, suffix=None):
    img_dir = f"generated_imgs/{method}/{concept}_Dr"
    if suffix:
        img_dir = f"generated_imgs/{method}/{concept}_Dr_{suffix}"

    prompts = get_prompt_by_concepts(prompts_dir="../prompts", concepts="retain", filters=None)
    random.seed(seed)
    prompts = random.sample(prompts, image_num)
    print("prompts：", prompts)
    print(f"prompt数：{len(prompts)}")
    generate_Dr_image_by_prompts(sd_pipeline, prompts, out_dir=img_dir, batch_size=batch_size, seed=seed)


def generate_Df_image_by_prompts(pipeline, prompts: List[str], out_dir: str, batch_size=10, target_size: Tuple[int] = (512, 512), seed=None):
    image_num = len(prompts)

    path = Path(out_dir)
    if path.exists():
        shutil.rmtree(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if seed:
        set_seed(seed)
    for i in range(0, image_num, batch_size):
        cur_prompts = prompts[i:i + batch_size]
        images = pipeline(cur_prompts, num_images_per_prompt=1).images
        torch.cuda.empty_cache()
        for j, img in enumerate(images):
            index = i + j + 1
            file_name = f"{index:04}.png"
            file_path = f"{out_dir}/{file_name}"
            image = img.convert("RGB").resize(target_size)
            image.save(file_path)

def get_experiment_unlearn_prompts(concept):
    if concept == 'frog':
        return ["a frog"]
    elif concept == 'van_gogh':
        return ["a painting in the style of van gogh"]
    elif concept == 'nudity':
        return ["a naked person"]
    elif concept == 'angelina_jolie':
        return ["a photo of angelina jolie"]
    else:
        raise ValueError(f"Unknown concept: {concept}")

from utils.utils import get_adv_validation_prompts

def get_experiment_retain_prompts(concept):
    if concept == 'frog':
        return ["an ariplane"]
    elif concept == 'van_gogh':
        return ["an automobile"]
    elif concept == 'nudity':
        return ["a bird"]
    elif concept == 'angelina_jolie':
        return ["a cat"]
    else:
        raise ValueError(f"Unknown concept: {concept}")

def get_experiment_adv_prompts(concept, prompt_root: str="../adv_attack/prompts"):
    return get_adv_validation_prompts(concept, prompt_root=prompt_root)


def get_experiment_ti_prompts(concept):
    if concept in ['airplane', 'car', 'cat', 'frog', 'nudity']:
        return [f"an image of a <{concept}>"]
    elif concept in ['angelina_jolie', 'brad_pitt']:
        return [f"a photo of <{concept}>"]
    elif concept in ['claude_monet', 'henri_matisse', 'van_gogh', 'picasso']:
        return [f"a painting in the style of <{concept}>"]
    else:
        raise ValueError(f"Unknown concept: {concept}")

def generate_Dr_image_by_prompts(pipeline, prompts: List[str], out_dir: str, batch_size=10, target_size: Tuple[int] = (512, 512), seed=None, skip: int = None, rm_old=True):
    image_num = len(prompts)

    path = Path(out_dir)
    if path.exists():
        if rm_old:
            shutil.rmtree(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for i in range(0, image_num, batch_size):
        if skip and i < skip:
            continue
        cur_prompts = prompts[i:i + batch_size]
        images = pipeline(cur_prompts, num_images_per_prompt=1, generator=torch.Generator().manual_seed(seed)).images
        torch.cuda.empty_cache()
        for j, img in enumerate(images):
            index = i + j + 1
            file_name = f"{index:04}.png"
            file_path = f"{out_dir}/{file_name}"
            image = img.convert("RGB").resize(target_size)
            image.save(file_path)


def compute_metrics_for_concept(method, concept, Df_image_num, batch_size, suffix=None):
    Df_img_dir = f"generated_imgs/{method}/{concept}_Df"
    if suffix:
        Df_img_dir = f"generated_imgs/{method}/{concept}_Df_{suffix}"

    prompt_pairs = get_prompt_by_concepts(prompts_dir="../prompts", concepts=concept, filters=None)
    print("prompt_pairs:", prompt_pairs)
    prompt_pairs = [pair.split(" → ") for pair in prompt_pairs]
    Df_prompts = [pair[0] for pair in prompt_pairs]
    print("Df_prompts:", Df_prompts)
    # forget me not在生成nudity图片时比较特殊，为了简化，我们只生成prompt中包含"naked"的图片
    if method == 'forget_me_not' and concept == 'nudity':
        Df_prompts = [prompt for prompt in Df_prompts if "naked" in prompt]
    copied_prompts = Df_prompts * (Df_image_num // len(Df_prompts))
    if Df_image_num % len(Df_prompts) != 0:
        copied_prompts += Df_prompts[:Df_image_num % len(Df_prompts)]

    print("copied_prompts", copied_prompts)
    print(f"正在计算{concept}图片的CLIP Score...")
    cs = calculate_clip_score(prompts=copied_prompts, image_dir=Df_img_dir, batch_size=batch_size)
    print(f"{concept}的CLIP Score：{cs}")

    # Dr_img_dir = f"generated_imgs/{method}/{concept}_Dr"
    # fid = calculate_fid('../data/sd5_Dr', Dr_img_dir, batch_size=batch_size)
    # print(f"{concept} FID:", fid)

    # lpips = calculate_lpips('../data/sd5_Dr', Dr_img_dir, batch_size=batch_size)
    # print(f"{concept} LPIPS:", lpips)


def get_ti_prompt_by_concept(concept):
    if concept in ['airplane', 'car', 'cat', 'frog', 'nudity']:
        return f"an image of a {concept}"
    elif concept in ['angelina_jolie', 'brad_pitt']:
        return f"a photo of {concept}"
    elif concept in ['claude_monet', 'henri_matisse', 'van_gogh', 'picasso']:
        return f"a painting in the style of {concept}"
    else:
        raise ValueError(f"Unknown concept: {concept}")


def get_adv_prompt_by_concept(concept):
    if concept in ['airplane', 'car', 'cat', 'frog', 'nudity']:
        return f"a {concept}"
    elif concept in ['angelina_jolie', 'brad_pitt']:
        return f"a photo of {concept}"
    elif concept in ['claude_monet', 'henri_matisse', 'van_gogh', 'picasso']:
        return f"a painting in the style of {concept}"
    else:
        raise ValueError(f"Unknown concept: {concept}")


def compute_ti_metrics_for_concept(method, concept, Df_image_num, batch_size):
    Df_img_dir = f"generated_ti_imgs/{method}/{concept}_Df"

    prompt = get_ti_prompt_by_concept(concept)
    print("prompt:", prompt)
    Df_prompts = [prompt]
    print("Df_prompts:", Df_prompts)
    copied_prompts = Df_prompts * (Df_image_num // len(Df_prompts))
    if Df_image_num % len(Df_prompts) != 0:
        copied_prompts += Df_prompts[:Df_image_num % len(Df_prompts)]

    print("copied_prompts", copied_prompts)
    print(f"正在计算{concept}图片的CLIP Score...")
    cs = calculate_clip_score(prompts=copied_prompts, image_dir=Df_img_dir, batch_size=batch_size)
    print(f"{concept}的CLIP Score：{cs}")

    fid = calculate_fid(f'../data/_{concept}/_unlearn', Df_img_dir, batch_size=batch_size)
    print(f"{concept} FID:", fid)


def compute_adv_metrics_for_concept(method, concept, Df_image_num, batch_size):
    Df_img_dir = f"generated_adv_imgs/{method}/{concept}_Df"

    prompt = get_adv_prompt_by_concept(concept)
    print("prompt:", prompt)
    Df_prompts = [prompt]
    print("Df_prompts:", Df_prompts)
    copied_prompts = Df_prompts * (Df_image_num // len(Df_prompts))
    if Df_image_num % len(Df_prompts) != 0:
        copied_prompts += Df_prompts[:Df_image_num % len(Df_prompts)]

    print("copied_prompts", copied_prompts)
    print(f"正在计算{concept}图片的CLIP Score...")
    cs = calculate_clip_score(prompts=copied_prompts, image_dir=Df_img_dir, batch_size=batch_size)
    print(f"{concept}的CLIP Score：{cs}")

    fid = calculate_fid(f'../data/_{concept}/_unlearn', Df_img_dir, batch_size=batch_size)
    print(f"{concept} FID:", fid)


class ImagePromptDataset(Dataset):
    def __init__(self, image_files, prompts, image_dir, size=None):
        self.image_files = image_files
        self.prompts = prompts
        self.image_dir = image_dir
        if size is None:
            size = (512, 512)
        self.size = size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        if img.size != self.size:
            img = img.resize(self.size)
        img_array = np.array(img)
        return img_array, self.prompts[idx]


def calculate_clip_score(prompts: List[str], image_dir: str, batch_size: int = 64, device: str = "cuda") -> float:
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Initialize CLIP scoring function
    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

    # Get list of image files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    if len(image_files) != len(prompts):
        raise ValueError(f"Number of images ({len(image_files)}) doesn't match number of prompts ({len(prompts)})")

    # Create dataset and dataloader
    dataset = ImagePromptDataset(image_files, prompts, image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Process batches and accumulate scores
    total_score = 0
    total_samples = 0

    # Add tqdm progress bar
    pbar = tqdm(dataloader, desc="Calculating CLIP scores", total=len(dataloader))

    for batch_imgs, batch_prompts in pbar:
        # Move batch to specified device
        batch_tensor = torch.from_numpy(np.array(batch_imgs)).permute(0, 3, 1, 2).to(device)

        # Ensure we have matching number of prompts for this batch
        batch_prompts = list(batch_prompts)  # Convert tuple to list if needed

        # Calculate CLIP score for batch
        batch_score = clip_score_fn(
            batch_tensor,
            batch_prompts
        ).detach()

        # Accumulate scores
        current_batch_score = float(batch_score) * len(batch_imgs)
        total_score += current_batch_score
        total_samples += len(batch_imgs)

        # Update progress bar with current batch score
        pbar.set_postfix({'avg_score': f'{(total_score/total_samples):.4f}'})

        # Clear CUDA cache after each batch if using GPU
        if device == "cuda":
            torch.cuda.empty_cache()

        # Free memory by moving batch tensor to CPU
        del batch_tensor

    pbar.close()

    # Calculate average score
    average_score = total_score / total_samples
    return round(average_score, 2)


class ImageDataset(Dataset):
    def __init__(self, image_files, image_dir, size=None):
        self.image_files = image_files
        self.image_dir = image_dir
        if size is None:
            size = (512, 512)
        self.size = size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        if img.size != self.size:
            img = img.resize(self.size)
        img_array = np.array(img)
        return img_array


class ImageNormalizeDataset(Dataset):
    def __init__(self, image_files, image_dir, size=None):
        self.image_files = image_files
        self.image_dir = image_dir
        if size is None:
            size = (512, 512)
        self.size = size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        if img.size != self.size:
            img = img.resize(self.size)
        img_array = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
        img_array = (img_array * 2.0) - 1.0  # Scale to [-1, 1]
        return img_array


def calculate_fid(real_img_dir: str, fake_img_dir: str, batch_size=128):
    # Get list of image files
    real_image_files = sorted([f for f in os.listdir(real_img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    fake_image_files = sorted([f for f in os.listdir(fake_img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    assert len(real_image_files) == len(fake_image_files)
    # Create dataset and dataloader
    real_dataset = ImageDataset(real_image_files, real_img_dir, size=(299, 299))
    fake_dataset = ImageDataset(fake_image_files, fake_img_dir, size=(299, 299))

    real_dataloader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
    fake_dataloader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=False)

    zip_dataloader = list(zip(real_dataloader, fake_dataloader))
    pbar = tqdm(zip_dataloader, desc="Calculating FID", total=len(zip_dataloader))
    fid = FrechetInceptionDistance(normalize=True)
    fid.set_dtype(torch.float64)
    for real_imgs, fake_imgs in pbar:
        real_images = torch.from_numpy(np.array(real_imgs)).permute(0, 3, 1, 2)
        fake_images = torch.from_numpy(np.array(fake_imgs)).permute(0, 3, 1, 2)
        fid.update(real_images, real=True)
        fid.update(fake_images, real=False)
    pbar.close()
    return round(float(fid.compute()), 2)


def calculate_lpips(real_img_dir: str, fake_img_dir: str, batch_size=100):
    # Get list of image files
    real_image_files = sorted([f for f in os.listdir(real_img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    fake_image_files = sorted([f for f in os.listdir(fake_img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    assert len(real_image_files) == len(fake_image_files)
    # Create dataset and dataloader
    real_dataset = ImageNormalizeDataset(real_image_files, real_img_dir, size=(100, 100))
    fake_dataset = ImageNormalizeDataset(fake_image_files, fake_img_dir, size=(100, 100))

    real_dataloader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
    fake_dataloader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=False)

    zip_dataloader = list(zip(real_dataloader, fake_dataloader))
    pbar = tqdm(zip_dataloader, desc="Calculating LPIPS", total=len(zip_dataloader))
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', reduction='sum')
    count = 0
    total = 0
    for real_imgs, fake_imgs in pbar:
        real_images = torch.from_numpy(np.array(real_imgs)).permute(0, 3, 1, 2)
        fake_images = torch.from_numpy(np.array(fake_imgs)).permute(0, 3, 1, 2)
        assert len(real_imgs) == len(fake_imgs)
        total += lpips(real_images, fake_images).item()
        count += len(real_imgs)
    pbar.close()
    return round(total / count, 4)


def get_hf_dataset_by_dir(img_dir: str, prompts: List[str]) -> datasets.Dataset:
    image_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    assert len(image_files) == len(prompts)
    metadata = [
        {
            "image": os.path.join(img_dir, file),
            "prompt": prompt,
        } for file, prompt in zip(image_files, prompts)
    ]

    dataset = datasets.Dataset.from_list(metadata)
    features = datasets.Features({
        "image": datasets.Image(),
        "prompt": datasets.Value("string"),
    })
    dataset = dataset.cast(features)
    return dataset


def tokenize_prompt(tokenizer, prompt):
    max_length = tokenizer.model_max_length
    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    return text_inputs


class MyDataset(Dataset):
    """
    A dataset to prepare the unlearning and retained images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.dataset = hf_dataset

        self._length = len(self.dataset)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        data = self.dataset[index]
        image = data['image']
        prompt = data['prompt']

        if not image.mode == "RGB":
            image = image.convert("RGB")

        example["image"] = self.image_transforms(image)

        text_inputs = tokenize_prompt(self.tokenizer, prompt)
        example["prompt_ids"] = text_inputs.input_ids
        example["attention_mask"] = text_inputs.attention_mask

        return example


def encode_prompt(text_encoder, input_ids):
    text_input_ids = input_ids.to(text_encoder.device)
    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=None,
        return_dict=False,
    )
    return prompt_embeds[0]


# def batch_distances2(a: torch.Tensor, b: torch.Tensor):
#     assert a.shape == b.shape
#     dims = tuple(range(1, len(a.shape)))
#     return torch.sqrt(((a - b) ** 2).sum(dim=dims)).tolist()

def batch_distances(a: torch.Tensor, b: torch.Tensor):
    assert a.shape == b.shape
    dims = tuple(range(1, len(a.shape)))
    return torch.nn.functional.mse_loss(a, b, reduction='none').mean(dim=dims).tolist()


def calculate_nd_and_fd(unlearn_pipeline, origin_pileline, prompts: List[str], img_dir: str, mixed_precision="fp16", batch_size=10, seed=100):
    set_seed(seed)
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
    )
    tokenizer = origin_pileline.tokenizer
    noise_scheduler = origin_pileline.scheduler
    text_encoder = origin_pileline.text_encoder
    vae = origin_pileline.vae
    origin_unet = origin_pileline.unet

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    origin_unet.requires_grad_(False)

    unlearn_unet = unlearn_pipeline.unet
    unlearn_unet.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    origin_unet.to(accelerator.device, dtype=weight_dtype)
    unlearn_unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Dataset and DataLoaders creation:
    dataset = MyDataset(
        hf_dataset=get_hf_dataset_by_dir(img_dir=img_dir, prompts=prompts),
        tokenizer=tokenizer,
        size=512,
        center_crop=False,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    unlearn_unet, origin_unet, dataloader = accelerator.prepare(
        unlearn_unet, origin_unet,  dataloader
    )

    origin_unet.eval()
    unlearn_unet.eval()
    text_encoder.eval()
    vae.eval()

    progress_bar = tqdm(range(0, len(dataloader)), desc="Calculating FD and PND")

    noise_distances = []
    feature_distances = []
    for step, batch in enumerate(dataloader):
        image = batch['image']
        prompt_ids = batch['prompt_ids']
        with autocast("cuda"):
            input = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
            noise = torch.randn_like(input)
            bsz, *_ = input.shape
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=input.device).long()
            noisy_input = noise_scheduler.add_noise(input, noise, timesteps)

            encoder_hidden_states = encode_prompt(text_encoder, prompt_ids)
            target_modules = [
                'down_blocks.0', 'down_blocks.1', 'down_blocks.2', 'down_blocks.3',
                'mid_block',
                'up_blocks.0', 'up_blocks.1', 'up_blocks.2', 'up_blocks.3'
            ]
            with torch.no_grad():
                noise_pred, intermediate_outputs = get_intermediate_outputs(
                    model=unlearn_unet,
                    inputs={"sample": noisy_input, "timestep": timesteps, "encoder_hidden_states": encoder_hidden_states, "return_dict": False},
                    target_modules=target_modules
                )
                origin_noise_pred, origin_intermediate_outputs = get_intermediate_outputs(
                    model=origin_unet,
                    inputs={"sample": noisy_input, "timestep": timesteps, "encoder_hidden_states": encoder_hidden_states, "return_dict": False},
                    target_modules=target_modules
                )

        noise_distance = batch_distances(noise_pred[0], origin_noise_pred[0])
        noise_distances.extend(noise_distance)
        layer_distances = []
        for output, origin_output in zip(intermediate_outputs, origin_intermediate_outputs):
            layer_distances.append(batch_distances(output, origin_output))

        # layer_distances shape 为(layer_num, batch_size)
        feature_distance = torch.tensor(layer_distances).mean(dim=0).tolist()  # shape 为(batch_size,)
        feature_distances.extend(feature_distance)
        progress_bar.update(1)

    progress_bar.close()
    average_noise_distance = sum(noise_distances) / len(noise_distances)
    average_feature_distance = sum(feature_distances) / len(feature_distances)
    return average_noise_distance, average_feature_distance
