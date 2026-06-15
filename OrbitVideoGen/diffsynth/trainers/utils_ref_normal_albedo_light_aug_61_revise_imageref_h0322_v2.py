import imageio, os, torch, warnings, torchvision, argparse, json, glob
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import numpy as np
import random
import cv2
import math


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("image",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat

        self.base_path = base_path
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.repeat = repeat

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True

        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in tqdm(f):
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]


    def generate_metadata(self, folder):
        image_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            image_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["image"] = image_list
        metadata["prompt"] = prompt_list
        return metadata


    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image


    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width


    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        return image


    def load_data(self, file_path):
        return self.load_image(file_path)


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                if isinstance(data[key], list):
                    path = [os.path.join(self.base_path, p) for p in data[key]]
                    data[key] = [self.load_data(p) for p in path]
                else:
                    path = os.path.join(self.base_path, data[key])
                    data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data


    def __len__(self):
        return len(self.data) * self.repeat

class VideoDataset_vace_pt(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm", "gif"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat

        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True

        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]


    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata


    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image


    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width


    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    def _load_gif(self, file_path):
        gif_img = Image.open(file_path)
        frame_count = 0
        delays, frames = [], []
        while True:
            delay = gif_img.info.get('duration', 100) # ms
            delays.append(delay)
            rgb_frame = gif_img.convert("RGB")
            croped_frame = self.crop_and_resize(rgb_frame, *self.get_height_width(rgb_frame))
            frames.append(croped_frame)
            frame_count += 1
            try:
                gif_img.seek(frame_count)
            except:
                break
        if any((delays[0] != i) for i in delays):
            minimal_interval = min([i for i in delays if i > 0])
            start_end_idx_map = [((sum(delays[:i]), sum(delays[:i+1])), i) for i in range(len(delays))]
            _frames = []
            last_match = 0
            for i in range(sum(delays) // minimal_interval):
                current_time = minimal_interval * i
                for idx, ((start, end), frame_idx) in enumerate(start_end_idx_map[last_match:]):
                    if start <= current_time < end:
                        _frames.append(frames[frame_idx])
                        last_match = idx + last_match
                        break
            frames = _frames
        num_frames = len(frames)
        if num_frames > self.num_frames:
            num_frames = self.num_frames
        else:
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        frames = frames[:num_frames]
        return frames

    def load_video(self, file_path):
        if file_path.lower().endswith(".gif"):
            return self._load_gif(file_path)
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames


    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames


    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension


    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension


    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None



    def __getitem__(self, data_id):
        # === 第一步：尝试使用传入的 data_id ===
        try:
            data = self.data[data_id % len(self.data)].copy()
            path = None  # 用于后续构造 pt_path
            valid = True

            # 加载所有需要的文件
            for key in self.data_file_keys:
                if 'vace_reference_image' in key:
                    data[key] = self.load_data(data[key])
                    continue
                if key in data:
                    path = os.path.join(self.base_path, data[key])
                    loaded_data = self.load_data(path)
                    if loaded_data is None:
                        warnings.warn(f"Failed to load {path}")
                        valid = False
                        break
                    data[key] = loaded_data

                    # 检查 list 长度：>1 但 ≠121 的 list 无效
                    if isinstance(loaded_data, list):
                        if len(loaded_data) > 1 and len(loaded_data) != 121:
                            print('video_length not 121')
                            valid = False
                            break

            if valid:
                # 尝试加载 latent
                if path is not None:
                    video_path = path.replace('videos', f'latents_{self.height}')
                    pt_path = os.path.join(os.path.dirname(video_path), 'latents.pt')

                    if os.path.exists(pt_path):
                        try:
                            pt_dict = torch.load(pt_path, map_location='cpu')
                            data['pre_extracted_vae_feature_dict'] = pt_dict
                        except Exception as e:
                            print(f"Failed to load latents.pt at {pt_path}: {e}")
                            valid = False
                    else:
                        print(f"Latent file not found: {pt_path}")
                        valid = False


            if valid:
                return data  # ✅ 成功：直接返回

        except Exception as e:
            print(f"Error processing data_id {data_id}: {e}")

        # === 第二步：上面失败了，fallback 到随机重试 ===
        max_retries = 100
        for tr in range(max_retries):
            print(tr)
            try:
                random_id = random.randint(0, len(self.data) - 1)
                data = self.data[random_id].copy()
                path = None
                valid = True

                for key in self.data_file_keys:
                    if key in data:
                        path = os.path.join(self.base_path, data[key])
                        loaded_data = self.load_data(path)
                        if loaded_data is None:
                            valid = False
                            break
                        data[key] = loaded_data

                        if isinstance(loaded_data, list):
                            if len(loaded_data) > 1 and len(loaded_data) != 121:
                                print('video length not 121')
                                valid = False
                                break

                if not valid:
                    continue

                # 检查 latent
                if path is not None:
                    video_path = path.replace('videos', f'latents_{self.height}')
                    pt_path = os.path.join(os.path.dirname(video_path), 'latents.pt')

                    if os.path.exists(pt_path):
                        try:
                            pt_dict = torch.load(pt_path, map_location='cpu')
                            data['pre_extracted_vae_feature_dict'] = pt_dict
                        except:
                            continue
                    else:
                        continue
                else:
                    continue

                return data  # ✅ 随机采样成功

            except:
                continue

        # === 极端情况：重试多次仍失败 ===
        raise RuntimeError("Failed to load any valid sample after fallback retries.")


    def __len__(self):
        return len(self.data) * self.repeat


def augment_image(image: Image.Image,
                  scale_range: tuple = (0.9, 1.2),
                  rotate_range: tuple = (-10, 10)) -> Image.Image:
    """
    对输入的 RGBA PIL 图像进行缩放+旋转（单次仿射变换），保持输出尺寸与原图一致。

    优化点：
    1. scale 范围 [0.9, 1.2]，使用 Beta 分布使大部分采样集中在 1.0 附近
    2. 背景填充为 0（黑色/透明），适配 RGBA 输入
    3. 缩放和旋转合并为单次仿射变换，避免两次插值导致的模糊

    参数:
        image (PIL.Image): 输入图像（RGBA）
        scale_range (tuple): 缩放比例范围，默认 (0.9, 1.2)
        rotate_range (tuple): 旋转角度范围，默认 (-10, 10)

    返回:
        PIL.Image: 变换后的图像（RGB），尺寸与输入相同
    """
    # 确保输入是 RGBA
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    width, height = image.size
    cx, cy = width / 2.0, height / 2.0

    # --- 非均匀 scale 采样：Beta(5, 5) 映射到 [0.9, 1.2]，峰值在中点 1.05 附近 ---
    # Beta(5,5) 是对称分布，集中在 0.5 附近，映射后集中在 (0.9+1.2)/2 = 1.05
    # 但我们想让峰值更接近 1.0，所以用 Beta(5, 3)，峰值偏左 ~0.4 -> 映射到 ~1.02
    beta_sample = np.random.beta(5, 3)  # 峰值 ~0.625 -> 映射到 0.9 + 0.625*0.3 = 1.09
    scale = scale_range[0] + beta_sample * (scale_range[1] - scale_range[0])

    # --- 随机旋转角度 ---
    angle_deg = random.uniform(rotate_range[0], rotate_range[1])

    # --- 构造单次仿射变换矩阵（OpenCV: 绕中心旋转+缩放） ---
    img_np = np.array(image)  # H x W x 4 (RGBA)
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)  # 2x3 矩阵

    # 仿射变换，borderValue 填充 0（黑色透明）
    transformed = cv2.warpAffine(
        img_np, M, (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    # 转回 PIL，用 alpha 通道合成到黑色背景
    result_rgba = Image.fromarray(transformed, 'RGBA')
    # 创建黑色背景
    background = Image.new('RGB', (width, height), (0, 0, 0))
    background.paste(result_rgba, (0, 0), result_rgba.split()[3])  # 用 alpha 作为 mask

    return background


def sample_frontal_frame_index(n_total: int, n_front: int = 30) -> int:
    """
    从旋转一圈视频中采样偏正面的帧索引。

    视频是旋转360度的一圈，共 n_total 帧。正面帧在开头和结尾处。
    我们取前 n_front 帧和后 n_front 帧作为候选池（对应正面 180 度范围），
    然后用非均匀采样使概率更多落在离 0 和离结尾更近的帧（即更正面的视角）。

    具体方法：将候选帧映射到 [0, 180] 度角度，用 cos 分布使正面（0度/180度端点）
    概率更高，侧面（90度中间）概率更低。

    参数:
        n_total: 视频总帧数
        n_front: 从头尾各取多少帧，默认 30

    返回:
        int: 采样到的帧索引 (0-indexed)
    """
    # 构建候选帧索引列表：前 n_front + 后 n_front
    front_indices = list(range(min(n_front, n_total)))
    back_indices = list(range(max(0, n_total - n_front), n_total))
    # 去重（当 n_total <= 2*n_front 时可能有重叠）
    candidates = sorted(set(front_indices + back_indices))

    if len(candidates) == 0:
        return 0
    if len(candidates) == 1:
        return candidates[0]

    # 计算每个候选帧到最近端点（0 或 n_total-1）的距离
    # 距离越小 = 越正面 = 权重越高
    n = len(candidates)
    weights = []
    for idx in candidates:
        # 到开头和结尾的最小距离
        dist_to_front = idx
        dist_to_back = n_total - 1 - idx
        min_dist = min(dist_to_front, dist_to_back)
        # 将距离映射到角度 [0, 90]，然后用 cos 使小距离（正面）权重高
        # max_dist = n_front, 归一化到 [0, pi/2]
        angle = (min_dist / max(n_front, 1)) * (math.pi / 2)
        w = math.cos(angle) + 0.1  # +0.1 保底，避免权重为 0
        weights.append(w)

    # 归一化为概率分布
    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    chosen = np.random.choice(len(candidates), p=probs)
    return candidates[chosen]


def is_image_near_black(image: Image.Image, threshold=10) -> bool:
    """
    判断图像是否接近全黑。

    Args:
        image: PIL 图像对象
        threshold: 像素值阈值 (0~255)，建议设为 5~15，默认 10

    Returns:
        bool: 如果图像所有通道的所有像素都 <= 阈值，返回 True
    """
    img_array = np.array(image)
    return np.all(img_array <= threshold)

class VideoDataset(torch.utils.data.Dataset):
    """
    [v2] 视频加载 + RGBA 图片 ref image。

    改进点（vs v1）：
    1. augment_image: scale [0.9, 1.2] Beta分布 + 单次仿射变换 + 黑色填充
    2. ref image fallback: 前30+后30帧，cos权重非均匀采样偏正面
    3. ref image 优先从 render PNG 图片序列加载（RGBA，保留alpha）
       其余数据（video, control_video, normal, albedo）仍从 .mp4 视频加载
    """

    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm", "gif"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat

        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True

        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]


    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata


    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image


    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width

    def get_height_width_half(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width // 2

    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
        return num_frames

    def load_video(self, file_path):
        """从 mp4 加载视频帧"""
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        frames = frames + frames[:1]
        reader.close()
        return frames

    def load_render_image_sequence(self, sample_dir, prefix='render', start_idx=1, num_frames=120):
        """
        加载 render PNG 图片序列（RGBA，保留 alpha 通道）。
        用于 ref image 的 fallback 加载。

        Args:
            sample_dir: 图片目录
            prefix: 文件前缀，默认 'render'
            start_idx: 起始索引，默认 1 (1-indexed)
            num_frames: 最大帧数

        Returns:
            list[PIL.Image]: 帧列表（保持原始 mode，不转 RGB）
        """
        frames = []
        for i in range(start_idx, start_idx + num_frames):
            img_path = os.path.join(sample_dir, f'{prefix}_{i:04d}.png')
            if not os.path.exists(img_path):
                break
            image = Image.open(img_path)
            image = self.crop_and_resize(image, *self.get_height_width(image))
            frames.append(image)
        return frames if len(frames) > 0 else None

    def _cos_weighted_sample(self, candidate_indices):
        """从候选帧索引中 cos 加权非均匀采样，偏正面帧（两端概率高）"""
        n = len(candidate_indices)
        if n <= 1:
            return candidate_indices[0]
        weights = []
        for i in range(n):
            dist_to_end = min(i, n - 1 - i)
            angle = (dist_to_end / max(n // 2, 1)) * (math.pi / 2)
            w = math.cos(angle) + 0.1
            weights.append(w)
        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        chosen_pos = np.random.choice(n, p=probs)
        return candidate_indices[chosen_pos]

    def load_data(self, file_path):
        """加载单张图片或视频"""
        ext = file_path.split(".")[-1].lower()
        if ext in self.image_file_extension:
            image = Image.open(file_path)
            image = self.crop_and_resize(image, *self.get_height_width(image))
            return [image]
        elif ext in self.video_file_extension:
            return self.load_video(file_path)
        else:
            return None


    def __getitem__(self, data_id):
        max_retry = 10
        retry_count = 0

        while retry_count < max_retry:
            try:
                if retry_count > 0:
                    data_id = random.randrange(0, len(self.data))

                data = self.data[data_id % len(self.data)].copy()

                # ========== 加载主数据（视频加载） ==========
                for key in self.data_file_keys:
                    if key in data:
                        path = os.path.join(self.base_path, data[key])
                        data[key] = self.load_video(path)
                        print(key, len(data[key]))
                        if data[key] is None:
                            warnings.warn(f"cannot load file {data[key]}.")
                            raise Exception(f"Failed to load {key}")

                # 加载预提取的 VAE 特征
                video_path = path.replace('videos', f'latents_{self.height}')
                pt_path = os.path.join(os.path.dirname(video_path), 'latents.pt')

                if os.path.exists(pt_path):
                    try:
                        pt_dict = torch.load(pt_path)
                        data['pre_extracted_vae_feature_dict'] = pt_dict
                    except Exception as e:
                        print(f'load pt error: {e}')
                        raise Exception("Failed to load pt file")

                assert len(data['video']) == 121
                assert len(data['video']) == len(data['control_video'])

                # 加载 attn mask
                attn_path = os.path.join(os.path.dirname(__file__), '..', '..', 'global_61hv_512.npy')
                assert os.path.exists(attn_path)
                attn_mask = np.load(attn_path)
                attn_mask = torch.from_numpy(attn_mask)
                data['attn_mask'] = attn_mask

                # ========== 加载 ref image（三个可选源按权重采样） ==========
                sample_dir = os.path.dirname(path)

                # 源 A: objaverse_60k_120h/images_curve (1-indexed, 120帧) — 60%
                ref_dir_A = sample_dir.replace('videos_curve/', 'images_curve/')
                has_A = os.path.exists(os.path.join(ref_dir_A, 'render_0001.png'))

                # 源 B: objavers_ref_random2 (000.png - 007.png) — 20%
                ref_dir_B = sample_dir.replace('objaverse_60k_120h/videos_curve/', 'objavers_ref_random2/')
                has_B = os.path.exists(os.path.join(ref_dir_B, '000.png'))

                # 源 C: objaverse_60k_1001_customh_3pointlight/images_curve (0-indexed, 30帧) — 20%
                ref_dir_C = sample_dir.replace(
                    'objaverse_60k_120h/videos_curve/',
                    'objaverse_60k_1001_customh_3pointlight/images_curve/'
                )
                has_C = os.path.exists(os.path.join(ref_dir_C, 'render_0000.png'))

                # 按可用性构建加权候选列表，不可用的权重归零后重新归一化
                sources = []  # (name, weight)
                if has_A: sources.append(('A', 0.6))
                if has_B: sources.append(('B', 0.2))
                if has_C: sources.append(('C', 0.2))

                if len(sources) == 0:
                    raise Exception("No ref image source available")

                src_names = [s[0] for s in sources]
                src_weights = np.array([s[1] for s in sources])
                src_weights = src_weights / src_weights.sum()
                chosen_src = src_names[np.random.choice(len(src_names), p=src_weights)]

                if chosen_src == 'A':
                    # objaverse_60k_120h/images_curve: 前30+后30, cos加权, 只加载1帧
                    total_A = 0
                    for i in range(1, 121):
                        if os.path.exists(os.path.join(ref_dir_A, f'render_{i:04d}.png')):
                            total_A += 1
                        else:
                            break
                    front_indices = list(range(1, min(31, total_A + 1)))
                    back_indices = list(range(max(1, total_A - 29), total_A + 1))
                    candidate_indices = front_indices + [i for i in back_indices if i not in front_indices]
                    chosen_frame_idx = self._cos_weighted_sample(candidate_indices)
                    ref_path = os.path.join(ref_dir_A, f'render_{chosen_frame_idx:04d}.png')
                    ref_frame = Image.open(ref_path)
                    ref_frame = self.crop_and_resize(ref_frame, *self.get_height_width(ref_frame))

                elif chosen_src == 'B':
                    # objavers_ref_random2: 随机选1张
                    frame_idx = random.randint(0, 7)
                    ref_path = os.path.join(ref_dir_B, f'{frame_idx:03d}.png')
                    if not os.path.exists(ref_path):
                        ref_path = os.path.join(ref_dir_B, '000.png')
                    ref_frame = Image.open(ref_path)
                    ref_frame = self.crop_and_resize(ref_frame, *self.get_height_width(ref_frame))

                else:  # chosen_src == 'C'
                    # 3pointlight: 前30帧(0-indexed), cos加权, 只加载1帧
                    total_C = 0
                    for i in range(0, 30):
                        if os.path.exists(os.path.join(ref_dir_C, f'render_{i:04d}.png')):
                            total_C += 1
                        else:
                            break
                    candidate_indices = list(range(total_C))
                    chosen_frame_idx = self._cos_weighted_sample(candidate_indices)
                    ref_path = os.path.join(ref_dir_C, f'render_{chosen_frame_idx:04d}.png')
                    ref_frame = Image.open(ref_path)
                    ref_frame = self.crop_and_resize(ref_frame, *self.get_height_width(ref_frame))

                reference_image = augment_image(ref_frame)
                data['random_ref_image'] = reference_image

                # ========== 加载 normal 和 albedo（视频加载） ==========
                normal_path = os.path.join(sample_dir, 'normal.mp4')
                normal_data = self.load_video(normal_path)[:self.num_frames]
                assert len(normal_data) == len(data['control_video'])
                data['normal'] = normal_data

                albedo_path = os.path.join(sample_dir, 'albedo.mp4')
                albedo_data = self.load_video(albedo_path)[:self.num_frames]
                assert len(albedo_data) == len(data['control_video'])
                black = is_image_near_black(albedo_data[0], threshold=10)
                if black:
                    print('black!!!!!!!!!!!!!!!!')
                assert not black
                data['albedo'] = albedo_data

                return data

            except Exception as e:
                retry_count += 1
                print(f"Attempt {retry_count} failed: {e}")
                if retry_count >= max_retry:
                    raise Exception(f"Failed to load data after {max_retry} attempts")



    def __len__(self):
        return len(self.data) * self.repeat



class CustomVideoDataset0(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video", "condition_video"),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm", "gif"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat

        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True

        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]


    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata


    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image


    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width


    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    def _load_gif(self, file_path):
        gif_img = Image.open(file_path)
        frame_count = 0
        delays, frames = [], []
        while True:
            delay = gif_img.info.get('duration', 100) # ms
            delays.append(delay)
            rgb_frame = gif_img.convert("RGB")
            croped_frame = self.crop_and_resize(rgb_frame, *self.get_height_width(rgb_frame))
            frames.append(croped_frame)
            frame_count += 1
            try:
                gif_img.seek(frame_count)
            except:
                break
        if any((delays[0] != i) for i in delays):
            minimal_interval = min([i for i in delays if i > 0])
            start_end_idx_map = [((sum(delays[:i]), sum(delays[:i+1])), i) for i in range(len(delays))]
            _frames = []
            last_match = 0
            for i in range(sum(delays) // minimal_interval):
                current_time = minimal_interval * i
                for idx, ((start, end), frame_idx) in enumerate(start_end_idx_map[last_match:]):
                    if start <= current_time < end:
                        _frames.append(frames[frame_idx])
                        last_match = idx + last_match
                        break
            frames = _frames
        num_frames = len(frames)
        if num_frames > self.num_frames:
            num_frames = self.num_frames
        else:
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        frames = frames[:num_frames]
        return frames

    def load_video(self, file_path):
        if file_path.lower().endswith(".gif"):
            return self._load_gif(file_path)
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames


    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames


    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension


    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension


    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()

        for key in self.data_file_keys:
            if key in data:
                file_name = data[key]
                file_path = file_name if file_name[0] == '/' else os.path.join(self.base_path, file_name)
                frames = self.load_data(file_path)
                if frames is None:
                    warnings.warn(f"Failed to load {key}: {file_path}")
                    return None
                data[key] = frames  # list of PIL.Image

        return data

    def __len__(self):
        return len(self.data) * self.repeat



class CustomVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None,
        metadata_path=None,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        max_pixels=1920 * 1080,
        height=None,
        width=None,
        height_division_factor=16,
        width_division_factor=16,
        data_file_keys=("video", "condition_video"),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm", "gif"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat

        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = tuple(ext.lower() for ext in image_file_extension)
        self.video_file_extension = tuple(ext.lower() for ext in video_file_extension)
        self.repeat = repeat

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
        else:
            raise ValueError("Either both height and width should be None, or both should be specified.")

        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext = file_name.split(".")[-1].lower()
            file_base = file_name.rsplit(".", 1)[0]
            if file_ext not in self.image_file_extension and file_ext not in self.video_file_extension:
                continue
            prompt_file = file_base + ".txt"
            if prompt_file not in file_set:
                continue
            with open(os.path.join(folder, prompt_file), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        return pd.DataFrame({"video": video_list, "prompt": prompt_list})

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        new_size = (round(width * scale), round(height * scale))
        image = torchvision.transforms.functional.resize(
            image, new_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image

    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
            return height, width
        else:
            return self.height, self.width

    def get_height_width_half(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
            return height, width
        else:
            return self.height, self.width // 2

    def get_num_frames(self, reader):
        total = int(reader.count_frames())
        num_frames = self.num_frames
        if total < num_frames:
            num_frames = total
        while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
            num_frames -= 1
        return num_frames

    def _load_gif(self, file_path):
        gif_img = Image.open(file_path)
        frames, delays = [], []
        frame_count = 0
        while True:
            delays.append(gif_img.info.get('duration', 100))
            rgb_frame = gif_img.convert("RGB")
            resized_frame = self.crop_and_resize(rgb_frame, *self.get_height_width(rgb_frame))
            frames.append(resized_frame)
            frame_count += 1
            try:
                gif_img.seek(frame_count)
            except EOFError:
                break

        if len(set(delays)) > 1:
            min_delay = min(d for d in delays if d > 0)
            total_duration = sum(delays)
            num_uniform_frames = total_duration // min_delay
            start_end_map = [(sum(delays[:i]), sum(delays[:i+1])) for i in range(len(delays))]
            uniform_frames = []
            for i in range(num_uniform_frames):
                t = min_delay * i
                for idx, (start, end) in enumerate(start_end_map):
                    if start <= t < end:
                        uniform_frames.append(frames[idx])
                        break
            frames = uniform_frames

        num_frames = len(frames)
        if num_frames > self.num_frames:
            num_frames = self.num_frames
        else:
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return frames[:num_frames]

    def load_video(self, file_path):
        if file_path.lower().endswith(".gif"):
            return self._load_gif(file_path)
        try:
            reader = imageio.get_reader(file_path)
            num_frames = self.get_num_frames(reader)
            frames = []
            for frame_id in range(num_frames):
                frame = reader.get_data(frame_id)
                pil_image = Image.fromarray(frame)
                resized = self.crop_and_resize(pil_image, *self.get_height_width(pil_image))
                frames.append(resized)
            frames = frames + frames[:1]
            reader.close()
            return frames
        except Exception as e:
            warnings.warn(f"Error reading video {file_path}: {str(e)}")
            return None

    def load_image(self, file_path):
        try:
            image = Image.open(file_path).convert("RGB")
            image = self.crop_and_resize(image, *self.get_height_width(image))
            return [image]
        except Exception as e:
            warnings.warn(f"Error reading image {file_path}: {str(e)}")
            return None

    def is_image(self, file_path):
        ext = file_path.split(".")[-1].lower()
        return ext in self.image_file_extension

    def is_video(self, file_path):
        ext = file_path.split(".")[-1].lower()
        return ext in self.video_file_extension

    def load_data(self, file_path):
        if not os.path.exists(file_path):
            warnings.warn(f"File not found: {file_path}")
            return None
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            warnings.warn(f"Unsupported file type: {file_path}")
            return None

    def __getitem__(self, data_id):
        max_attempts = 10
        original_id = data_id % len(self.data)
        attempt = 0

        while attempt < max_attempts:
            current_id = (original_id + attempt) % len(self.data)
            data = self.data[current_id].copy()

            try:
                loaded_data = {}
                valid = True

                for key in self.data_file_keys:
                    if key not in data:
                        warnings.warn(f"Missing key '{key}' in data.")
                        valid = False
                        break

                    file_name = data[key]

                    file_path = file_name if file_name.startswith('/') else os.path.join(self.base_path, file_name)

                    frames = self.load_data(file_path)
                    if frames is None:
                        warnings.warn(f"Failed to load {key}: {file_path}")
                        valid = False
                        break

                    loaded_data[key] = frames

                if valid:
                    v_frames = loaded_data.get("video")
                    c_frames = loaded_data.get("condition_video")
                    if v_frames is None or c_frames is None:
                        valid = False
                    else:
                        if len(v_frames) != len(c_frames):
                            warnings.warn(
                                f"Frame count mismatch: video={len(v_frames)}, condition_video={len(c_frames)} "
                                f"| Files: {data['video']} and {data.get('condition_video', 'unknown')}"
                            )
                            valid = False

                    if valid:
                        data.update(loaded_data)
                        return data

            except Exception as e:
                warnings.warn(f"Exception in __getitem__ for id {current_id}: {str(e)}")
                valid = False

            attempt += 1

        warnings.warn(f"Failed to load valid sample after {max_attempts} attempts (data_id={data_id}).")
        return None

    def __len__(self):
        return len(self.data) * self.repeat


class CustomVideoDataset_with_pt(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None,
        metadata_path=None,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        max_pixels=1920 * 1080,
        height=None,
        width=None,
        height_division_factor=16,
        width_division_factor=16,
        data_file_keys=("video", "condition_video"),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm", "gif"),
        repeat=1,
        args=None,
        cond_name='depth'
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat

        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = tuple(ext.lower() for ext in image_file_extension)
        self.video_file_extension = tuple(ext.lower() for ext in video_file_extension)
        self.repeat = repeat
        self.cond_name = cond_name

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
        else:
            raise ValueError("Either both height and width should be None, or both should be specified.")

        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext = file_name.split(".")[-1].lower()
            file_base = file_name.rsplit(".", 1)[0]
            if file_ext not in self.image_file_extension and file_ext not in self.video_file_extension:
                continue
            prompt_file = file_base + ".txt"
            if prompt_file not in file_set:
                continue
            with open(os.path.join(folder, prompt_file), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        return pd.DataFrame({"video": video_list, "prompt": prompt_list})

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        new_size = (round(width * scale), round(height * scale))
        image = torchvision.transforms.functional.resize(
            image, new_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image

    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
            return height, width
        else:
            return self.height, self.width

    def get_num_frames(self, reader):
        total = int(reader.count_frames())
        num_frames = self.num_frames
        if total < num_frames:
            num_frames = total
        while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
            num_frames -= 1
        return num_frames

    def _load_gif(self, file_path):
        gif_img = Image.open(file_path)
        frames, delays = [], []
        frame_count = 0
        while True:
            delays.append(gif_img.info.get('duration', 100))
            rgb_frame = gif_img.convert("RGB")
            resized_frame = self.crop_and_resize(rgb_frame, *self.get_height_width(rgb_frame))
            frames.append(resized_frame)
            frame_count += 1
            try:
                gif_img.seek(frame_count)
            except EOFError:
                break

        if len(set(delays)) > 1:
            min_delay = min(d for d in delays if d > 0)
            total_duration = sum(delays)
            num_uniform_frames = total_duration // min_delay
            start_end_map = [(sum(delays[:i]), sum(delays[:i+1])) for i in range(len(delays))]
            uniform_frames = []
            for i in range(num_uniform_frames):
                t = min_delay * i
                for idx, (start, end) in enumerate(start_end_map):
                    if start <= t < end:
                        uniform_frames.append(frames[idx])
                        break
            frames = uniform_frames

        num_frames = len(frames)
        if num_frames > self.num_frames:
            num_frames = self.num_frames
        else:
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return frames[:num_frames]

    def load_video(self, file_path):
        if file_path.lower().endswith(".gif"):
            return self._load_gif(file_path)

        try:
            reader = imageio.get_reader(file_path)
            num_frames = self.get_num_frames(reader)
            frames = []
            for frame_id in range(num_frames):
                frame = reader.get_data(frame_id)
                pil_image = Image.fromarray(frame)
                resized = self.crop_and_resize(pil_image, *self.get_height_width(pil_image))
                frames.append(resized)
            reader.close()
            return frames
        except Exception as e:
            warnings.warn(f"Error reading video {file_path}: {str(e)}")
            return None

    def load_image(self, file_path):
        try:
            image = Image.open(file_path).convert("RGB")
            image = self.crop_and_resize(image, *self.get_height_width(image))
            return [image]
        except Exception as e:
            warnings.warn(f"Error reading image {file_path}: {str(e)}")
            return None

    def is_image(self, file_path):
        ext = file_path.split(".")[-1].lower()
        return ext in self.image_file_extension

    def is_video(self, file_path):
        ext = file_path.split(".")[-1].lower()
        return ext in self.video_file_extension

    def load_data(self, file_path):
        if not os.path.exists(file_path):
            warnings.warn(f"File not found: {file_path}")
            return None
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            warnings.warn(f"Unsupported file type: {file_path}")
            return None

    def __getitem__(self, data_id):
        max_attempts = 5
        original_id = data_id % len(self.data)
        attempt = 0

        while attempt < max_attempts:
            current_id = (original_id + attempt) % len(self.data)
            data = self.data[current_id].copy()

            try:
                loaded_data = {}
                valid = True

                for key in self.data_file_keys:
                    if key not in data:
                        warnings.warn(f"Missing key '{key}' in data.")
                        valid = False
                        break

                    file_name = data[key]

                    file_path = file_name if file_name.startswith('/') else os.path.join(self.base_path, file_name)

                    video_path = os.path.join(self.base_path, data['video']).replace('videos', f'latents_{self.height}')
                    pt_path = os.path.join(os.path.dirname(video_path), 'latents.pt')

                    if 'condition_video' in key and os.path.exists(pt_path):
                        pt_dict = torch.load(pt_path)
                        loaded_data['condition_video'] = (pt_dict['position'] + pt_dict['normal']) if self.cond_name == 'pn' else pt_dict[self.cond_name]
                    else:
                        frames = self.load_data(file_path)

                        if frames is None:
                            warnings.warn(f"Failed to load {key}: {file_path}")
                            valid = False
                            break

                        loaded_data[key] = frames



                if valid:
                    v_frames = loaded_data.get("video")
                    c_frames = loaded_data.get("condition_video")

                    if v_frames is None or c_frames is None:
                        valid = False
                    else:
                        if isinstance(v_frames, torch.Tensor):
                            v_frame_count = v_frames.shape[-4] if len(v_frames.shape) >= 4 else 0
                            v_height = v_frames.shape[-2] if len(v_frames.shape) >= 2 else 0
                            v_width = v_frames.shape[-1] if len(v_frames.shape) >= 1 else 0
                        elif isinstance(v_frames, list) and len(v_frames) > 0:
                            v_frame_count = len(v_frames)
                            if hasattr(v_frames[0], 'size'):
                                v_width, v_height = v_frames[0].size
                            else:
                                v_height, v_width = 0, 0
                        else:
                            v_frame_count, v_height, v_width = 0, 0, 0

                        if isinstance(c_frames, torch.Tensor):
                            c_frame_count = (c_frames.shape[-3]-1)*4+1 if len(c_frames.shape) >= 3 else 0
                            c_height = c_frames.shape[-2]*16 if len(c_frames.shape) >= 2 else 0
                            c_width = c_frames.shape[-1]*16 if len(c_frames.shape) >= 1 else 0
                        elif isinstance(c_frames, list) and len(c_frames) > 0:
                            c_frame_count = len(c_frames)
                            if hasattr(c_frames[0], 'size'):
                                c_width, c_height = c_frames[0].size
                            else:
                                c_height, c_width = 0, 0
                        else:
                            c_frame_count, c_height, c_width = 0, 0, 0

                        if v_frame_count != c_frame_count:
                            warnings.warn(f"Frame count mismatch: video has {v_frame_count} frames, "
                                        f"condition_video has {c_frame_count} frames")
                            valid = False
                        elif v_height != c_height or v_width != c_width:
                            warnings.warn(f"Spatial size mismatch: video is {v_width}x{v_height}, "
                                        f"condition_video is {c_width}x{c_height}")
                            valid = False

                    if valid:
                        data.update(loaded_data)
                        return data


            except Exception as e:
                warnings.warn(f"Exception in __getitem__ for id {current_id}: {str(e)}")
                valid = False

            attempt += 1

        warnings.warn(f"Failed to load valid sample after {max_attempts} attempts (data_id={data_id}).")
        return None

    def __len__(self):
        return len(self.data) * self.repeat


class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()


    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self


    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules


    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names


    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        if upcast_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)
        return model


    def mapping_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
        return new_state_dict


    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict



class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0


    def on_step_end(self, accelerator, model, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator, model, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)

            model_unwrap = accelerator.unwrap_model(model)
            state_dict = model_unwrap.export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)

            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)



def custom_launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_workers: int = 8,
    save_steps: int = None,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=lambda x: x[0],
        num_workers=num_workers
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)



def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_workers: int = 8,
    save_steps: int = None,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
):
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(model: DiffusionTrainingModule, dataset, output_path="./models"):
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0])
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    os.makedirs(os.path.join(output_path, "data_cache"), exist_ok=True)
    for data_id, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            inputs = model.forward_preprocess(data)
            inputs = {key: inputs[key] for key in model.model_input_keys if key in inputs}
            torch.save(inputs, os.path.join(output_path, "data_cache", f"{data_id}.pth"))



def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1280*720, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument("--data_file_keys", type=str, default="image,video,condition_video", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--cond_name", type=str, default="depth", help="cond type.")
    parser.add_argument("--resume", type=str, default="depth", help="cond type.")
    return parser



def flux_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--align_to_opensource_format", default=False, action="store_true", help="Whether to align the lora format to opensource format. Only for DiT's LoRA.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    return parser



def qwen_image_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Paths to tokenizer.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--processor_path", type=str, default=None, help="Path to the processor. If provided, the processor will be used for image editing.")
    parser.add_argument("--enable_fp8_training", default=False, action="store_true", help="Whether to enable FP8 training. Only available for LoRA training on a single GPU.")
    return parser
