import imageio, os, torch, warnings, torchvision, argparse, json, glob
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import numpy as np
import random

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
        # delays canbe used to calculate framerates
        # i guess it is better to sample images with stable interval,
        # and using minimal_interval as the interval, 
        # and framerate = 1000 / minimal_interval
        if any((delays[0] != i) for i in delays):
            minimal_interval = min([i for i in delays if i > 0])
            # make a ((start,end),frameid) struct
            start_end_idx_map = [((sum(delays[:i]), sum(delays[:i+1])), i) for i in range(len(delays))]
            _frames = []
            # according gemini-code-assist, make it more efficient to locate
            # where to sample the frame
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
                        # import pdb;pdb.set_trace()
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
                  scale_range: tuple = (0.6, 1.2),
                  rotate_range: tuple = (-10, 10)) -> Image.Image:
    """
    对输入的PIL图像进行缩放（带pad/裁剪）和旋转，保持输出图像尺寸与原图一致
    
    参数:
        image (PIL.Image): 输入图像
        scale_range (tuple): 缩放比例范围，如 (0.9, 1.1)
        rotate_range (tuple): 旋转角度范围，如 (-10, 10)
    
    返回:
        PIL.Image: 变换后的图像，尺寸与输入相同
    """
    width, height = image.size
    center_x, center_y = width // 2, height // 2

    # 随机生成缩放比例
    scale = random.uniform(scale_range[0], scale_range[1])
    
    # 计算缩放后的尺寸
    new_width = int(width * scale)
    new_height = int(height * scale)

    # 缩放图像
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 创建一个与原图同尺寸的画布，填充为白色背景
    padded_image = Image.new("RGB", (width, height), (255, 255, 255))

    # 计算将缩放后的图像居中放置的位置
    paste_x = (width - new_width) // 2
    paste_y = (height - new_height) // 2

    # 将缩放后的图像粘贴到中心（如果变小则 padding；如果变大则会溢出，先不处理）
    # 注意：paste 不允许负坐标，所以太大时不能直接居中贴
    if new_width <= width and new_height <= height:
        # 情况1：缩小了，可以居中贴，其余 padding 白色（已初始化）
        padded_image.paste(resized_image, (paste_x, paste_y))
        temp_image = padded_image
    else:
        # 情况2：放大了，resized_image 比原图大，不能直接 paste
        # 我们先创建一个居中裁剪区域
        left = (new_width - width) // 2
        top = (new_height - height) // 2
        right = left + width
        bottom = top + height
        # 从放大的图像中裁剪出中间部分
        temp_image = resized_image.crop((left, top, right, bottom))

    # 现在 temp_image 是缩放后居中对齐、尺寸与原图一致的图像

    # 随机生成旋转角度
    angle = random.uniform(rotate_range[0], rotate_range[1])

    # 旋转图像，并设置 expand=False 以保持尺寸不变
    # 使用 expand=False 时，旋转会以中心为中心，超出边缘会被裁剪
    rotated_image = temp_image.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(255, 255, 255)  # 边缘填充白色
    )

    return rotated_image
def is_image_near_black(image: Image.Image, threshold=10) -> bool:
    """
    判断图像是否接近全黑。
    
    Args:
        image: PIL 图像对象
        threshold: 像素值阈值 (0~255)，建议设为 5~15，默认 10
    
    Returns:
        bool: 如果图像所有通道的所有像素都 <= 阈值，返回 True
    """
    # 转换为 numpy 数组以便处理
    img_array = np.array(image)
    
    # 如果是多通道图 (H, W, C)，检查所有通道
    return np.all(img_array <= threshold)

class VideoDataset(torch.utils.data.Dataset):
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
            # while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
            #     num_frames -= 1
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
        # delays canbe used to calculate framerates
        # i guess it is better to sample images with stable interval,
        # and using minimal_interval as the interval, 
        # and framerate = 1000 / minimal_interval
        if any((delays[0] != i) for i in delays):
            minimal_interval = min([i for i in delays if i > 0])
            # make a ((start,end),frameid) struct
            start_end_idx_map = [((sum(delays[:i]), sum(delays[:i+1])), i) for i in range(len(delays))]
            _frames = []
            # according gemini-code-assist, make it more efficient to locate
            # where to sample the frame
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
            frame = self.crop_and_resize(frame, *self.get_height_width_half(frame))
            frames.append(frame)
        # print('load video', len(frames))
        frames = frames + frames[:1]
        reader.close()
        return frames

    def load_video_half(self, file_path):
        if file_path.lower().endswith(".gif"):
            return self._load_gif(file_path)
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(0, num_frames, 2):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        # print('load video', len(frames))
        frames = frames + frames[:1]
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
        max_retry = 10  # 设置最大重试次数，避免无限循环
        retry_count = 0
        
        while retry_count < max_retry:
            try:
                # 随机选择一个数据ID
                if retry_count > 0:  # 如果是重试，随机选择一个新的数据
                    data_id = random.randrange(0, len(self.data))
                
                data = self.data[data_id % len(self.data)].copy()
                
                # 加载数据文件
                for key in self.data_file_keys:
                    if key in data:
                        path = os.path.join(self.base_path, data[key])
                        data[key] = self.load_video_half(path)
                        print(key, len(data[key]))
                        if data[key] is None:
                            warnings.warn(f"cannot load file {data[key]}.")
                            raise Exception(f"Failed to load {key}")
                
                # 加载预提取的VAE特征
                video_path = path.replace('videos', f'latents_{self.height}')
                pt_path = os.path.join(os.path.dirname(video_path), 'latents.pt')
                            
                if os.path.exists(pt_path):
                    try:
                        pt_dict = torch.load(pt_path)
                        data['pre_extracted_vae_feature_dict'] = pt_dict
                    except Exception as e:
                        print(f'load pt error: {e}')
                        raise Exception("Failed to load pt file")
                
                # 加载随机参考图像
                
                # ref_path = os.path.join(os.path.dirname(path), 'rgb.mp4')
                # ref_data = self.load_data(ref_path)
                # if ref_data is None:
                #     raise Exception("Failed to load reference image")
                assert len(data['video']) == 61
                assert len(data['video']) == len(data['control_video']) 



                attn_path = os.path.join(os.path.dirname(__file__), '..', '..', 'global_61hv_512.npy')
                assert os.path.exists(attn_path)
                attn_mask = np.load(attn_path)
                attn_mask = torch.from_numpy(attn_mask)  
                data['attn_mask'] = attn_mask




                tmp_path = os.path.dirname(path).replace('objaverse_60k_120hv000/videos_curve/','objavers_ref_random2/')
                
                frame_idx = random.randint(0, 7)
                ref_path0 = os.path.join(tmp_path, f'{frame_idx:03d}.png')
                

                if not os.path.exists(ref_path0):
                    # print('choose pre ref')
                    # import pdb;pdb.set_trace()
                    tmp_path = os.path.dirname(path).replace('objaverse_60k_120hv/','objaverse_60k_1001_customh/')
                    # tmp_path = tmp_path.replace('objaverse_60k_120hv/','objaverse_60k_1001_customh_3pointlight/')
                    ref_path2 = os.path.join(tmp_path, 'rgb.mp4')
                    ref_data2 = self.load_data(ref_path2)
                    if ref_data2 is None:
                        raise Exception("Failed to load reference light image")
                    
                    # tmp_path = os.path.dirname(path).replace('objaverse_60k_1001_customhv_flip/','objaverse_60k_1001_customh/')
                    # ref_path3 = os.path.join(tmp_path, 'rgb.mp4')
                    # ref_data3 = self.load_data(ref_path3)
                    # if ref_data3 is None:
                    #     raise Exception("Failed to load reference light image")

                    # tmp_path = os.path.dirname(path).replace('objaverse_60k_1001_customhv_flip/','objaverse_60k_1001_customv/')
                    # ref_path4 = os.path.join(tmp_path, 'rgb.mp4')
                    # ref_data4 = self.load_data(ref_path4)
                    # if ref_data4 is None:
                    #     raise Exception("Failed to load reference light image")
                                
                    ref_data = ref_data2[:7] + ref_data2[-7:]
  
                else:
                    ref_data = self.load_data(ref_path0)
                    print('choose ref', ref_data) 

                choose_index = random.randrange(0, len(ref_data)) 
                ref_data = ref_data[choose_index] 
                reference_image = augment_image(ref_data)
                
                W, H = reference_image.size
                # print(W, H)
                concatenated_image = Image.new('RGB', (W * 2, H))
                concatenated_image.paste(reference_image, (0, 0))      # 左边
                concatenated_image.paste(reference_image, (W, 0))      # 右边
                data['random_ref_image'] = concatenated_image
                # concatenated_image.save('ztest2.png')
                
                
                normal_path = os.path.join(os.path.dirname(path), 'normal.mp4')
                normal_data = self.load_video_half(normal_path)[:self.num_frames]
                assert len(normal_data) == len(data['control_video']) 
                data['normal'] = normal_data

                albedo_path = os.path.join(os.path.dirname(path), 'albedo.mp4')
                albedo_data = self.load_video_half(albedo_path)[:self.num_frames]
                assert len(albedo_data) == len(data['control_video']) 
                black = is_image_near_black(albedo_data[0], threshold=10) 
                if black:
                    print('black!!!!!!!!!!!!!!!!')
                assert not black
                data['albedo'] = albedo_data


                # import pdb;pdb.set_trace()
                path_dir = os.path.dirname(path)
                parts = os.path.normpath(path_dir).split(os.sep)[-2:]
                conf = os.path.join(*parts)
  
                # data['vertices'] = vertices
                # data['faces'] = faces
                # data['uvs'] = uvs
                # data['projections'] = projections
                # data['views'] = views
                # print('albedo', data['albedo'][0])
                # print('control_video', data['control_video'][0])
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
        # data_file_keys=("video",),
        data_file_keys=("video", "condition_video"),  # 👈 修改这里！
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
        # delays canbe used to calculate framerates
        # i guess it is better to sample images with stable interval,
        # and using minimal_interval as the interval, 
        # and framerate = 1000 / minimal_interval
        if any((delays[0] != i) for i in delays):
            minimal_interval = min([i for i in delays if i > 0])
            # make a ((start,end),frameid) struct
            start_end_idx_map = [((sum(delays[:i]), sum(delays[:i+1])), i) for i in range(len(delays))]
            _frames = []
            # according gemini-code-assist, make it more efficient to locate
            # where to sample the frame
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


    # def __getitem__(self, data_id):
    #     data = self.data[data_id % len(self.data)].copy()
    #     for key in self.data_file_keys:
    #         if key in data:
    #             path = os.path.join(self.base_path, data[key])
    #             data[key] = self.load_data(path)
    #             if data[key] is None:
    #                 warnings.warn(f"cannot load file {data[key]}.")
    #                 return None
    #     return data
    def __getitem__(self, data_id):
        # import pdb;pdb.set_trace()
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
        """
        Args:
            base_path: 默认根路径
            metadata_path: metadata 文件路径 (csv/json)
            num_frames: 目标帧数
            time_division_factor: 时间维度对齐因子，要求 num_frames % factor == remainder
            time_division_remainder: 同上，余数
            max_pixels: 动态分辨率时最大像素数
            height, width: 固定分辨率（若指定，则关闭动态）
            height_division_factor, width_division_factor: 分辨率对齐因子（如 16 for VAE）
            data_file_keys: 数据字段名，如 ["video", "condition_video"]
            image_file_extension: 支持的图像格式
            video_file_extension: 支持的视频格式
            repeat: 数据重复次数（用于延长 epoch）
            args: 可选参数对象，覆盖以上参数
            base_path_map: dict，为每个 key 指定不同的 base path
        """
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



        # 分辨率模式
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
        else:
            raise ValueError("Either both height and width should be None, or both should be specified.")

        # 加载 metadata
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
        """中心裁剪并缩放到目标尺寸"""
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        new_size = (round(width * scale), round(height * scale))
        image = torchvision.transforms.functional.resize(
            image, new_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image

    def get_height_width(self, image):
        """计算目标分辨率"""
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
        """计算目标分辨率"""
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
        """获取对齐后的帧数"""
        total = int(reader.count_frames())
        num_frames = self.num_frames
        if total < num_frames:
            num_frames = total
        while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
            num_frames -= 1
        return num_frames

    def _load_gif(self, file_path):
        """加载 GIF 并统一帧率"""
        gif_img = Image.open(file_path)
        frames, delays = [], []
        frame_count = 0
        while True:
            delays.append(gif_img.info.get('duration', 100))  # ms
            rgb_frame = gif_img.convert("RGB")
            resized_frame = self.crop_and_resize(rgb_frame, *self.get_height_width(rgb_frame))
            frames.append(resized_frame)
            frame_count += 1
            try:
                gif_img.seek(frame_count)
            except EOFError:
                break

        # 统一采样帧率（基于最小 delay）
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

        # 裁剪到目标帧数
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
                resized = self.crop_and_resize(pil_image, *self.get_height_width_half(pil_image))
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
                    # 检查 video 和 condition_video 帧数一致
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
        """
        Args:
            base_path: 默认根路径
            metadata_path: metadata 文件路径 (csv/json)
            num_frames: 目标帧数
            time_division_factor: 时间维度对齐因子，要求 num_frames % factor == remainder
            time_division_remainder: 同上，余数
            max_pixels: 动态分辨率时最大像素数
            height, width: 固定分辨率（若指定，则关闭动态）
            height_division_factor, width_division_factor: 分辨率对齐因子（如 16 for VAE）
            data_file_keys: 数据字段名，如 ["video", "condition_video"]
            image_file_extension: 支持的图像格式
            video_file_extension: 支持的视频格式
            repeat: 数据重复次数（用于延长 epoch）
            args: 可选参数对象，覆盖以上参数
            base_path_map: dict，为每个 key 指定不同的 base path
        """
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



        # 分辨率模式
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
        else:
            raise ValueError("Either both height and width should be None, or both should be specified.")

        # 加载 metadata
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
        """中心裁剪并缩放到目标尺寸"""
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        new_size = (round(width * scale), round(height * scale))
        image = torchvision.transforms.functional.resize(
            image, new_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image

    def get_height_width(self, image):
        """计算目标分辨率"""
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
        """获取对齐后的帧数"""
        total = int(reader.count_frames())
        num_frames = self.num_frames
        if total < num_frames:
            num_frames = total
        while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
            num_frames -= 1
        return num_frames

    def _load_gif(self, file_path):
        """加载 GIF 并统一帧率"""
        gif_img = Image.open(file_path)
        frames, delays = [], []
        frame_count = 0
        while True:
            delays.append(gif_img.info.get('duration', 100))  # ms
            rgb_frame = gif_img.convert("RGB")
            resized_frame = self.crop_and_resize(rgb_frame, *self.get_height_width(rgb_frame))
            frames.append(resized_frame)
            frame_count += 1
            try:
                gif_img.seek(frame_count)
            except EOFError:
                break

        # 统一采样帧率（基于最小 delay）
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

        # 裁剪到目标帧数
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
                    # 检查 video 和 condition_video 帧数一致
                    v_frames = loaded_data.get("video")
                    c_frames = loaded_data.get("condition_video")
                    
                    if v_frames is None or c_frames is None:
                        valid = False
                    else:
                        # 获取 video 的帧数和空间尺寸
                        if isinstance(v_frames, torch.Tensor):
                            v_frame_count = v_frames.shape[-4] if len(v_frames.shape) >= 4 else 0
                            v_height = v_frames.shape[-2] if len(v_frames.shape) >= 2 else 0
                            v_width = v_frames.shape[-1] if len(v_frames.shape) >= 1 else 0
                        elif isinstance(v_frames, list) and len(v_frames) > 0:
                            v_frame_count = len(v_frames)
                            # 假设是 PIL Image 列表
                            if hasattr(v_frames[0], 'size'):
                                v_width, v_height = v_frames[0].size
                            else:
                                v_height, v_width = 0, 0
                        else:
                            v_frame_count, v_height, v_width = 0, 0, 0
                            
                        # 获取 condition_video 的帧数和空间尺寸
                        if isinstance(c_frames, torch.Tensor):
                            # batch_size=1 的 tensor，帧数在倒数第3个维度
                            c_frame_count = (c_frames.shape[-3]-1)*4+1 if len(c_frames.shape) >= 3 else 0
                            c_height = c_frames.shape[-2]*16 if len(c_frames.shape) >= 2 else 0
                            c_width = c_frames.shape[-1]*16 if len(c_frames.shape) >= 1 else 0
                        elif isinstance(c_frames, list) and len(c_frames) > 0:
                            # list of PIL Images
                            c_frame_count = len(c_frames)
                            # 假设是 PIL Image 列表
                            if hasattr(c_frames[0], 'size'):
                                c_width, c_height = c_frames[0].size
                            else:
                                c_height, c_width = 0, 0
                        else:
                            c_frame_count, c_height, c_width = 0, 0, 0
                            
                        # 检查帧数是否一致
                        if v_frame_count != c_frame_count:
                            warnings.warn(f"Frame count mismatch: video has {v_frame_count} frames, "
                                        f"condition_video has {c_frame_count} frames")
                            valid = False
                        # 检查空间尺寸是否一致
                        elif v_height != c_height or v_width != c_width:
                            warnings.warn(f"Spatial size mismatch: video is {v_width}x{v_height}, "
                                        f"condition_video is {c_width}x{c_height}")
                            valid = False

                    if valid:
                        # import pdb;pdb.set_trace()
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
            # import pdb;pdb.set_trace()
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
