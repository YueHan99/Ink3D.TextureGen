"""
Training with interleaved H+V conditions:
  H position.mp4   [0::2] → 60 frames
+ V position_flip.mp4 [0::2] → 60 frames
+ H position.mp4   [0]    →  1 frame  = 121 frames
Same for normal.mp4 and albedo.mp4 from H + V dirs.
"""
import torch, os, random, warnings, json
import numpy as np
from PIL import Image
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new_14b_ref_drop_normal_revise import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from diffsynth.data.video import VideoData
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class HVIterleaveDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        import pandas as pd
        self.base_path = args.dataset_base_path
        self.height = args.height
        self.width = args.width
        self.num_frames = 121
        self.half_frames = 60

        df = pd.read_csv(os.path.join(self.base_path, args.dataset_metadata_path))
        self.data = df.to_dict(orient="records")
        print(f"Loaded {len(self.data)} paired H+V samples")

    def load_video_frames(self, path, indices):
        """Load specific frame indices from a video file."""
        vd = VideoData(path, height=self.height, width=self.width)
        n = min(len(vd), max(indices) + 1)
        return [vd[i] for i in indices if i < n]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx % len(self.data)]

        # H paths
        h_video_path = os.path.join(self.base_path, row["h_video"])
        h_control_path = os.path.join(self.base_path, row["h_control_video"])
        # V paths
        v_video_path = os.path.join(self.base_path, row["v_video"])
        v_control_path = os.path.join(self.base_path, row["v_control_video"])

        # H dir: strip albedo.mp4 to get sample dir
        h_dir = os.path.dirname(h_video_path)
        v_dir = os.path.dirname(v_video_path)

        # Stride-2 indices: 120 frames → 60 frames (0, 2, 4, ..., 118)
        h_indices = list(range(0, 119, 2))   # 60 frames
        v_indices = list(range(0, 119, 2))   # 60 frames

        # Load control video (position) from H and V
        h_control = self.load_video_frames(h_control_path, h_indices)
        v_control = self.load_video_frames(v_control_path, v_indices)

        # Concat: H[0::2] + V[0::2] + H[0]
        control_video = h_control + v_control + [h_control[0]]

        # Load albedo video from H and V
        h_video = self.load_video_frames(h_video_path, h_indices)
        v_video_frames = self.load_video_frames(v_video_path, v_indices)
        video = h_video + v_video_frames + [h_video[0]]

        # Load normal from H and V dirs
        h_normal = self.load_video_frames(os.path.join(h_dir, "normal.mp4"), h_indices)
        v_normal = self.load_video_frames(os.path.join(v_dir, "normal.mp4"), v_indices)
        normal = h_normal + v_normal + [h_normal[0]]

        if len(control_video) != 121 or len(video) != 121 or len(normal) != 121:
            print(f"WARNING: frame count mismatch for idx={idx} ctrl={len(control_video)} video={len(video)} normal={len(normal)}, skipping")
            return self.__getitem__((idx + 1) % len(self.data))

        # Reference image (same logic as original: use images_curve dir)
        img_dir = h_dir.replace('videos_curve/', 'images_curve/')
        ref_candidates = []
        for i in range(1, 121):
            p = os.path.join(img_dir, f'render_{i:04d}.png')
            if os.path.exists(p):
                ref_candidates.append(i)
            else:
                break
        if ref_candidates:
            chosen = random.choice(ref_candidates)
            ref_path = os.path.join(img_dir, f'render_{chosen:04d}.png')
            ref_img = Image.open(ref_path).convert("RGB")
        else:
            ref_img = video[0].convert("RGB")

        return {
            "video": video,
            "control_video": control_video,
            "normal": normal,
            "random_ref_image": ref_img,
            "prompt": row.get("prompt", "This is a 3D model"),
        }


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self, model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None, lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        lora_rank=32, lora_checkpoint=None, use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False, extra_inputs=None,
        max_timestep_boundary=1.0, min_timestep_boundary=0.0,
    ):
        super().__init__()
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]) for i in model_id_with_origin_paths]
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))

        if lora_base_model is not None or lora_checkpoint is not None:
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                patch_embedding_state_dict1 = {}
                if 'patch_embedding.weight' in state_dict:
                    patch_embedding_state_dict1['bias'] = state_dict['patch_embedding.bias']
                    patch_embedding_state_dict1['weight'] = state_dict['patch_embedding.weight']
                state_dict = self.mapping_lora_state_dict(state_dict)
            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model),
                target_modules=lora_target_modules.split(","), lora_rank=lora_rank
            )
            if lora_checkpoint is not None:
                load_result = model.load_state_dict(state_dict, strict=False)
                print(f"LoRA loaded: {lora_checkpoint}, {len(state_dict)} keys")
                if len(load_result[1]) > 0:
                    print(f"Warning, LoRA key mismatch: {load_result[1]}")
                setattr(self.pipe, lora_base_model, model)
                if patch_embedding_state_dict1:
                    self.pipe.dit.patch_embedding.load_state_dict(patch_embedding_state_dict1)
            else:
                patch_embedding_state_dict1 = None

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.pipe.dit.patch_embedding.train()
        self.pipe.dit.patch_embedding.requires_grad_(True)

    def forward_preprocess(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": 1, "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False, "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        for extra_input in self.extra_inputs:
            if extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data['random_ref_image']
                inputs_shared['normal'] = data['normal']
            else:
                inputs_shared[extra_input] = data[extra_input]
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    dataset = HVIterleaveDataset(args=args)
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    model_logger = ModelLogger(
        output_path=args.output_path,
        remove_prefix_in_ckpt=getattr(args, 'remove_prefix_in_ckpt', None)
    )
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate, weight_decay=getattr(args, 'weight_decay', 0))
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    launch_training_task(
        dataset, model, model_logger, optimizer, scheduler,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 1),
        save_steps=args.save_steps,
        find_unused_parameters=getattr(args, 'find_unused_parameters', False),
        num_workers=getattr(args, 'dataset_num_workers', 8),
    )
