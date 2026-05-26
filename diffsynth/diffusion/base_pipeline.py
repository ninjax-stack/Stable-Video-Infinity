from PIL import Image
import torch
import numpy as np
from einops import repeat, reduce
from typing import Union
from ..core import AutoTorchModule, AutoWrappedLinear, load_state_dict, ModelConfig
from ..utils.lora import GeneralLoRALoader
from ..models.model_loader import ModelPool
from ..utils.controlnet import ControlNetInput


class PipelineUnit:
    def __init__(
        self,
        seperate_cfg: bool = False,
        take_over: bool = False,
        input_params: tuple[str] = None,
        output_params: tuple[str] = None,
        input_params_posi: dict[str, str] = None,
        input_params_nega: dict[str, str] = None,
        onload_model_names: tuple[str] = None
    ):
        self.seperate_cfg = seperate_cfg
        self.take_over = take_over
        self.input_params = input_params
        self.output_params = output_params
        self.input_params_posi = input_params_posi
        self.input_params_nega = input_params_nega
        self.onload_model_names = onload_model_names

    def fetch_input_params(self):
        params = []
        if self.input_params is not None:
            for param in self.input_params:
                params.append(param)
        if self.input_params_posi is not None:
            for _, param in self.input_params_posi.items():
                params.append(param)
        if self.input_params_nega is not None:
            for _, param in self.input_params_nega.items():
                params.append(param)
        params = sorted(list(set(params)))
        return params
    
    def fetch_output_params(self):
        params = []
        if self.output_params is not None:
            for param in self.output_params:
                params.append(param)
        return params

    def process(self, pipe, **kwargs) -> dict:
        return {}
    
    def post_process(self, pipe, **kwargs) -> dict:
        return {}


class BasePipeline(torch.nn.Module):

    def __init__(
        self,
        device="cuda", torch_dtype=torch.float16,
        height_division_factor=64, width_division_factor=64,
        time_division_factor=None, time_division_remainder=None,
    ):
        super().__init__()
        # The device and torch_dtype is used for the storage of intermediate variables, not models.
        self.device = device
        self.torch_dtype = torch_dtype
        # The following parameters are used for shape check.
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # VRAM management
        self.vram_management_enabled = False
        # Pipeline Unit Runner
        self.unit_runner = PipelineUnitRunner()
        # LoRA Loader
        self.lora_loader = GeneralLoRALoader
        
        
    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        super().to(*args, **kwargs)
        return self


    def check_resize_height_width(self, height, width, num_frames=None):
        # Shape check
        if height % self.height_division_factor != 0:
            height = (height + self.height_division_factor - 1) // self.height_division_factor * self.height_division_factor
            print(f"height % {self.height_division_factor} != 0. We round it up to {height}.")
        if width % self.width_division_factor != 0:
            width = (width + self.width_division_factor - 1) // self.width_division_factor * self.width_division_factor
            print(f"width % {self.width_division_factor} != 0. We round it up to {width}.")
        if num_frames is None:
            return height, width
        else:
            if num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames = (num_frames + self.time_division_factor - 1) // self.time_division_factor * self.time_division_factor + self.time_division_remainder
                print(f"num_frames % {self.time_division_factor} != {self.time_division_remainder}. We round it up to {num_frames}.")
            return height, width, num_frames


    def preprocess_image(self, image, torch_dtype=None, device=None, pattern="B C H W", min_value=-1, max_value=1):
        # Transform a PIL.Image to torch.Tensor
        image = torch.Tensor(np.array(image, dtype=np.float32))
        image = image.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        image = image * ((max_value - min_value) / 255) + min_value
        image = repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))
        return image


    def preprocess_video(self, video, torch_dtype=None, device=None, pattern="B C T H W", min_value=-1, max_value=1):
        # Transform a list of PIL.Image to torch.Tensor
        video = [self.preprocess_image(image, torch_dtype=torch_dtype, device=device, min_value=min_value, max_value=max_value) for image in video]
        video = torch.stack(video, dim=pattern.index("T") // 2)
        return video


    def vae_output_to_image(self, vae_output, pattern="B C H W", min_value=-1, max_value=1):
        # Transform a torch.Tensor to PIL.Image
        if pattern != "H W C":
            vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean")
        image = ((vae_output - min_value) * (255 / (max_value - min_value))).clip(0, 255)
        image = image.to(device="cpu", dtype=torch.uint8)
        image = Image.fromarray(image.numpy())
        return image


    def vae_output_to_video(self, vae_output, pattern="B C T H W", min_value=-1, max_value=1):
        # Transform a torch.Tensor to list of PIL.Image
        if pattern != "T H W C":
            vae_output = reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")
        video = [self.vae_output_to_image(image, pattern="H W C", min_value=min_value, max_value=max_value) for image in vae_output]
        return video


    def load_models_to_device(self, model_names):
        if self.vram_management_enabled:
            # offload models
            for name, model in self.named_children():
                if name not in model_names:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        if hasattr(model, "offload"):
                            model.offload()
                        else:
                            for module in model.modules():
                                if hasattr(module, "offload"):
                                    module.offload()
            torch.cuda.empty_cache()
            # onload models
            for name, model in self.named_children():
                if name in model_names:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        if hasattr(model, "onload"):
                            model.onload()
                        else:
                            for module in model.modules():
                                if hasattr(module, "onload"):
                                    module.onload()


    def generate_noise(self, shape, seed=None, rand_device="cpu", rand_torch_dtype=torch.float32, device=None, torch_dtype=None):
        # Initialize Gaussian noise
        generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=rand_device, dtype=rand_torch_dtype)
        noise = noise.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        return noise

        
    def get_vram(self):
        return torch.cuda.mem_get_info(self.device)[1] / (1024 ** 3)
    
    def get_module(self, model, name):
        if "." in name:
            name, suffix = name[:name.index(".")], name[name.index(".") + 1:]
            if name.isdigit():
                return self.get_module(model[int(name)], suffix)
            else:
                return self.get_module(getattr(model, name), suffix)
        else:
            return getattr(model, name)
    
    def freeze_except(self, model_names):
        self.eval()
        self.requires_grad_(False)
        for name in model_names:
            module = self.get_module(self, name)
            if module is None:
                print(f"No {name} models in the pipeline. We cannot enable training on the model. If this occurs during the data processing stage, it is normal.")
                continue
            module.train()
            module.requires_grad_(True)
                
    
    def blend_with_mask(self, base, addition, mask):
        return base * (1 - mask) + addition * mask
    
    
    def step(self, scheduler, latents, progress_id, noise_pred, input_latents=None, inpaint_mask=None, **kwargs):
        timestep = scheduler.timesteps[progress_id]
        if inpaint_mask is not None:
            noise_pred_expected = scheduler.return_to_timestep(scheduler.timesteps[progress_id], latents, input_latents)
            noise_pred = self.blend_with_mask(noise_pred_expected, noise_pred, inpaint_mask)
        latents_next = scheduler.step(noise_pred, timestep, latents)
        return latents_next
    
    
    def split_pipeline_units(self, model_names: list[str]):
        return PipelineUnitGraph().split_pipeline_units(self.units, model_names)
    
    
    def flush_vram_management_device(self, device):
        for module in self.modules():
            if isinstance(module, AutoTorchModule):
                module.offload_device = device
                module.onload_device = device
                module.preparing_device = device
                module.computation_device = device
                
    
    def load_lora(
        self,
        module: torch.nn.Module,
        lora_config: Union[ModelConfig, str] = None,
        alpha=1,
        hotload=None,
        state_dict=None,
    ):
        if state_dict is None:
            if isinstance(lora_config, str):
                lora = load_state_dict(lora_config, torch_dtype=self.torch_dtype, device=self.device)
            else:
                lora_config.download_if_necessary()
                lora = load_state_dict(lora_config.path, torch_dtype=self.torch_dtype, device=self.device)
        else:
            lora = state_dict
        lora_loader = self.lora_loader(torch_dtype=self.torch_dtype, device=self.device)
        lora = lora_loader.convert_state_dict(lora)
        if hotload is None:
            hotload = hasattr(module, "vram_management_enabled") and getattr(module, "vram_management_enabled")
        if hotload:
            if not (hasattr(module, "vram_management_enabled") and getattr(module, "vram_management_enabled")):
                raise ValueError("VRAM Management is not enabled. LoRA hotloading is not supported.")
            updated_num = 0
            for _, module in module.named_modules():
                if isinstance(module, AutoWrappedLinear):
                    name = module.name
                    lora_a_name = f'{name}.lora_A.weight'
                    lora_b_name = f'{name}.lora_B.weight'
                    if lora_a_name in lora and lora_b_name in lora:
                        updated_num += 1
                        module.lora_A_weights.append(lora[lora_a_name] * alpha)
                        module.lora_B_weights.append(lora[lora_b_name])
            print(f"{updated_num} tensors are patched by LoRA. You can use `pipe.clear_lora()` to clear all LoRA layers.")
        else:
            lora_loader.fuse_lora_to_base_model(module, lora, alpha=alpha)
            
            
    def clear_lora(self):
        cleared_num = 0
        for name, module in self.named_modules():
            if isinstance(module, AutoWrappedLinear):
                if hasattr(module, "lora_A_weights"):
                    if len(module.lora_A_weights) > 0:
                        cleared_num += 1
                    module.lora_A_weights.clear()
                if hasattr(module, "lora_B_weights"):
                    module.lora_B_weights.clear()
        print(f"{cleared_num} LoRA layers are cleared.")
        
    
    def download_and_load_models(self, model_configs: list[ModelConfig] = [], vram_limit: float = None):
        #changed by daniel to load custom model
        from ..models.model_loader import ModelPool
        import os

        # Define the custom converter to handle various prefixes
        def universal_wan_converter(state_dict):
            new_state_dict = {}
            for k in state_dict:
                v = state_dict[k]
                new_key = k
                if k.startswith("model.diffusion_model."):
                    new_key = k[len("model.diffusion_model."):]
                elif k.startswith("model."):
                    new_key = k[len("model."):]
                elif k.startswith("diffusion_pytorch_model."):
                    new_key = k[len("diffusion_pytorch_model."):]
                
                # T5 mapping for HF models
                if "encoder.block." in new_key:
                    new_key = new_key.replace("encoder.block.", "blocks.")
                    new_key = new_key.replace(".layer.0.SelfAttention.", ".attn.")
                    new_key = new_key.replace(".layer.0.layer_norm.", ".norm1.")
                    new_key = new_key.replace(".layer.1.DenseReluDense.wi_0.", ".ffn.gate.0.")
                    new_key = new_key.replace(".layer.1.DenseReluDense.wi_1.", ".ffn.fc1.")
                    new_key = new_key.replace(".layer.1.DenseReluDense.wo.", ".ffn.fc2.")
                    new_key = new_key.replace(".layer.1.layer_norm.", ".norm2.")
                    new_key = new_key.replace(".attn.relative_attention_bias.", ".pos_embedding.embedding.")
                elif new_key == "encoder.final_layer_norm.weight":
                    new_key = "norm.weight"
                elif new_key in ["shared.weight", "encoder.embed_tokens.weight"]:
                    new_key = "token_embedding.weight"
                
                new_state_dict[new_key] = v
            return new_state_dict

        # Monkey patch ModelPool.import_model_class to support "UniversalWanConverter"
        original_import = ModelPool.import_model_class
        def patched_import_model_class(self_pool, model_class_path):
            if model_class_path == "UniversalWanConverter":
                return universal_wan_converter
            return original_import(self_pool, model_class_path)
        ModelPool.import_model_class = patched_import_model_class

        # Monkey patch ModelPool.auto_load_model to handle unknown hashes for T5/DiT
        original_auto_load = ModelPool.auto_load_model
        def patched_auto_load_model(self_pool, path, vram_config=None, vram_limit=None, clear_parameters=False):
            try:
                return original_auto_load(self_pool, path, vram_config, vram_limit, clear_parameters)
            except ValueError as e:
                filename = path[0] if isinstance(path, list) else path
                filename = os.path.basename(filename).lower()
                
                # Try to guess model type from filename
                model_name = None
                if "umt5" in filename or "t5" in filename:
                    model_name = "wan_video_text_encoder"
                elif "vae" in filename:
                    model_name = "wan_video_vae"
                elif "clip" in filename:
                    model_name = "wan_video_image_encoder"
                elif (
                    "custom-wan2" in filename
                    or "diffusion_pytorch_model" in filename
                    or filename.endswith(".safetensors")
                ):
                    # Catch-all: any unrecognized safetensors (umt5/vae/clip
                    # already routed above) is most likely a Wan DiT variant —
                    # custom checkpoints, A14B fine-tunes, repacks. Uses the
                    # most recent wan_video_dit template (A14B geometry); non-
                    # A14B-shaped files fail at load_state_dict, which is correct.
                    model_name = "wan_video_dit"
                
                if model_name:
                    print(f"Hash check failed for {filename}, but guessing model type: {model_name}. Forcing load.")
                    # Find a matching config as template
                    template_config = None
                    from ..configs import MODEL_CONFIGS
                    for config in reversed(MODEL_CONFIGS):
                        if config.get("model_name") == model_name:
                            template_config = config
                            break
                    if template_config:
                        # Use load_model_file with the template config but our path
                        model = self_pool.load_model_file(template_config, path, vram_config or self_pool.default_vram_config(), vram_limit=vram_limit)
                        if clear_parameters: self_pool.clear_parameters(model)
                        self_pool.model.append(model)
                        self_pool.model_name.append(model_name)
                        self_pool.model_path.append(path)
                        return
                raise e
        ModelPool.auto_load_model = patched_auto_load_model

        # Configuration for the custom Wan 2.2 I2V models
        # (Based on inspect: 40 layers, dim 5120, ffn_dim 13824, in_dim 36),add here for new models
        wan_series = [
            {
                "model_hash": "ce0d6f9edef505c078665c29046eb82e",
                "model_name": "wan_video_dit",
                "model_class": "diffsynth.models.wan_video_dit.WanModel",
                "extra_kwargs": {'has_image_input': False, 'patch_size': [1, 2, 2], 'in_dim': 36, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06, 'require_clip_embedding': False,},
                "state_dict_converter": "UniversalWanConverter"
            },
            {
                "model_hash": "5b013604280dd715f8457c6ed6d6a626",
                "model_name": "wan_video_dit",
                "model_class": "diffsynth.models.wan_video_dit.WanModel",
                "extra_kwargs": {'has_image_input': False, 'patch_size': [1, 2, 2], 'in_dim': 36, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06, 'require_clip_embedding': False,},
                "state_dict_converter": "UniversalWanConverter"
            },
             {
                "model_hash": "1663aabb1c900fdd416fe4bf56cd0a1b",
                "model_name": "wan_video_text_encoder",
                "model_class": "diffsynth.models.wan_video_text_encoder.WanTextEncoder",
                "state_dict_converter": "UniversalWanConverter"
            }
            
        ]
        from ..configs import MODEL_CONFIGS
        # Update existing configs to use our universal converter if they match Wan models
        for config in MODEL_CONFIGS:
            if config.get("model_name") in ["wan_video_dit", "wan_video_vae", "wan_video_text_encoder", "wan_video_image_encoder"]:
                if "state_dict_converter" not in config:
                    config["state_dict_converter"] = "UniversalWanConverter"
        
        MODEL_CONFIGS += wan_series
        model_pool = ModelPool()
        for model_config in model_configs:
            # model_config.download_if_necessary()
            import glob,os
            model_config.reset_local_model_path()
            model_config.path = glob.glob(os.path.join(model_config.local_model_path, model_config.model_id, model_config.origin_file_pattern))
            vram_config = model_config.vram_config()
            vram_config["computation_dtype"] = vram_config["computation_dtype"] or self.torch_dtype
            vram_config["computation_device"] = vram_config["computation_device"] or self.device
            model_pool.auto_load_model(
                model_config.path,
                vram_config=vram_config,
                vram_limit=vram_limit,
                clear_parameters=model_config.clear_parameters,
            )
        #end change
        return model_pool
    
    
    def check_vram_management_state(self):
        vram_management_enabled = False
        for module in self.children():
            if hasattr(module, "vram_management_enabled") and getattr(module, "vram_management_enabled"):
                vram_management_enabled = True
        return vram_management_enabled
    
    
    def cfg_guided_model_fn(self, model_fn, cfg_scale, inputs_shared, inputs_posi, inputs_nega, **inputs_others):
        noise_pred_posi = model_fn(**inputs_posi, **inputs_shared, **inputs_others)
        if cfg_scale != 1.0:
            noise_pred_nega = model_fn(**inputs_nega, **inputs_shared, **inputs_others)
            noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
        else:
            noise_pred = noise_pred_posi
        return noise_pred


class PipelineUnitGraph:
    def __init__(self):
        pass
    
    def build_edges(self, units: list[PipelineUnit]):
        # Establish dependencies between units
        # to search for subsequent related computation units.
        last_compute_unit_id = {}
        edges = []
        for unit_id, unit in enumerate(units):
            for input_param in unit.fetch_input_params():
                if input_param in last_compute_unit_id:
                    edges.append((last_compute_unit_id[input_param], unit_id))
            for output_param in unit.fetch_output_params():
                last_compute_unit_id[output_param] = unit_id
        return edges
    
    def build_chains(self, units: list[PipelineUnit]):
        # Establish updating chains for each variable
        # to track their computation process.
        params = sum([unit.fetch_input_params() + unit.fetch_output_params() for unit in units], [])
        params = sorted(list(set(params)))
        chains = {param: [] for param in params}
        for unit_id, unit in enumerate(units):
            for param in unit.fetch_output_params():
                chains[param].append(unit_id)
        return chains
    
    def search_direct_unit_ids(self, units: list[PipelineUnit], model_names: list[str]):
        # Search for units that directly participate in the model's computation.
        related_unit_ids = []
        for unit_id, unit in enumerate(units):
            for model_name in model_names:
                if unit.onload_model_names is not None and model_name in unit.onload_model_names:
                    related_unit_ids.append(unit_id)
                    break
        return related_unit_ids
    
    def search_related_unit_ids(self, edges, start_unit_ids, direction="target"):
        # Search for subsequent related computation units.
        related_unit_ids = [unit_id for unit_id in start_unit_ids]
        while True:
            neighbors = []
            for source, target in edges:
                if direction == "target" and source in related_unit_ids and target not in related_unit_ids:
                    neighbors.append(target)
                elif direction == "source" and source not in related_unit_ids and target in related_unit_ids:
                    neighbors.append(source)
            neighbors = sorted(list(set(neighbors)))
            if len(neighbors) == 0:
                break
            else:
                related_unit_ids.extend(neighbors)
        related_unit_ids = sorted(list(set(related_unit_ids)))
        return related_unit_ids
    
    def search_updating_unit_ids(self, units: list[PipelineUnit], chains, related_unit_ids):
        # If the input parameters of this subgraph are updated outside the subgraph,
        # search for the units where these updates occur.
        first_compute_unit_id = {}
        for unit_id in related_unit_ids:
            for param in units[unit_id].fetch_input_params():
                if param not in first_compute_unit_id:
                    first_compute_unit_id[param] = unit_id
        updating_unit_ids = []
        for param in first_compute_unit_id:
            unit_id = first_compute_unit_id[param]
            chain = chains[param]
            if unit_id in chain and chain.index(unit_id) != len(chain) - 1:
                for unit_id_ in chain[chain.index(unit_id) + 1:]:
                    if unit_id_ not in related_unit_ids:
                        updating_unit_ids.append(unit_id_)
        related_unit_ids.extend(updating_unit_ids)
        related_unit_ids = sorted(list(set(related_unit_ids)))
        return related_unit_ids
    
    def split_pipeline_units(self, units: list[PipelineUnit], model_names: list[str]):
        # Split the computation graph,
        # separating all model-related computations.
        related_unit_ids = self.search_direct_unit_ids(units, model_names)
        edges = self.build_edges(units)
        chains = self.build_chains(units)
        while True:
            num_related_unit_ids = len(related_unit_ids)
            related_unit_ids = self.search_related_unit_ids(edges, related_unit_ids, "target")
            related_unit_ids = self.search_updating_unit_ids(units, chains, related_unit_ids)
            if len(related_unit_ids) == num_related_unit_ids:
                break
            else:
                num_related_unit_ids = len(related_unit_ids)
        related_units = [units[i] for i in related_unit_ids]
        unrelated_units = [units[i] for i in range(len(units)) if i not in related_unit_ids]
        return related_units, unrelated_units


class PipelineUnitRunner:
    def __init__(self):
        pass

    def __call__(self, unit: PipelineUnit, pipe: BasePipeline, inputs_shared: dict, inputs_posi: dict, inputs_nega: dict) -> tuple[dict, dict]:
        if unit.take_over:
            # Let the pipeline unit take over this function.
            inputs_shared, inputs_posi, inputs_nega = unit.process(pipe, inputs_shared=inputs_shared, inputs_posi=inputs_posi, inputs_nega=inputs_nega)
        elif unit.seperate_cfg:
            # Positive side
            processor_inputs = {name: inputs_posi.get(name_) for name, name_ in unit.input_params_posi.items()}
            if unit.input_params is not None:
                for name in unit.input_params:
                    processor_inputs[name] = inputs_shared.get(name)
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_posi.update(processor_outputs)
            # Negative side
            if inputs_shared["cfg_scale"] != 1:
                processor_inputs = {name: inputs_nega.get(name_) for name, name_ in unit.input_params_nega.items()}
                if unit.input_params is not None:
                    for name in unit.input_params:
                        processor_inputs[name] = inputs_shared.get(name)
                processor_outputs = unit.process(pipe, **processor_inputs)
                inputs_nega.update(processor_outputs)
            else:
                inputs_nega.update(processor_outputs)
        else:
            processor_inputs = {name: inputs_shared.get(name) for name in unit.input_params}
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_shared.update(processor_outputs)
        return inputs_shared, inputs_posi, inputs_nega
