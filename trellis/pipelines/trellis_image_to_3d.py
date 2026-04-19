from typing import *
from contextlib import contextmanager
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import rembg
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp


class TrellisImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisImageTo3DPipeline, TrellisImageTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])

        return new_pipeline
    
    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        # Always prefer local torch.hub cache to avoid flaky network/GitHub checks.
        hub_repo_dir = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main")
        if not os.path.isdir(hub_repo_dir):
            raise RuntimeError(
                "Local DINOv2 hub cache not found. Expected repo at: "
                f"{hub_repo_dir}. "
                "Please warm up cache once (with network) and rerun in offline mode."
            )
        dinov2_model = torch.hub.load(
            hub_repo_dir,
            name,
            source="local",
            pretrained=True,
            trust_repo=True,
        )
        self.models['image_cond_model'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            if getattr(self, 'rembg_session', None) is None:
                self.rembg_session = rembg.new_session('u2net')
            output = rembg.remove(input, session=self.rembg_session)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples
        
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

        return coords

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        if 'radiance_field' in formats:
            ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)

    def _normalize_view_z_rotations(
        self,
        view_azimuths: Sequence[float],
        reference_azimuth: float = 0.0,
    ) -> List[int]:
        """Map view azimuths to quarter turns around the latent z axis.

        ``reference_azimuth`` defines the canonical front anchor. For the
        benchmark data, the front input view is at -90 degrees, so callers
        should pass that explicit anchor instead of assuming 0 degrees.
        """
        quarter_turns: List[int] = []
        for azimuth in view_azimuths:
            relative_azimuth = float(azimuth) - float(reference_azimuth)
            quarter_steps = int(round(relative_azimuth / 90.0))
            if not np.isclose(relative_azimuth, quarter_steps * 90.0):
                raise ValueError(
                    "view_aligned_stochastic only supports 90-degree azimuths, "
                    f"got relative azimuth {relative_azimuth}"
                )
            quarter_turns.append(quarter_steps % 4)
        return quarter_turns

    def _rotate_dense_sample_z(
        self,
        sample: torch.Tensor,
        quarter_turns: int,
    ) -> torch.Tensor:
        k = int(quarter_turns) % 4
        if k == 0:
            return sample
        return torch.rot90(sample, k=k, dims=(2, 3)).contiguous()

    def _rotate_sparse_coords_z(
        self,
        coords: torch.Tensor,
        spatial_resolution: int,
        quarter_turns: int,
    ) -> torch.Tensor:
        k = int(quarter_turns) % 4
        if k == 0:
            return coords

        rotated = coords.clone()
        x = rotated[:, 1].long()
        y = rotated[:, 2].long()
        limit = int(spatial_resolution) - 1
        for _ in range(k):
            x, y = limit - y, x
        rotated[:, 1] = x.to(rotated.dtype)
        rotated[:, 2] = y.to(rotated.dtype)
        return rotated.contiguous()

    def _rotate_sparse_sample_z(
        self,
        sample: sp.SparseTensor,
        spatial_resolution: int,
        quarter_turns: int,
    ) -> sp.SparseTensor:
        k = int(quarter_turns) % 4
        if k == 0:
            return sample
        rotated_coords = self._rotate_sparse_coords_z(
            sample.coords,
            spatial_resolution=spatial_resolution,
            quarter_turns=k,
        )
        return sample.replace(sample.feats, rotated_coords)

    def _rotate_sample_z(
        self,
        sample: Union[torch.Tensor, sp.SparseTensor],
        model: nn.Module,
        quarter_turns: int,
    ) -> Union[torch.Tensor, sp.SparseTensor]:
        k = int(quarter_turns) % 4
        if k == 0:
            return sample
        if isinstance(sample, torch.Tensor):
            return self._rotate_dense_sample_z(sample, k)
        if isinstance(sample, sp.SparseTensor):
            spatial_resolution = int(getattr(model, "resolution", 0))
            if spatial_resolution <= 0:
                if sample.coords.numel() == 0:
                    raise ValueError("Cannot infer sparse rotation resolution from empty coords.")
                spatial_resolution = int(sample.coords[:, 1:].max().item()) + 1
            return self._rotate_sparse_sample_z(sample, spatial_resolution, k)
        raise TypeError(f"Unsupported sample type for view rotation: {type(sample)}")

    def _relative_view_azimuths(
        self,
        view_azimuths: Sequence[float],
        reference_azimuth: float = 0.0,
    ) -> List[float]:
        relative_azimuths: List[float] = []
        for azimuth in view_azimuths:
            relative = (float(azimuth) - float(reference_azimuth) + 180.0) % 360.0 - 180.0
            relative_azimuths.append(relative)
        return relative_azimuths

    def _build_dense_canonical_view_mask(
        self,
        sample: torch.Tensor,
        direction_x: float,
        direction_y: float,
        floor: float,
        sharpness: float,
    ) -> torch.Tensor:
        if sample.ndim != 5:
            raise ValueError(f"Expected dense sample with 5 dims, got {sample.shape}")

        resolution_x = int(sample.shape[2])
        resolution_y = int(sample.shape[3])
        mask_dtype = torch.float32
        x_coords = torch.linspace(
            -1.0 + 1.0 / resolution_x,
            1.0 - 1.0 / resolution_x,
            resolution_x,
            device=sample.device,
            dtype=mask_dtype,
        )
        y_coords = torch.linspace(
            -1.0 + 1.0 / resolution_y,
            1.0 - 1.0 / resolution_y,
            resolution_y,
            device=sample.device,
            dtype=mask_dtype,
        )
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij')
        score = grid_x * float(direction_x) + grid_y * float(direction_y)
        mask_xy = float(floor) + (1.0 - float(floor)) * torch.sigmoid(float(sharpness) * score)
        return mask_xy.to(device=sample.device, dtype=sample.dtype).view(1, 1, resolution_x, resolution_y, 1)

    def _build_sparse_canonical_view_weights(
        self,
        sample: sp.SparseTensor,
        spatial_resolution: int,
        direction_x: float,
        direction_y: float,
        floor: float,
        sharpness: float,
    ) -> torch.Tensor:
        coords = sample.coords
        coord_dtype = torch.float32
        resolution = float(spatial_resolution)
        centered_x = ((coords[:, 1].to(dtype=coord_dtype) + 0.5) / resolution) * 2.0 - 1.0
        centered_y = ((coords[:, 2].to(dtype=coord_dtype) + 0.5) / resolution) * 2.0 - 1.0
        score = centered_x * float(direction_x) + centered_y * float(direction_y)
        weights = float(floor) + (1.0 - float(floor)) * torch.sigmoid(float(sharpness) * score)
        return weights.to(device=sample.device, dtype=sample.feats.dtype)

    def _apply_canonical_view_weighting(
        self,
        pred: Union[torch.Tensor, sp.SparseTensor],
        model: nn.Module,
        sampler_name: str,
        relative_azimuth: float,
        view_weighting: Dict[str, float],
    ) -> Union[torch.Tensor, sp.SparseTensor]:
        floor = float(view_weighting['floor'])
        if sampler_name == 'sparse_structure_sampler':
            nonreference_scale = float(view_weighting['ss_nonreference_scale'])
            sharpness = float(view_weighting['ss_sharpness'])
        elif sampler_name == 'slat_sampler':
            nonreference_scale = float(view_weighting['slat_nonreference_scale'])
            sharpness = float(view_weighting['slat_sharpness'])
        else:
            raise ValueError(f"Unsupported sampler for canonical view weighting: {sampler_name}")

        redistribution_strength = (
            1.0
            if np.isclose(relative_azimuth % 360.0, 0.0)
            else float(np.clip(nonreference_scale, 0.0, 1.0))
        )
        theta = np.deg2rad(relative_azimuth)
        # Benchmark cardinal labels use front=-90, right=0, back=90, left=180.
        # With front as the canonical anchor, the visible half-space therefore
        # lies on -y for theta=0 and rotates around z from there.
        direction_x = float(np.sin(theta))
        direction_y = float(-np.cos(theta))

        if isinstance(pred, torch.Tensor):
            mask = self._build_dense_canonical_view_mask(
                pred,
                direction_x=direction_x,
                direction_y=direction_y,
                floor=floor,
                sharpness=sharpness,
            )
            normalized_mask = mask / mask.mean().clamp_min(1e-6)
            redistribution_weights = 1.0 + redistribution_strength * (normalized_mask - 1.0)
            return pred * redistribution_weights.to(dtype=pred.dtype)

        if isinstance(pred, sp.SparseTensor):
            spatial_resolution = int(getattr(model, "resolution", 0))
            if spatial_resolution <= 0:
                if pred.coords.numel() == 0:
                    raise ValueError("Cannot infer sparse weighting resolution from empty coords.")
                spatial_resolution = int(pred.coords[:, 1:].max().item()) + 1
            weights = self._build_sparse_canonical_view_weights(
                pred,
                spatial_resolution=spatial_resolution,
                direction_x=direction_x,
                direction_y=direction_y,
                floor=floor,
                sharpness=sharpness,
            )
            normalized_weights = weights / weights.mean().clamp_min(1e-6)
            redistribution_weights = 1.0 + redistribution_strength * (normalized_weights - 1.0)
            while redistribution_weights.ndim < pred.feats.ndim:
                redistribution_weights = redistribution_weights.unsqueeze(-1)
            return pred.replace(pred.feats * redistribution_weights)

        raise TypeError(f"Unsupported prediction type for canonical view weighting: {type(pred)}")

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal['stochastic', 'multidiffusion', 'view_aligned_stochastic', 'canonical_weighted_stochastic'] = 'stochastic',
        view_azimuths: Optional[Sequence[float]] = None,
        view_reference_azimuth: float = 0.0,
        canonical_view_weighting: Optional[Dict[str, float]] = None,
    ):
        """
        Inject a sampler with multiple images as condition.
        
        Args:
            sampler_name (str): The name of the sampler to inject.
            num_images (int): The number of images to condition on.
            num_steps (int): The number of steps to run the sampler for.
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, f'_old_inference_model', sampler._inference_model)
        pipeline = self

        if mode in ('stochastic', 'view_aligned_stochastic', 'canonical_weighted_stochastic'):
            if num_steps is None:
                raise ValueError(f"{mode} requires an explicit sampler step count.")
            if num_images > num_steps:
                print(f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m")

            cond_indices = (np.arange(num_steps) % num_images).tolist()
            quarter_turns = None
            relative_azimuths = None
            if mode == 'view_aligned_stochastic':
                if view_azimuths is None:
                    raise ValueError("view_aligned_stochastic requires view_azimuths.")
                if len(view_azimuths) != num_images:
                    raise ValueError(
                        "view_azimuths length must match num_images, got "
                        f"{len(view_azimuths)} vs {num_images}"
                    )
                quarter_turns = pipeline._normalize_view_z_rotations(
                    view_azimuths,
                    reference_azimuth=view_reference_azimuth,
                )
            elif mode == 'canonical_weighted_stochastic':
                if view_azimuths is None:
                    raise ValueError("canonical_weighted_stochastic requires view_azimuths.")
                if canonical_view_weighting is None:
                    raise ValueError(
                        "canonical_weighted_stochastic requires canonical_view_weighting."
                    )
                if len(view_azimuths) != num_images:
                    raise ValueError(
                        "view_azimuths length must match num_images, got "
                        f"{len(view_azimuths)} vs {num_images}"
                    )
                relative_azimuths = pipeline._relative_view_azimuths(
                    view_azimuths,
                    reference_azimuth=view_reference_azimuth,
                )

            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx:cond_idx+1]
                if mode == 'stochastic':
                    return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)
                if mode == 'canonical_weighted_stochastic':
                    assert relative_azimuths is not None
                    assert canonical_view_weighting is not None
                    pred = self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)
                    return pipeline._apply_canonical_view_weighting(
                        pred,
                        model=model,
                        sampler_name=sampler_name,
                        relative_azimuth=relative_azimuths[cond_idx],
                        view_weighting=canonical_view_weighting,
                    )

                assert quarter_turns is not None
                sample_rotation = quarter_turns[cond_idx]
                rotated_x_t = pipeline._rotate_sample_z(x_t, model, sample_rotation)
                rotated_pred = self._old_inference_model(model, rotated_x_t, t, cond=cond_i, **kwargs)
                return pipeline._rotate_sample_z(rotated_pred, model, (-sample_rotation) % 4)
        
        elif mode =='multidiffusion':
            from .samplers import FlowEulerSampler
            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    return pred
            
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        yield

        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, f'_old_inference_model')

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        mode: Literal['stochastic', 'multidiffusion', 'view_aligned_stochastic', 'canonical_weighted_stochastic'] = 'stochastic',
        view_azimuths: Optional[Sequence[float]] = None,
        view_reference_azimuth: float = 0.0,
        canonical_view_weighting: Optional[Dict[str, float]] = None,
    ) -> dict:
        """
        Run the pipeline with multiple images as condition

        Args:
            images (List[Image.Image]): The multi-view images of the assets
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
            view_azimuths (Optional[Sequence[float]]): Per-image azimuths used by
                ``view_aligned_stochastic`` to rotate the current latent around
                the z axis before denoising and then rotate the predicted update
                back to canonical coordinates.
            view_reference_azimuth (float): Canonical front azimuth used to turn
                absolute view azimuths into relative rotations.
        """
        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        cond = self.get_cond(images)
        cond['neg_cond'] = cond['neg_cond'][:1]
        torch.manual_seed(seed)
        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get('steps')
        with self.inject_sampler_multi_image(
            'sparse_structure_sampler',
            len(images),
            ss_steps,
            mode=mode,
            view_azimuths=view_azimuths,
            view_reference_azimuth=view_reference_azimuth,
            canonical_view_weighting=canonical_view_weighting,
        ):
            coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')
        with self.inject_sampler_multi_image(
            'slat_sampler',
            len(images),
            slat_steps,
            mode=mode,
            view_azimuths=view_azimuths,
            view_reference_azimuth=view_reference_azimuth,
            canonical_view_weighting=canonical_view_weighting,
        ):
            slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
