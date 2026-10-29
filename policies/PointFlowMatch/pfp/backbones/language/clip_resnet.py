'ResNet + CLIP Text Encoder + FiLM\nA multimodal encoder combining multi-camera image features and language features, using FiLM for language+state modulation'

import torch
import torch.nn as nn
from diffusion_policy.model.vision.model_getter import get_resnet
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from pfp.backbones.language.clip import load_clip, tokenize


class FiLM(nn.Module):
    'Feature-wise Linear Modulation: Modulating image features with language+robot state features'
    def __init__(self, img_feat_dim: int, cond_feat_dim: int):
        'Args:\n    img_feat_dim: Image feature dimension\n    cond_feat_dim: Condition feature dimension (lang_feat_dim + robot_state_dim)'
        super().__init__()
        self.img_feat_dim = img_feat_dim
        self.cond_feat_dim = cond_feat_dim
        
        # MLP Predicts gamma and beta (from language+state joint features)
        self.mlp = nn.Sequential(
            nn.Linear(cond_feat_dim, cond_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cond_feat_dim, img_feat_dim * 2)
        )

    def forward(self, img_feats: torch.Tensor, cond_feats: torch.Tensor) -> torch.Tensor:
        'Args:\n    img_feats: [B, T, img_feat_dim] Image features\n    cond_feats: [B, T, cond_feat_dim] Condition features (language + robot state)\nReturns:\n    modulated: [B, T, img_feat_dim] Modulated image features'
        B, T, D = img_feats.shape
        assert D == self.img_feat_dim
        
        # Predicts gamma and beta
        gb = self.mlp(cond_feats)  # [B, T, 2*D]
        gb = gb.view(B, T, 2, D)   # [B, T, 2, D]
        gamma = gb[:, :, 0]        # [B, T, D]
        beta = gb[:, :, 1]         # [B, T, D]
        
        # FiLM: (1 + gamma) * x + beta
        out = img_feats * (1.0 + gamma) + beta
        return out


class ClipResNet(nn.Module):
    'Multimodal Encoder: ResNet (multi-camera images) + CLIP text encoder (language) + FiLM'
    
    def __init__(
        self,
        shape_meta: dict,
        clip_model_name: str = "RN50",
        device: str = "cuda",
        freeze_clip: bool = True,
        crop_shape: tuple = (76, 76),
        random_crop: bool = True,
        use_group_norm: bool = True,
        share_rgb_model: bool = False,
        imagenet_norm: bool = True,
    ):
        super().__init__()
        
        self.device = device
        self.clip_model_name = clip_model_name
        self.freeze_clip = freeze_clip
        
        # 1. Image Encoder (ResNet + MultiImageObsEncoder, returns separate features)
        rgb_model = get_resnet(name="resnet18")
        self.img_encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model=rgb_model,
            crop_shape=crop_shape,
            random_crop=random_crop,
            use_group_norm=use_group_norm,
            share_rgb_model=share_rgb_model,
            imagenet_norm=imagenet_norm,
            return_separate_features=True,  # Separately return RGB and low_dim features
        )
        
        # Get dimension information
        num_cameras = len([k for k in shape_meta["obs"].keys() if k.startswith("img_")])
        self.robot_state_dim = shape_meta["obs"].get("robot_state", {}).get("shape", [0])[0]
        
        # Dynamic dimension
        self.rgb_feat_dim = None   # RGB feature dimension per time step
        self.film = None           # FiLM module (delayed initialization)
        
        # 2. Load CLIP text encoder
        self.clip_model, _ = load_clip(clip_model_name, device=device, jit=False)
        self.lang_feat_dim = self.clip_model.text_projection.shape[1]
        
        # Freeze CLIP
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
            print("✅ CLIP text encoder frozen")
        else:
            print("⚠️  CLIP text encoder trainable")
        
        print(f"✅ ResNetCLIPEncoder (with FiLM: lang+state → img) initialized")
        print(f"   - Number of cameras: {num_cameras}")
        print(f"   - Language feature dimension: {self.lang_feat_dim}")
        print(f"   - Robot state dimension: {self.robot_state_dim}")
        print(f"   - Image feature dimension will be determined on the first forward pass")
    
    def encode_images(
        self, 
        images: torch.Tensor, 
        robot_state_obs: torch.Tensor
    ) -> dict:
        "Encode multi-camera images, separate and return RGB features and robot_state\n\nArgs:\n- images: [B, T, num_cameras, H, W, C]\n- robot_state_obs: [B, T, state_dim]\n\nReturns:\n- dict with:\n  - 'rgb': [B, T, rgb_feat_dim]\n  - 'robot_state': [B, T, state_dim]\n"
        B, T = images.shape[0], images.shape[1]
        
        # Flatten batch and time
        images_flat = images.reshape(-1, *images.shape[2:])  # [B*T, num_cameras, H, W, C]
        images_flat = images_flat.permute(0, 1, 4, 2, 3)     # [B*T, num_cameras, C, H, W]
        
        robot_state_flat = robot_state_obs.float().reshape(-1, *robot_state_obs.shape[2:])  # [B*T, state_dim]
        
        # Create obs_dict
        obs_dict = {f"img_{i}": images_flat[:, i] for i in range(images_flat.shape[1])}
        obs_dict["robot_state"] = robot_state_flat
        
        # ResNet encoding (separate features)
        features = self.img_encoder(obs_dict)  # {'rgb': [B*T, rgb_feat_dim], 'low_dim': [B*T, state_dim]}
        
        rgb_feat = features['rgb']        # [B*T, rgb_feat_dim]
        low_dim_feat = features['low_dim']  # [B*T, state_dim]
        
        # Reshape: [B*T, D] -> [B, T, D]
        rgb_feat = rgb_feat.reshape(B, T, -1)
        low_dim_feat = low_dim_feat.reshape(B, T, -1)
        
        # Initialize dimensions (first call)
        if self.rgb_feat_dim is None:
            self.rgb_feat_dim = rgb_feat.shape[-1]
            # Condition feature dimension = language feature + robot state
            cond_feat_dim = self.lang_feat_dim + self.robot_state_dim
            # Initialize FiLM module
            self.film = FiLM(
                img_feat_dim=self.rgb_feat_dim,
                cond_feat_dim=cond_feat_dim
            ).to(self.device)
            print(f"   - RGB feature dimension: {self.rgb_feat_dim}")
            print(f"   - Conditioning dimension (language and state): {cond_feat_dim}")
        
        return {
            'rgb': rgb_feat,
            'robot_state': low_dim_feat
        }
    
    def encode_text(self, task_descriptions) -> torch.Tensor:
        'Encode task description'
        if isinstance(task_descriptions, torch.Tensor):
            text_tokens = task_descriptions.to(self.device)
        else:
            text_tokens = tokenize(task_descriptions).to(self.device)
        
        if self.freeze_clip:
            with torch.no_grad():
                lang_feat = self.clip_model.encode_text(text_tokens).float()
        else:
            lang_feat = self.clip_model.encode_text(text_tokens).float()
        
        return lang_feat
    
    def forward(
        self, 
        images: torch.Tensor, 
        robot_state_obs: torch.Tensor,
        task_descriptions: list[str]
    ) -> torch.Tensor:
        'Forward pass'
        B, T = images.shape[0], images.shape[1]
        
        # Encode images (separate RGB and robot_state)
        img_features = self.encode_images(images, robot_state_obs)
        rgb_feat = img_features['rgb']            # [B, T, rgb_feat_dim]
        robot_state = img_features['robot_state']  # [B, T, state_dim]
        
        # Encode language
        lang_feat = self.encode_text(task_descriptions)  # [B, lang_feat_dim]
        
        # Broadcast language features to time dimension
        lang_feat_broadcast = lang_feat.unsqueeze(1).expand(-1, T, -1)  # [B, T, lang_feat_dim]
        
        # Concatenate language and state as condition features
        cond_feat = torch.cat([lang_feat_broadcast, robot_state], dim=-1)  # [B, T, lang_feat_dim + state_dim]
        
        # Modulate RGB features with FiLM
        modulated_rgb = self.film(rgb_feat, cond_feat)  # [B, T, rgb_feat_dim]
        
        # Flatten time dimension (modulated RGB + robot_state)
        modulated_rgb_flat = modulated_rgb.reshape(B, T * self.rgb_feat_dim)  # [B, T*rgb_feat_dim]
        robot_state_flat = robot_state.reshape(B, T * self.robot_state_dim)    # [B, T*state_dim]
        
        # Final concatenation: modulated RGB + robot_state + language features (global information)
        final_feat = torch.cat([
            modulated_rgb_flat,    # [B, T*rgb_feat_dim]
            robot_state_flat,      # [B, T*state_dim]
            lang_feat              # [B, lang_feat_dim]
        ], dim=-1)  # [B, T*rgb_feat_dim + T*state_dim + lang_feat_dim]
        
        return final_feat


if __name__ == "__main__":
    """Test the encoder."""
    print("\n" + "="*70)
    print("Test ResNetCLIPEncoder with FiLM (language and state to image)")
    print("="*70)
    
    # Create shape_meta
    shape_meta = {
        "obs": {
            "img_0": {"shape": [3, 128, 128], "type": "rgb"},
            "img_1": {"shape": [3, 128, 128], "type": "rgb"},
            "img_2": {"shape": [3, 128, 128], "type": "rgb"},
            "img_3": {"shape": [3, 128, 128], "type": "rgb"},
            "img_4": {"shape": [3, 128, 128], "type": "rgb"},
            "robot_state": {"shape": [10], "type": "low_dim"}
        }
    }
    
    # Create encoder
    encoder = ClipResNet(
        shape_meta=shape_meta,
        clip_model_name="RN50",
        device="cuda",
        freeze_clip=True,
    ).cuda()
    
    # Test data
    B, T, num_cameras, H, W, C = 2, 2, 5, 128, 128, 3
    state_dim = 10
    
    images = torch.randn(B, T, num_cameras, H, W, C).cuda()
    robot_state = torch.randn(B, T, state_dim).cuda()
    task_descriptions = ["close the door", "open the box"]
    
    print(f"\nInputs:")
    print(f"  Images: {images.shape}")
    print(f"  Robot state: {robot_state.shape}")
    print(f"  Task descriptions: {task_descriptions}")
    
    # Forward Propagation
    output = encoder(images, robot_state, task_descriptions)
    
    print(f"\nOutputs:")
    print(f"  Fused features: {output.shape}")
    print(f"  Output dimension: {output.shape[-1]}")
    
    # Parameter Statistics
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    film_params = sum(p.numel() for p in encoder.film.parameters() if p.requires_grad)
    
    print(f"\n📊 Parameter statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  FiLM parameters: {film_params:,}")
    
    del encoder
    torch.cuda.empty_cache()

    print("\n✅ FiLM fusion test completed (language and state to image).")
    print("="*70)
