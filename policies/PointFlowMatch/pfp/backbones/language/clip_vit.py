import torch
import torch.nn as nn
from torchvision.transforms import Resize, Normalize
from pfp.backbones.language.clip import load_clip, tokenize
from pfp.backbones.common.cross_attention import SafetyCrossAttn


class ClipViT(nn.Module):
    def __init__(
        self,
        shape_meta: dict,
        device: str = "cuda",
        freeze_clip: bool = True,
        image_out_dim: int = 512,
        lang_out_dim: int = 512,
        robot_out_dim: int = 512,
        safety_out_dim: int = 256,
        n_head: int = 8,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.device = device
        self.freeze_clip = freeze_clip

        # load CLIP as language and image encoder
        self.clip_model, _ = load_clip("ViT-B/16", device=device, jit=False)
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        self.clip_model.eval()

        self.clip_embed_dim = self.clip_model.text_projection.shape[1]

        # get shape info
        self.num_cameras = len([k for k in shape_meta["obs"].keys() if k.startswith("img_")])
        self.robot_dim = shape_meta["obs"].get("robot_state", {}).get("shape", [0])[0]
        
        # image preprocessing
        self.image_resize = Resize((224, 224), antialias=True)
        self.clip_normalize = Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )

        # trainable projection layers
        self.img_proj = nn.Sequential(
            nn.Linear(self.clip_embed_dim, image_out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(image_out_dim, image_out_dim)
        )
        self.lang_proj = nn.Sequential(
            nn.Linear(self.clip_embed_dim, lang_out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(lang_out_dim, lang_out_dim)
        )
        self.robot_proj = nn.Sequential(
            nn.Linear(self.robot_dim, robot_out_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(robot_out_dim // 2, robot_out_dim)
        )
        assert image_out_dim == lang_out_dim == robot_out_dim, "For Cross-Attention, image_out_dim, lang_out_dim and robot_out_dim must be the same"

        # trainable corss attention module
        self.safety = SafetyCrossAttn(
            nhead=n_head,
            img_in_dim=image_out_dim,
            state_in_dim=robot_out_dim,
            lang_in_dim=lang_out_dim,
            safety_dim=safety_out_dim,
            attn_dropout=attn_dropout
        )

        self.img_out_dim = image_out_dim
        self.lang_out_dim = lang_out_dim
        self.robot_out_dim = robot_out_dim
        self.safety_out_dim = safety_out_dim

        self.out_dim = self.num_cameras * self.img_out_dim + self.robot_out_dim + self.lang_out_dim + self.safety_out_dim

        print("Clip ViT B 16 initialized with:")
        print(f"   - Freeze: {freeze_clip}")
        print(f"   - Number of cameras: {self.num_cameras}")
        print(f"   - Image output dimension: {self.img_out_dim}")
        print(f"   - Language output dimension: {self.lang_out_dim}")
        print(f"   - Robot output dimension: {self.robot_out_dim}")
        print(f"   - Safety output dimension: {self.safety_out_dim}")
        print(f"   - Total output dimension: {self.out_dim}")

    def _prep_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.dtype not in [torch.float16, torch.float32, torch.float64]:
            images = images.float()
        if images.max() > 1.0:
            images = images / 255.0
        x = self.image_resize(images)
        x = self.clip_normalize(x)
        return x
    
    def _encode_text(self, task_descriptions) -> torch.Tensor:
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
        task_descriptions,
        is_safe:torch.Tensor,
        human_intervention:torch.Tensor,
        call_human_help:torch.Tensor,    
    ) -> torch.Tensor:
        
        B, T, N_cam = images.shape[0], images.shape[1], images.shape[2]
        if N_cam != self.num_cameras:
            raise ValueError(f"Expected {self.num_cameras} cameras, but got {N_cam}")
        
        # -------- Imgae -> patch tokens --------
        x = images.reshape(B * T * N_cam, *images.shape[3:])  # [B*T*N_cam, H, W, 3]
        x = x.permute(0, 3, 1, 2).contiguous()  # [B*T*N_cam, 3, H, W]
        x = self._prep_images(x)  # [B*T*N_cam, 3, 224, 224]
        x = x.to(dtype=self.clip_model.dtype)

        if self.freeze_clip:
            with torch.no_grad():
                tokens_width = self.clip_model.visual.forward_spatial(x)  # (B*T*N, P, width)
        else:
            tokens_width = self.clip_model.visual.forward_spatial(x)  # (B*T*N, P, width)
        
        proj = self.clip_model.visual.proj              # (width, clip_embed_dim)
        tokens_embed = (tokens_width @ proj).float()    # [B*T*N_cam, P, clip_embed_dim]
        images_out = self.img_proj(tokens_embed)        # [B*T*N_cam, P, img_out_dim]
        
        P = images_out.shape[1]
        img_tokens = images_out.view(B, T, N_cam, P, self.img_out_dim)  # [B, T, N_cam, P, img_out_dim]

        # global image feature: pool patches then pool time then flatten cameras
        img_cam_step = img_tokens.mean(dim=3)                      # (B,T,N,D)
        img_cam = img_cam_step.mean(dim=1)                         # (B,N,D)
        img_global = img_cam.reshape(B, N_cam * self.img_out_dim)  # (B,N*D)

        # -------- Language instruction --------
        lang_feat = self._encode_text(task_descriptions)  # [B, clip_embed_dim]
        if lang_feat is None:
            lang_feat = torch.zeros(B, self.clip_embed_dim, device=images.device)
        else:
            lang_feat = self.lang_proj(lang_feat)  # [B, lang_out_dim]

        # -------- Robot state --------
        s = robot_state_obs.reshape(B * T, self.robot_dim)  # [B*T, robot_dim]
        s_feat = self.robot_proj(s).view(B, T, self.robot_out_dim)  # [B, T, robot_out_dim]
        s_global = s_feat.mean(dim=1)  # [B, robot_out_dim]

        # ------- Safety information --------
        safety_feat = self.safety(
            img_tokens=img_tokens,     # [B,T,N,P,D]
            robot_state_obs=s_feat,   # [B,T,robot_out_dim]
            lang_feat=lang_feat,     # [B, lang_out_dim]
            safety_value=is_safe,
            human_intervention_value=human_intervention,
            call_human_help=call_human_help,
        )  # [B,T,safety_out_dim]

        # Concat all features:
        safety_global = safety_feat.mean(dim=1)  # [B,safety_out_dim]
        fused_global = torch.cat([img_global, s_global, lang_feat, safety_global], dim=-1)

        return fused_global, safety_feat  # [B,total_out_dim], [B,T,safety_out_dim]


if __name__ == "__main__":
    """Test the cross-attention encoder."""
    print("\n" + "="*70)
    print("Test ClipViT with cross-attention")
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = ClipViT(
        shape_meta=shape_meta,
        device=device,
        freeze_clip=True,
        image_out_dim=512,
        lang_out_dim=512,
        robot_out_dim=512,
        safety_out_dim=256,
        n_head=8,
    ).to(device)
    
    # Test data
    B, T, num_cameras, H, W, C = 4, 2, 5, 128, 128, 3
    state_dim = 10
    
    images = torch.randn(B, T, num_cameras, H, W, C).to(device)
    robot_state = torch.randn(B, T, state_dim).to(device)
    is_safe = torch.randint(0, 2, (B, T, 1), device=device).float()
    human_intervention = torch.randint(0, 2, (B, T, 1), device=device).float()
    call_human_help = torch.randint(0, 2, (B, T, 1), device=device).float()
    task_descriptions = ["close the door"] * B
    
    print(f"\nInputs:")
    print(f"  Images: {images.shape}")
    print(f"  Robot state: {robot_state.shape}")
    print(f"  is_safe: {is_safe.shape}")
    print(f"  human_intervention: {human_intervention.shape}")
    print(f"  call_human_help: {call_human_help.shape}")
    print(f"  Task descriptions: {task_descriptions}")
    
    # Forward Propagation
    fused, safety_feat = encoder(
        images,
        robot_state,
        task_descriptions,
        is_safe,
        human_intervention,
        call_human_help,
    )
    
    print(f"\nOutputs:")
    print(f"  Fused features: {fused.shape}")
    print(f"  Safety features: {safety_feat.shape}")
    print(f"  Output dimension: {fused.shape[-1]}")
    
    # Parameter Statistics
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\n📊 Parameter statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    del encoder
    if device == "cuda":
        torch.cuda.empty_cache()

    print("\n" + "="*70)
    print("✅ ClipViT cross-attention test completed")
    print("="*70)
