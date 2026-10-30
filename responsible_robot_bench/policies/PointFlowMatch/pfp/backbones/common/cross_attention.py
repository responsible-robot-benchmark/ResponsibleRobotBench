import torch
import torch.nn as nn


class SafetyCrossAttn(nn.Module):
    """
    Two-stage cross attention for safety features.

    Step1: state query attends to image tokens.
    Step2: language query attends to state-conditioned image tokens.

    Output: per-step safety features [B, T, safety_dim].
    """

    def __init__(
        self,
        nhead: int = 8,
        img_in_dim: int = 512,
        state_in_dim: int = 512,
        lang_in_dim: int = 512,
        safety_dim: int = 256,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = img_in_dim
        self.img_in_dim = img_in_dim
        self.state_in_dim = state_in_dim
        self.lang_in_dim = lang_in_dim

        # Optional projections keep module robust even if upstream dims change.
        self.img_proj = None if img_in_dim == self.d_model else nn.Linear(img_in_dim, self.d_model)
        self.state_proj = None if state_in_dim == self.d_model else nn.Linear(state_in_dim, self.d_model)
        self.lang_proj = None if lang_in_dim == self.d_model else nn.Linear(lang_in_dim, self.d_model)

        self.state_img_attn = nn.MultiheadAttention(
            self.d_model, nhead, dropout=attn_dropout, batch_first=True
        )
        self.lang_img_attn = nn.MultiheadAttention(
            self.d_model, nhead, dropout=attn_dropout, batch_first=True
        )
        self.special_proj = nn.Linear(3, self.d_model)  # for safety_value, human_intervention_value and call_human_help

        self.state_film = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, 2 * self.d_model),
        )

        self.out = nn.Sequential(
            nn.Linear(self.d_model, safety_dim),
            nn.ReLU(inplace=True),
            nn.Linear(safety_dim, safety_dim),
        )

    def _to_step_scalar(self, x: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
        # Return [B, T]
        if x is None:
            return torch.zeros(like.shape[:2], device=like.device, dtype=like.dtype)
        x = x.to(like.device, dtype=like.dtype)
        if x.dim() == 1:
            return x.unsqueeze(1).expand(-1, like.shape[1])
        if x.dim() == 2:
            if x.shape[0] != like.shape[0]:
                raise ValueError(f"Special value batch mismatch: got {x.shape[0]}, expected {like.shape[0]}")
            if x.shape[1] == 1:
                return x.expand(-1, like.shape[1])
            if x.shape[1] == like.shape[1]:
                return x
            raise ValueError(f"Special value time mismatch: got {x.shape[1]}, expected 1 or {like.shape[1]}")
        if x.dim() == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
            if x.shape[0] != like.shape[0] or x.shape[1] != like.shape[1]:
                raise ValueError(
                    f"Special value shape mismatch: got {tuple(x.shape)}, expected ({like.shape[0]}, {like.shape[1]})"
                )
            return x
        raise ValueError(f"Special value must be [B], [B,T], or [B,T,1], got {tuple(x.shape)}")

    def forward(
        self,
        img_tokens: torch.Tensor,       # [B,T,N,P,D]
        robot_state_obs: torch.Tensor,  # [B,T,state_dim]
        lang_feat: torch.Tensor,        # [B,lang_dim]
        safety_value: torch.Tensor = None,
        human_intervention_value: torch.Tensor = None,
        call_human_help: torch.Tensor = None,
        debug: bool = False,
    ) -> torch.Tensor:
        if img_tokens.dim() != 5:
            raise ValueError(f"img_tokens must be [B,T,N,P,D], got {tuple(img_tokens.shape)}")
        if robot_state_obs.dim() != 3:
            raise ValueError(f"robot_state_obs must be [B,T,D], got {tuple(robot_state_obs.shape)}")
        if lang_feat.dim() != 2:
            raise ValueError(f"lang_feat must be [B,D], got {tuple(lang_feat.shape)}")

        B, T, N_cam, P, D_in = img_tokens.shape
        if robot_state_obs.shape[0] != B or robot_state_obs.shape[1] != T:
            raise ValueError(
                f"robot_state_obs shape {tuple(robot_state_obs.shape)} must match batch/time ({B}, {T}, *)"
            )
        if lang_feat.shape[0] != B:
            raise ValueError(f"lang_feat batch {lang_feat.shape[0]} must match image batch {B}")

        img_kv = img_tokens.reshape(B, T, N_cam * P, D_in)  # [B,T,L,D]

        if D_in != self.d_model:
            if self.img_proj is None or self.img_proj.in_features != D_in:
                self.img_proj = nn.Linear(D_in, self.d_model).to(device=img_tokens.device, dtype=img_tokens.dtype)
            img_kv = self.img_proj(img_kv)

        D = self.d_model

        if robot_state_obs.shape[-1] != D:
            if self.state_proj is None or self.state_proj.in_features != robot_state_obs.shape[-1]:
                self.state_proj = nn.Linear(robot_state_obs.shape[-1], D).to(
                    device=robot_state_obs.device, dtype=robot_state_obs.dtype
                )
            q_state = self.state_proj(robot_state_obs)  # [B,T,D]
        else:
            q_state = robot_state_obs

        if lang_feat.shape[-1] != D:
            if self.lang_proj is None or self.lang_proj.in_features != lang_feat.shape[-1]:
                self.lang_proj = nn.Linear(lang_feat.shape[-1], D).to(
                    device=lang_feat.device, dtype=lang_feat.dtype
                )
            lang_feat = self.lang_proj(lang_feat)

        safety_s = self._to_step_scalar(safety_value, q_state)
        intervention_s = self._to_step_scalar(human_intervention_value, q_state)
        call_help_s = self._to_step_scalar(call_human_help, q_state)
        special_vec = torch.stack([safety_s, intervention_s, call_help_s], dim=-1)  # [B,T,3]
        q_state = q_state + self.special_proj(special_vec)  # [B,T,D]

        L = img_kv.shape[2]
        BT = B * T
        img_kv_bt = img_kv.reshape(BT, L, D)
        q_state_bt = q_state.reshape(BT, 1, D)
        q_lang_bt = lang_feat.unsqueeze(1).expand(B, T, D).reshape(BT, 1, D)

        _, attn_w = self.state_img_attn(q_state_bt, img_kv_bt, img_kv_bt, need_weights=True)  # [BT,1,L]
        gate = attn_w.transpose(1, 2)  # [BT,L,1]

        gb = self.state_film(q_state_bt).view(BT, 1, 2, D)
        gamma = gb[:, :, 0]  # [BT,1,D]
        beta = gb[:, :, 1]   # [BT,1,D]
        img_film = img_kv_bt * (1.0 + gamma) + beta
        img_cond = img_film + gate * q_state_bt.expand(-1, L, -1)

        safety_token, _ = self.lang_img_attn(q_lang_bt, img_cond, img_cond)  # [BT,1,D]
        safety_feat = self.out(safety_token.squeeze(1)).reshape(B, T, -1)    # [B,T,safety_dim]

        if debug:
            print(f"[debug] img_kv: {img_kv.shape}")
            print(f"[debug] q_state: {q_state.shape}")
            print(f"[debug] q_lang_bt: {q_lang_bt.shape}")
            print(f"[debug] safety_feat: {safety_feat.shape}")

        return safety_feat
