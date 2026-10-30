from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
import hydra
import copy
import torch.nn.functional as F
from omegaconf import OmegaConf
from composer.models import ComposerModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from pfp.policy.base_policy import BasePolicy
from pfp import DEVICE, REPO_DIRS

class DiTPolicy(ComposerModel, BasePolicy):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        n_obs_steps: int,
        n_pred_steps: int,
        num_k_infer: int,
        obs_encoder: nn.Module,
        dit_net: nn.Module,
        augment_data: bool = False,
        loss_weights: dict[int] = None,
        norm_pcd_center: list = None,
        diffusion_steps: int = 100,
    ) -> None:
        ComposerModel.__init__(self)
        BasePolicy.__init__(self, n_obs_steps)
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_obs_steps = n_obs_steps
        self.n_pred_steps = n_pred_steps
        self.num_k_infer = num_k_infer
        self.obs_encoder = obs_encoder
        self.dit_net = dit_net
        self.norm_pcd_center = norm_pcd_center
        self.augment_data = augment_data
        self.l_w = loss_weights
        self.diffusion_steps = diffusion_steps
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=diffusion_steps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )
        self.ny_shape = (n_pred_steps, y_dim)
        return

    def set_num_k_infer(self, num_k_infer: int):
        self.num_k_infer = num_k_infer
        return

    def _norm_obs(self, obs: torch.Tensor) -> torch.Tensor:
        obs[..., :3] -= torch.tensor(self.norm_pcd_center, device=DEVICE)
        return obs

    def _norm_robot_state(self, robot_state: torch.Tensor) -> torch.Tensor:
        robot_state[..., :3] -= torch.tensor(self.norm_pcd_center, device=DEVICE)
        robot_state[..., 9] -= torch.tensor(0.5, device=DEVICE)
        return robot_state

    def _denorm_robot_state(self, robot_state: torch.Tensor) -> torch.Tensor:
        robot_state[..., :3] += torch.tensor(self.norm_pcd_center, device=DEVICE)
        robot_state[..., 9] += torch.tensor(0.5, device=DEVICE)
        return robot_state

    def _norm_data(self, batch):
        required = [
            "images",
            "robot_state_obs",
            "robot_state_pred",
            "language_tokens",
            "is_safe",
            "human_intervention",
            "call_human_help",
        ]
        if not isinstance(batch, dict):
            raise TypeError(f"Batch must be dict with keys {required}, got type: {type(batch)}")
        missing = [k for k in required if k not in batch]
        if missing:
            raise ValueError(f"Batch missing required keys: {missing}")
        obs = batch["images"]
        robot_state_obs = batch["robot_state_obs"]
        robot_state_pred = batch["robot_state_pred"]
        language_tokens = batch["language_tokens"]
        is_safe = batch["is_safe"]
        human_intervention = batch["human_intervention"]
        call_human_help = batch["call_human_help"]

        obs = self._norm_obs(obs)
        robot_state_obs = self._norm_robot_state(robot_state_obs)
        robot_state_pred = self._norm_robot_state(robot_state_pred)

        out = {
            'obs': obs,
            'robot_state_obs': robot_state_obs,
            'robot_state_pred': robot_state_pred,
            'language_tokens': language_tokens,
            'is_safe': is_safe,
            'human_intervention': human_intervention,
            'call_human_help': call_human_help,
        }
        return out

    def forward(self, batch):
        # Only for placeholder, not used in practice
        return 0

    def loss(self, outputs, batch) -> torch.Tensor:
        with torch.no_grad():
            batch = self._norm_data(batch)
            if self.augment_data:
                # Optional: Implement data augmentation
                pass
        obs = batch['obs']
        robot_state_obs = batch['robot_state_obs']
        robot_state_pred = batch['robot_state_pred']
        language_tokens = batch['language_tokens']
        is_safe = batch['is_safe']
        human_intervention = batch['human_intervention']
        call_human_help = batch['call_human_help']
        # Encode observations
        obs_enc, _ = self.obs_encoder(
            obs,
            robot_state_obs,
            language_tokens,
            is_safe,
            human_intervention,
            call_human_help,
        )
        # Sample noise
        B = obs_enc.shape[0]
        actions = robot_state_pred
        noise = torch.randn_like(actions)
        timesteps = torch.randint(0, self.diffusion_steps, (B,), device=DEVICE).long()
        noise_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        # Predict with DiT network
        _, noise_pred = self.dit_net(noise_actions, timesteps, obs_enc)
        # loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        self.logger.log_metrics({"loss/train/dit": loss.item()})
        return loss

    def eval_forward(self, batch, outputs=None) -> torch.Tensor:
        batch = self._norm_data(batch)
        obs = batch['obs']
        robot_state_obs = batch['robot_state_obs']
        robot_state_pred = batch['robot_state_pred']
        language_tokens = batch['language_tokens']
        is_safe = batch['is_safe']
        human_intervention = batch['human_intervention']
        call_human_help = batch['call_human_help']
        pred_y = self.infer_y(obs, robot_state_obs, language_tokens, is_safe, human_intervention, call_human_help)
        mse = nn.functional.mse_loss(pred_y, robot_state_pred)
        self.logger.log_metrics({"metrics/eval/dit_mse": mse.item()})
        return pred_y

    def infer_y(
        self,
        obs: torch.Tensor,
        robot_state_obs: torch.Tensor,
        language_tokens: torch.Tensor,
        is_safe: torch.Tensor,
        human_intervention: torch.Tensor,
        call_human_help: torch.Tensor,
        noise=None,
        return_traj=False,
    ) -> torch.Tensor:
        obs_enc, _ = self.obs_encoder(
            obs,
            robot_state_obs,
            language_tokens,
            is_safe,
            human_intervention,
            call_human_help,
        )
        B = obs_enc.shape[0]
        z = torch.randn((B, *self.ny_shape), device=DEVICE) if noise is None else noise
        traj = [z]
        self.noise_scheduler.set_timesteps(self.num_k_infer)
        for timestep in self.noise_scheduler.timesteps:
            batched_timestep = timestep.unsqueeze(0).repeat(B).to(DEVICE)
            _, noise_pred = self.dit_net(z, batched_timestep, obs_enc)
            z = self.noise_scheduler.step(
                model_output=noise_pred, timestep=timestep, sample=z
            ).prev_sample
            traj.append(z)
        if return_traj:
            return torch.stack(traj)
        return traj[-1]

    @classmethod
    def load_from_checkpoint(
        cls,
        ckpt_name: str,
        ckpt_episode: str,
        num_k_infer: int = None,
        **kwargs,
    ):
        ckpt_dir = REPO_DIRS.CKPT / ckpt_name
        ckpt_path_list = list(ckpt_dir.glob(f"{ckpt_episode}*"))
        assert len(ckpt_path_list) > 0, f"No checkpoint found in {ckpt_dir} with {ckpt_episode}"
        assert len(ckpt_path_list) < 2, f"Multiple ckpts found in {ckpt_dir} with {ckpt_episode}"
        ckpt_fpath = ckpt_path_list[0]

        state_dict = torch.load(ckpt_fpath, map_location=DEVICE)
        cfg = OmegaConf.load(ckpt_dir / "config.yaml")
        assert cfg.model._target_.split(".")[-1] == cls.__name__
        model: DiTPolicy = hydra.utils.instantiate(cfg.model)

        # If the checkpoint contains dit_net.obs_proj but the instantiated model
        # does not yet have that module (it may be created lazily in forward),
        # create it here to allow loading the saved weights.
        ckpt_keys = set(state_dict["state"]["model"].keys())
        # look for weight/bias keys for obs_proj
        weight_key = "dit_net.obs_proj.weight"
        bias_key = "dit_net.obs_proj.bias"
        if (weight_key in ckpt_keys or bias_key in ckpt_keys) and not any(
            k.startswith("dit_net.obs_proj") for k in model.state_dict().keys()
        ):
            # infer shapes from checkpoint tensors
            if weight_key in state_dict["state"]["model"]:
                w = state_dict["state"]["model"][weight_key]
                out_f, in_f = int(w.shape[0]), int(w.shape[1])
            else:
                # fallback: try to infer from bias
                b = state_dict["state"]["model"][bias_key]
                out_f = int(b.shape[0])
                # unknown in_f -> assume hidden_dim equals out_f (best-effort)
                in_f = out_f
            # create the Linear module and register it
            try:
                model.dit_net.obs_proj = nn.Linear(in_f, out_f)
            except Exception:
                # fallback: attach under dit_net as attribute
                setattr(model.dit_net, "obs_proj", nn.Linear(in_f, out_f))

        # Process obs_encoder.film module if checkpoint exists but current model does not
        film_keys_prefix = "obs_encoder.film.mlp."
        film_keys_in_ckpt = [k for k in ckpt_keys if k.startswith(film_keys_prefix)]
        has_film_in_model = any("obs_encoder.film" in k for k in model.state_dict().keys())
        
        if film_keys_in_ckpt and not has_film_in_model:
            # Infer FiLM dimensions from checkpoint
            # mlp.0: Linear(cond_feat_dim, cond_feat_dim)
            # mlp.2: Linear(cond_feat_dim, img_feat_dim * 2)
            weight_0 = state_dict["state"]["model"].get(f"{film_keys_prefix}0.weight")
            weight_2 = state_dict["state"]["model"].get(f"{film_keys_prefix}2.weight")
            
            if weight_0 is not None and weight_2 is not None:
                cond_feat_dim = int(weight_0.shape[0])
                img_feat_dim_x2 = int(weight_2.shape[0])
                img_feat_dim = img_feat_dim_x2 // 2
                
                # Reconstruct FiLM module (following FiLM structure in ClipResNet/ClipViT)
                film_mlp = nn.Sequential(
                    nn.Linear(cond_feat_dim, cond_feat_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(cond_feat_dim, img_feat_dim * 2)
                )
                
                # Create a simple FiLM wrapper module
                class FiLMModule(nn.Module):
                    def __init__(self, mlp, img_feat_dim, cond_feat_dim):
                        super().__init__()
                        self.mlp = mlp
                        self.img_feat_dim = img_feat_dim
                        self.cond_feat_dim = cond_feat_dim
                    
                    def forward(self, img_feats, cond_feats):
                        B, T, D = img_feats.shape
                        gb = self.mlp(cond_feats)
                        gb = gb.view(B, T, 2, D)
                        gamma = gb[:, :, 0]
                        beta = gb[:, :, 1]
                        return img_feats * (1.0 + gamma) + beta
                
                film_module = FiLMModule(film_mlp, img_feat_dim, cond_feat_dim)
                model.obs_encoder.film = film_module

        # Load weights permissively; any remaining mismatches will be ignored
        model.load_state_dict(state_dict["state"]["model"], strict=True)
        model.to(DEVICE)
        model.eval()
        if num_k_infer is not None:
            model.set_num_k_infer(num_k_infer)
        return model
    

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return nn.GELU(approximate="tanh")
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def _with_pos_embed(tensor, pos=None):
    return tensor if pos is None else tensor + pos


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)

        Returns:
            Tensor of shape (seq_len, batch_size, d_model) with positional encodings added
        """
        pe = self.pe[: x.shape[0]]
        pe = pe.repeat((1, x.shape[1], 1))
        return pe.detach().clone()


class _TimeNetwork(nn.Module):
    def __init__(self, time_dim, out_dim, learnable_w=False):
        assert time_dim % 2 == 0, "time_dim must be even!"
        half_dim = int(time_dim // 2)
        super().__init__()

        w = np.log(10000) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        self.register_parameter("w", nn.Parameter(w, requires_grad=learnable_w))

        self.out_net = nn.Sequential(
            nn.Linear(time_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        assert len(x.shape) == 1, "assumes 1d input timestep array"
        x = x[:, None] * self.w[None]
        x = torch.cat((torch.cos(x), torch.sin(x)), dim=1)
        return self.out_net(x)


class _SelfAttnEncoder(nn.Module):
    def __init__(
        self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="gelu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, pos):
        q = k = _with_pos_embed(src, pos)
        src2, _ = self.self_attn(q, k, value=src, need_weights=False)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class _ShiftScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)
        self.shift = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c)[None] + self.shift(c)[None]

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.scale.weight)
        nn.init.xavier_uniform_(self.shift.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.bias)


class _ZeroScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c)[None]

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)


class _DiTDecoder(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # create modulation layers
        self.attn_mod1 = _ShiftScaleMod(d_model)
        self.attn_mod2 = _ZeroScaleMod(d_model)
        self.mlp_mod1 = _ShiftScaleMod(d_model)
        self.mlp_mod2 = _ZeroScaleMod(d_model)

    def forward(self, x, t, cond):
        # process the conditioning vector first
        cond = torch.mean(cond, axis=0)
        cond = cond + t

        x2 = self.attn_mod1(self.norm1(x), cond)
        x2, _ = self.self_attn(x2, x2, x2, need_weights=False)
        x = self.attn_mod2(self.dropout1(x2), cond) + x

        x2 = self.mlp_mod1(self.norm2(x), cond)
        x2 = self.linear2(self.dropout2(self.activation(self.linear1(x2))))
        x2 = self.mlp_mod2(self.dropout3(x2), cond)
        return x + x2

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for s in (self.attn_mod1, self.attn_mod2, self.mlp_mod1, self.mlp_mod2):
            s.reset_parameters()


class _FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, t, cond):
        # process the conditioning vector first
        cond = torch.mean(cond, axis=0)
        cond = cond + t

        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        x = x * scale[None] + shift[None]
        x = self.linear(x)
        return x.transpose(0, 1)

    def reset_parameters(self):
        for p in self.parameters():
            nn.init.zeros_(p)


class _TransformerEncoder(nn.Module):
    def __init__(self, base_module, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(base_module) for _ in range(num_layers)]
        )

        for l in self.layers:
            l.reset_parameters()

    def forward(self, src, pos):
        x, outputs = src, []
        for layer in self.layers:
            x = layer(x, pos)
            outputs.append(x)
        return outputs


class _TransformerDecoder(_TransformerEncoder):
    def forward(self, src, t, all_conds):
        x = src
        for layer, cond in zip(self.layers, all_conds):
            x = layer(x, t, cond)
        return x


class _DiTNoiseNet(nn.Module):
    def __init__(
        self,
        ac_dim,
        ac_chunk,
        time_dim=256,
        hidden_dim=512,
        num_blocks=6,
        dropout=0.1,
        dim_feedforward=2048,
        nhead=8,
        activation="gelu",
    ):
        super().__init__()

        # positional encoding blocks
        self.enc_pos = _PositionalEncoding(hidden_dim)
        self.register_parameter(
            "dec_pos",
            nn.Parameter(torch.empty(ac_chunk, 1, hidden_dim), requires_grad=True),
        )
        nn.init.xavier_uniform_(self.dec_pos.data)

        # input encoder mlps
        self.time_net = _TimeNetwork(time_dim, hidden_dim)
        self.ac_proj = nn.Sequential(
            nn.Linear(ac_dim, ac_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ac_dim, hidden_dim),
        )

        # obs encoder projection (to hidden_dim)
        self.obs_proj = None  # will be set in forward_enc if needed

        # encoder blocks
        encoder_module = _SelfAttnEncoder(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.encoder = _TransformerEncoder(encoder_module, num_blocks)

        # decoder blocks
        decoder_module = _DiTDecoder(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.decoder = _TransformerDecoder(decoder_module, num_blocks)

        # turns predicted tokens into epsilons
        self.eps_out = _FinalLayer(hidden_dim, ac_dim)

        print(
            "number of diffusion parameters: {:e}".format(
                sum(p.numel() for p in self.parameters())
            )
        )

    def forward(self, noise_actions, time, obs_enc, enc_cache=None):
        if enc_cache is None:
            enc_cache = self.forward_enc(obs_enc)
        return enc_cache, self.forward_dec(noise_actions, time, enc_cache)
    
    def forward_enc(self, obs_enc):
        # Obs_enc is compatible with [batch, feature_dim] or [batch, seq_len, feature_dim]
        if obs_enc.dim() == 2:
            obs_enc = obs_enc.unsqueeze(1)  # [B, 1, feat_dim]
        B, seq_len, feat_dim = obs_enc.shape
        hidden_dim = self.enc_pos.pe.shape[-1]
        if feat_dim != hidden_dim:
            # create obs_proj if not already created
            if self.obs_proj is None:
                self.obs_proj = nn.Linear(feat_dim, hidden_dim).to(obs_enc.device)
            obs_enc = self.obs_proj(obs_enc)
        obs_enc = obs_enc.transpose(0, 1)  # [seq_len, B, hidden_dim]
        pos = self.enc_pos(obs_enc)
        enc_cache = self.encoder(obs_enc, pos)
        return enc_cache

    def forward_dec(self, noise_actions, time, enc_cache):
        time_enc = self.time_net(time)
        
        ac_tokens = self.ac_proj(noise_actions)
        ac_tokens = ac_tokens.transpose(0, 1)
        dec_in = ac_tokens + self.dec_pos

        # apply decoder
        dec_out = self.decoder(dec_in, time_enc, enc_cache)

        # apply final epsilon prediction layer
        return self.eps_out(dec_out, time_enc, enc_cache[-1])
