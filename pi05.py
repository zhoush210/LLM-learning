import torch
import torch.nn as nn

class Pi0_Pi05_Forward(nn.Module):
    def __init__(self, is_pi05=False):
        super().__init__()
        self.is_pi05 = is_pi05
        # ... 初始化网络层 (省略) ...

    def forward(self, images, text, state, noisy_actions, timestep):
        # ==========================================
        # 1. 处理 Prefix (前缀): 视觉和文本 Tokens
        # ==========================================
        img_tokens = self.siglip(images)         # [B, N_img, dim]
        text_tokens = self.llm_embed(text)       # [B, N_txt, dim]
        prefix_tokens = torch.cat([img_tokens, text_tokens], dim=1)

        # ==========================================
        # 2. 处理 Suffix (后缀): 状态和动作 Tokens
        # ==========================================
        action_tokens = self.action_in_proj(noisy_actions)  # [B, action_horizon, dim]
        time_emb = self.sincos_emb(timestep)                # [B, dim]

        if not self.is_pi05:
            # 【pi0 专属逻辑】
            # a. 连续状态投影为 1 个 Token
            state_token = self.state_proj(state).unsqueeze(1) 
            
            # b. 将时间步 Copy 匹配 action 的长度，并沿特征维度 Concat
            time_tokens = time_emb.unsqueeze(1).repeat(1, action_horizon, 1)
            action_time_concat = torch.cat([action_tokens, time_tokens], dim=-1)
            action_expert_tokens = self.action_time_mlp(action_time_concat)
            
            # Suffix 包含：状态 Token + 动作 Tokens
            suffix_tokens = torch.cat([state_token, action_expert_tokens], dim=1)
            adarms_cond = None
            
        else:
            # 【pi0.5 专属逻辑】
            # a. 连续状态投影被移除 (源码中 if not pi05 才加 state_token)
            
            # b. 时间步通过 MLP 生成 adarms 的条件变量
            adarms_cond = self.time_mlp(time_emb)
            
            # Suffix 仅包含：动作 Tokens (不和时间步 concat)
            suffix_tokens = action_tokens 

        # ==========================================
        # 3. 核心大模型主干 (PaliGemma)
        # ==========================================
        # 将 Prefix 和 Suffix 拼成一个长序列喂给大模型
        all_tokens = torch.cat([prefix_tokens, suffix_tokens], dim=1)
        
        # 注意：这里会用到一个特殊的 Mask，让 Suffix 可以 attend 到 Prefix，
        # 并在 pi0.5 中通过 adarms_cond 注入时间步噪声等级
        out_tokens = self.llm(all_tokens, attention_mask=..., adarms_cond=adarms_cond)

        # ==========================================
        # 4. 提取流匹配的目标速度 (v_t)
        # ==========================================
        # 只取序列最后 action_horizon 长度的输出
        suffix_out = out_tokens[:, -action_horizon:]
        v_t = self.action_out_proj(suffix_out)

        return v_t