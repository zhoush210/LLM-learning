import math
import torch
from torch import nn
import torch.nn.functional as F


"""
Flow Matching 最小可运行实现（2D 数据）
------------------------------------
核心思想：
1) 采样基础分布 x0 ~ N(0, I)
2) 采样目标分布样本 x1
3) 在 t ~ U(0,1) 上构造线性插值路径：x_t = (1-t)x0 + t x1
4) 该路径的真实速度场：u_t = d x_t / dt = x1 - x0
5) 训练网络 v_theta(x_t, t) 拟合 u_t（MSE）
6) 采样时从 x(0)=z~N(0,I) 用 ODE Euler 积分到 t=1
"""


class FlowMatchingMLP(nn.Module):
	"""简单的速度场网络 v_theta(x, t) -> R^2。"""

	def __init__(self, x_dim=2, hidden_dim=128):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(x_dim + 1, hidden_dim),
			nn.SiLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.SiLU(),
			nn.Linear(hidden_dim, x_dim),
		)

	def forward(self, x, t):
		# x: (B, x_dim), t: (B, 1)
		return self.net(torch.cat([x, t], dim=-1))


def sample_target_distribution(batch_size, device):
	"""
	目标分布：8 个高斯团组成的 2D 环形混合分布。
	返回形状：(B, 2)
	"""
	k = 8
	# 每个样本随机选择一个簇
	idx = torch.randint(0, k, (batch_size,), device=device)
	angles = 2 * math.pi * idx.float() / k
	centers = torch.stack([2.0 * torch.cos(angles), 2.0 * torch.sin(angles)], dim=-1)
	noise = 0.15 * torch.randn(batch_size, 2, device=device)
	return centers + noise


def flow_matching_loss(model, x1):
	"""
	条件 Flow Matching 损失（线性插值路径）。
	输入 x1: (B, 2)
	"""
	device = x1.device
	batch_size = x1.size(0)

	# 基础分布样本 x0
	x0 = torch.randn_like(x1)
	# t ~ Uniform(0,1)
	t = torch.rand(batch_size, 1, device=device)

	# 路径点 x_t 与真实速度 u_t
	x_t = (1.0 - t) * x0 + t * x1
	u_t = x1 - x0

	v_t = model(x_t, t)
	return F.mse_loss(v_t, u_t)


@torch.no_grad()
def sample_from_model(model, num_samples=1024, num_steps=200, device="cpu"):
	"""
	用 Euler 方法积分 ODE:
	  dx/dt = v_theta(x, t), t in [0,1]
	"""
	model.eval()
	x = torch.randn(num_samples, 2, device=device)
	dt = 1.0 / num_steps

	for step in range(num_steps):
		t_scalar = step / num_steps
		t = torch.full((num_samples, 1), t_scalar, device=device)
		v = model(x, t)
		x = x + v * dt

	return x


def train_flow_matching(
	steps=1000,
	batch_size=512,
	lr=1e-3,
	hidden_dim=128,
	device=None,
):
	"""训练入口，返回训练后的 model 和每一步 loss。"""
	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"

	model = FlowMatchingMLP(x_dim=2, hidden_dim=hidden_dim).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	losses = []
	model.train()
	for step in range(steps):
		x1 = sample_target_distribution(batch_size, device)
		loss = flow_matching_loss(model, x1)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		losses.append(loss.item())

		if (step + 1) % max(1, steps // 5) == 0:
			print(f"[train] step {step + 1}/{steps}, loss={loss.item():.6f}")

	return model, losses


def test_flow_matching_components():
	"""基础功能测试：形状、数值稳定性、训练有效性。"""
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"[test] device = {device}")

	# 1) 前向形状测试
	model = FlowMatchingMLP().to(device)
	x = torch.randn(16, 2, device=device)
	t = torch.rand(16, 1, device=device)
	v = model(x, t)
	assert v.shape == (16, 2), f"模型输出形状错误: {v.shape}"
	assert torch.isfinite(v).all(), "模型输出包含非有限值"
	print("[test] forward shape/finiteness passed")

	# 2) 损失可计算测试
	x1 = sample_target_distribution(32, device)
	loss = flow_matching_loss(model, x1)
	assert loss.ndim == 0, "loss 不是标量"
	assert torch.isfinite(loss), "loss 非有限"
	print(f"[test] loss compute passed, initial loss = {loss.item():.6f}")

	# 3) 短训练有效性测试（只做轻量验证）
	trained_model, losses = train_flow_matching(steps=200, batch_size=256, lr=1e-3, device=device)
	first_mean = sum(losses[:20]) / 20
	last_mean = sum(losses[-20:]) / 20
	print(f"[test] first_mean={first_mean:.6f}, last_mean={last_mean:.6f}")
	# 不做强硬单调要求，但通常应下降
	assert last_mean < first_mean * 1.05, "训练未表现出收敛趋势（末段 loss 未明显优于初段）"
	print("[test] training trend passed")

	# 4) 采样测试
	samples = sample_from_model(trained_model, num_samples=128, num_steps=100, device=device)
	assert samples.shape == (128, 2), f"采样形状错误: {samples.shape}"
	assert torch.isfinite(samples).all(), "采样结果包含非有限值"
	print("[test] sampling passed")

	print("All flow matching tests passed.")


if __name__ == "__main__":
	test_flow_matching_components()
