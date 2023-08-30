# %%
import torch
import numpy as np
import math
import utils

# %%
E = 0.572  # 572Pa
t_0 = 1  # 1s
gamma = torch.Tensor([0.2])
theta = torch.pi / 10.0  # Conical
a = 8.0 / (3.0 * torch.pi) * np.tan(theta)
b = torch.Tensor([2.0])
v = 10.0  # 10um/s
# %%
coeff = E * t_0**gamma * a * b * v**b
coeff = coeff * utils.beta(b, 1.0 - gamma)


# %%
def PLR_Force(time, coeff, b, gamma):
    return coeff * time ** (b - gamma)


def t1(time, t_max):
    coeff = 2.0 ** (1.0 / (1.0 - gamma))
    return torch.clamp(time - coeff * (time - t_max), 0.0, None)


# %%
t_array = torch.arange(0.0, 0.401, 0.001)
t_max = (t_array[0] + t_array[len(t_array) - 1]) / 2
print(t_array)


# %%
class PLR(torch.nn.Module):
    def __init__(self):
        """
        생성자에서 4개의 매개변수를 생성(instantiate)하고, 멤버 변수로 지정합니다.
        """
        super().__init__()
        self.E = torch.nn.Parameter(torch.randn(()))
        self.G = torch.nn.Parameter(torch.randn(()))
        self.a = 8.0 / (3.0 * torch.pi) * np.tan(theta)
        self.b = torch.Tensor([2.0])
        self.v = torch.Tensor([10.0])
        self.t_0 = torch.Tensor([1])

    def forward(self, t):
        """
        순전파 함수에서는 입력 데이터의 텐서를 받고 출력 데이터의 텐서를 반환해야 합니다.
        텐서들 간의 임의의 연산뿐만 아니라, 생성자에서 정의한 Module을 사용할 수 있습니다.
        """
        return self.E * self.t_0**self.G * self.a * self.b * v**self.b

    def string(self):
        """
        Python의 다른 클래스(class)처럼, PyTorch 모듈을 사용해서 사용자 정의 메소드를 정의할 수 있습니다.
        """
        return f"relaxation function = {self.E.item()}*{t}^{self.G.item()}"


# %%

# 입력값과 출력값을 갖는 텐서들을 생성합니다.
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# 위에서 정의한 클래스로 모델을 생성합니다.
# model = Polynomial3()

# 손실 함수와 optimizer를 생성합니다. SGD 생성자에 model.paramaters()를 호출해주면
# 모델의 멤버 학습 가능한 (torch.nn.Parameter로 정의된) 매개변수들이 포함됩니다.
criterion = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
for t in range(2000):
    # 순전파 단계: 모델에 x를 전달하여 예측값 y를 계산합니다.
    y_pred = model(x)

    # 손실을 계산하고 출력합니다.
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # 변화도를 0으로 만들고, 역전파 단계를 수행하고, 가중치를 갱신합니다.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Result: {model.string()}")
# %%
