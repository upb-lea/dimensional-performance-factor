import numpy as np
from matplotlib import pyplot as plt

# Problem definition
R = 7.5e-3
T_ambient = 50
h = (29.5 - 1.8) / 2 / 1000  # half center leg core length (window height - air gap length)
z = np.linspace(0, h, 20)
p_v = 500_000  # W / m³
k = 5  # W/mK


T = T_ambient + p_v / (2*k) * (h**2 - z**2)

print(f"total power in cylinder = {p_v * h * R**2 * np.pi}")
print(f"Tmax = {max(T)}")

plt.plot(z, T)
plt.show()

