import numpy as np
from pyswarm import pso

# Tọa độ của các cá thể
individuals = np.random.rand(50, 2) * 10


# Số lượng đầu đọc RFID
num_rfid_readers = 10

rfid_radius = 2  # Bán kính của mỗi đầu đọc RFID

# Hàm mục tiêu
def objective_function(positions):
    positions = positions.reshape((num_rfid_readers, 2))
    overlap_penalty = 0
    coverage_penalty = 0

    # Kiểm tra sự chồng lấp giữa các đầu đọc RFID
    for i in range(num_rfid_readers):
        for j in range(i + 1, num_rfid_readers):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < 2 * rfid_radius:
                overlap_penalty += (2 * rfid_radius - dist) ** 2

    # Kiểm tra tất cả cá thể được bao phủ
    for individual in individuals:
        covered = False
        for reader in positions:
            if np.linalg.norm(individual - reader) <= rfid_radius:
                covered = True
                break
        if not covered:
            coverage_penalty += 1

    # Tổng hợp hình phạt
    return overlap_penalty + coverage_penalty * 100

# Các giới hạn
lb = np.zeros(2 * num_rfid_readers)
ub = np.ones(2 * num_rfid_readers) * 10

# Thực hiện PSO
best_positions, best_cost = pso(objective_function, lb, ub, swarmsize=30, maxiter=100)

# Vẽ kết quả
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# Vẽ các cá thể dưới dạng các điểm màu xanh
ax.scatter(individuals[:, 0], individuals[:, 1], color='blue', label='Individuals')

# Vẽ các đầu đọc RFID tối ưu
best_positions = best_positions.reshape((num_rfid_readers, 2))
for x, y in best_positions:
    ax.plot(x, y, 'o', color='red', label='RFID Reader' if np.array_equal((x, y), best_positions[0]) else "")
    circle = plt.Circle((x, y), rfid_radius, color='red', alpha=0.3)
    ax.add_patch(circle)

# Đặt nhãn trục và tiêu đề
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_title('Optimized RFID Reader Positions using PSO')

# Thiết lập tỷ lệ của biểu đồ
ax.set_aspect('equal', 'box')

# Hiển thị chú thích (legend)
ax.legend()

# Hiển thị biểu đồ
plt.show()
