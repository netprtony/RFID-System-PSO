import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
grid_x, grid_y = 25, 25 # Kích thước của 1 lớp học
num_individuals = 250 # Sô lượng sinh viên
num_rfid_readers = 20  # Số lượng đầu đọc RFID
individuals = np.random.rand(num_individuals, 2) * [grid_x, grid_y]
num_iterations = 200  # Số vòng lặp

rfid_radius = 3.69 # Bán kính vùng phủ sóng của đầu đọc

# Hàm để kiểm tra khoảng cách giữa hai điểm
def is_valid_distance(new_point, points, min_distance):
    for point in points:
        if np.linalg.norm(new_point - point) < min_distance:
            return False
    return True

# Tạo ngẫu nhiên vị trí ban đầu của các đầu đọc RFID với điều kiện khoảng cách
rfid_readers = np.zeros((num_rfid_readers, 2))
count = 0
while count < num_rfid_readers:
    new_reader = np.random.rand(2) * [grid_x, grid_y]
    if is_valid_distance(new_reader, rfid_readers[:count], rfid_radius):  # Kiểm tra khoảng cách với các đầu đọc đã thêm
        rfid_readers[count] = new_reader
        count += 1

rfid_readers = np.array(rfid_readers)

# Vận tốc ban đầu của các đầu đọc RFID
velocities = np.random.rand(num_rfid_readers, 2) * 0.1  # Tốc độ nhỏ

# Khởi tạo biểu đồ
fig, ax = plt.subplots()
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_xlim(0, grid_x)  
ax.set_ylim(0, grid_y)  
ax.set_aspect('equal', 'box')

# Vẽ các cá thể
ax.scatter(individuals[:, 0], individuals[:, 1], color='blue', label='Individuals')

# Các danh sách để theo dõi các điểm cần vẽ
scatter_rfid = ax.scatter(rfid_readers[:, 0], rfid_readers[:, 1], color='red', label='RFID Readers')
circles = [plt.Circle((x, y), rfid_radius, color='red', fill=True, alpha=0.2, linestyle='--') for x, y in rfid_readers]
for circle in circles:
    ax.add_artist(circle)

# Hàm tính độ nhiễu của một cá thể (sinh viên)
def noiseLevel_Individual(individual):
    level = 0
    for reader in rfid_readers:
        distance = np.linalg.norm(individual - reader)
        if distance < rfid_radius:
            level+=1
    return level
# Hàm kiểm tra một cá thể (sinh viên) có được bao phủ hay chưa?
def IsIndividualCover(individual):
    if noiseLevel_Individual(individual) == 0:
        return False
    else: return True
# Hàm đếm các cá thể chưa được bao phủ
def CountIndividualNotCover():
    count = 0
    for individual in individuals:
        if not IsIndividualCover(individual):
            count+=1
    return count

# Hàm mục tiêu: Đếm số cá thể không được bao phủ bởi bất kỳ đầu đọc nào
def objective_function(rfid_readers):
    total_tags = len(individuals)
    covered_count = 0

    # Duyệt qua từng cá thể và kiểm tra nếu chúng nằm trong bán kính của bất kỳ đầu đọc nào
    for individual in individuals:
        if any(np.linalg.norm(individual - reader) <= rfid_radius for reader in rfid_readers):
            covered_count += 1

    # Tính tỷ lệ phủ sóng
    coverage_ratio = covered_count / total_tags
    return coverage_ratio

# Lưu vị trí tốt nhất của các đầu đọc
pbest_positions = np.copy(rfid_readers)
pbest_scores = np.full(num_rfid_readers, np.inf)  # Số lượng cá thể không được bao phủ (càng thấp càng tốt)

gbest_position = rfid_readers[0]  # Khởi tạo với một giá trị hợp lệ từ vị trí đầu đọc ban đầu
gbest_score = np.inf 
max_no_change_iterations = 5  # Số lần không thay đổi tối đa trước khi dừng
no_change_counter = 0         # Bộ đếm cho số lần không thay đổi liên tiếp
previous_gbest_score = gbest_score  # Biến lưu giá trị của gbest từ lần chạy trước
def update(frame):
    global rfid_readers, velocities, gbest_position, gbest_score, previous_gbest_score, no_change_counter, max_no_change_iterations
    # Đánh giá hàm mục tiêu cho toàn bộ quần thể
    coverage_ratio = objective_function(rfid_readers)
    # In ra tỷ lệ bao phủ (%) và giá trị bao phủ hiện tại
    print(f"Coverage Ratio = {coverage_ratio * 100:.2f}% ({coverage_ratio:.4f})")
    # Kiểm tra nếu tất cả các thẻ đã được bao phủ
    if coverage_ratio == 1.0:
        print(f"All tags are covered at iteration. Stopping the optimization.")
        return
    for i in range(num_rfid_readers):
         # Đánh giá hàm mục tiêu cho từng cá thể, chỉ đánh giá theo vị trí của đầu đọc i
        current_score = objective_function(np.array([rfid_readers[i]]))

        # Cập nhật vị trí tốt nhất của cá thể (pbest)
        if current_score > pbest_scores[i]:  # Tìm max coverage ratio (càng nhiều cá thể được phủ càng tốt)
            pbest_scores[i] = current_score
            pbest_positions[i] = np.copy(rfid_readers[i])

        # Cập nhật vị trí tốt nhất của toàn bộ quần thể (gbest)
        if current_score > gbest_score:  # Nếu điểm số hiện tại của cá thể i tốt hơn gbest
            gbest_score = current_score
            gbest_position = np.copy(rfid_readers[i])  # Cập nhật vị trí tốt nhất toàn quần thể
   # Kiểm tra điều kiện dừng
    if gbest_score == previous_gbest_score:
        no_change_counter += 1  # Tăng bộ đếm nếu gbest_score không thay đổi
    else:
        no_change_counter = 0   # Reset bộ đếm nếu có thay đổi
        previous_gbest_score = gbest_score  # Cập nhật gbest_score mới nhất

    # Dừng vòng lặp nếu qua 5 lần không có sự thay đổi
    if no_change_counter >= max_no_change_iterations:
        print(f"Stopping early after iterations due to no change in objective function.")

    # Cập nhật vận tốc và vị trí của các đầu đọc RFID
    for i in range(num_rfid_readers):
        # Cập nhật vận tốc
        inertia = 0.5  # Hệ số quán tính
        cognitive = 0.8  # Hệ số học hỏi cá nhân
        social = 0.9  # Hệ số học hỏi xã hội

        r1, r2 = np.random.rand(2)
        velocities[i] = (inertia * velocities[i] +
                         cognitive * r1 * (pbest_positions[i] - rfid_readers[i]) +
                         social * r2 * (gbest_position - rfid_readers[i]))

        # Cập nhật vị trí
        new_position = rfid_readers[i] + velocities[i]

        # Giới hạn vị trí trong vùng làm việc
        new_position[0] = np.clip(new_position[0], 0, grid_x)
        new_position[1] = np.clip(new_position[1], 0, grid_y)

        # Kiểm tra khoảng cách với các đầu đọc khác
        #if is_valid_distance(new_position, np.delete(rfid_readers, i, axis=0), rfid_radius):
        rfid_readers[i] = new_position
        print(f"Còn {CountIndividualNotCover()} sinh viên chưa được bao phủ")
        # In ra vị trí trước và sau khi cập nhật
        print(f"RFID Reader {i+1} cập nhật vị trí từ {rfid_readers[i]} đến {new_position}")
    # Cập nhật vị trí các đầu đọc trên biểu đồ
    scatter_rfid.set_offsets(rfid_readers)

    # Cập nhật các vòng tròn
    for j, circle in enumerate(circles):
        circle.center = (rfid_readers[j, 0], rfid_readers[j, 1])

    return scatter_rfid, *circles

# Tạo hoạt hình
ani = FuncAnimation(fig, update, frames=num_iterations, repeat=False, blit=False, interval=500)

#plt.legend()
plt.show()