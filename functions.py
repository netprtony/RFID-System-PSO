import numpy as np
from colorama import Fore, Style, init
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.cluster import KMeans
from classes import Readers
from classes import  GRID_X, GRID_Y
from utils import calculate_covered_tags, calculate_interference_basic, fitness_function_basic, RFID_RADIUS
UPDATE_INTERVAL = 500
DIM = 2
EXCLUSION_FORCE = 0.2 # Hệ số lực đẩy
ATTRACTION_FORCE = 1  # Hệ số lực hút

# Định nghĩa hằng số cho lực
REPULSION_FORCE_COEF = 1.0  # Hệ số lực đẩy
ATTRACTION_FORCE_COEF = 0.5  # Hệ số lực hút
IDEAL_DISTANCE = 10.0 # Khoảng cách lý tưởng giữa các đầu đọc
GRID_SIZE = 1.6
def initialize_readers_with_kmeans(tags, num_readers):
    """Khởi tạo vị trí đầu đọc sử dụng thuật toán KMeans."""
    positions = np.array([tag.position for tag in tags])
    kmeans = KMeans(n_clusters=num_readers, random_state=42).fit(positions)
    return [Readers(position=center) for center in kmeans.cluster_centers_]
    
def create_grid(grid_size, grid_x, grid_y):
    """Tạo lưới các điểm trong không gian hoạt động."""
    x_coords = np.arange(0, grid_x, grid_size)
    y_coords = np.arange(0, grid_y, grid_size)
    grid_points = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)
    return grid_points

def snap_to_grid(position, grid_points):
    """Đưa vị trí về điểm trên lưới gần nhất."""
    distances = np.linalg.norm(grid_points - position, axis=1)
    nearest_point = grid_points[np.argmin(distances)]
    return nearest_point
def Reader_GRID(readers, GRID_SIZE):
    grid_points = create_grid(GRID_SIZE, GRID_X, GRID_Y)
   #Điều chỉnh vị trí các đầu đọc về mắt lưới gần nhất
    for reader in readers:
        reader.position = snap_to_grid(reader.position, grid_points)
    return readers

def selection_mechanism(tags, initial_num_readers, COVER_THRESHOLD):
    """Hàm chọn đầu đọc dựa trên KMeans và điều chỉnh vị trí về mắt lưới."""
    readers = []  # Danh sách đầu đọc
    num_readers = initial_num_readers  # Số lượng đầu đọc ban đầu
    tracking = []
    #Tạo lưới
    #grid_points = create_grid(GRID_SIZE, grid_x, grid_y)
    
    while True:
        tracking.append([calculate_covered_tags(readers, tags), len(readers)])
        # Khởi tạo các đầu đọc với vị trí cụm từ KMeans
        kmeans_readers = initialize_readers_with_kmeans(tags, num_readers)

        # #Điều chỉnh vị trí các đầu đọc về mắt lưới gần nhất
        # for reader in kmeans_readers:
        #     reader.position = snap_to_grid(reader.position, grid_points)

        # Đặt trạng thái bao phủ của tất cả các thẻ
        for tag in tags:
            tag.covered = False
            # Kiểm tra nếu thẻ nằm trong vùng phủ sóng của bất kỳ đầu đọc nào
            for reader in kmeans_readers:
                if np.linalg.norm(tag.position - reader.position) <= RFID_RADIUS:
                    tag.covered = True
                    break

        # Tính tỷ lệ thẻ được bao phủ
        coverage_ratio = calculate_covered_tags(kmeans_readers, tags, RFID_RADIUS) / 100
        print(f"Độ bao phủ: {coverage_ratio:.2%}")
        #BieuDoReader(kmeans_readers, tags)
        # Nếu tỷ lệ bao phủ đạt yêu cầu, thoát khỏi vòng lặp
        if coverage_ratio >= COVER_THRESHOLD:
            readers = kmeans_readers
            break

        # Nếu không đạt, tăng số lượng đầu đọc và lặp lại
        num_readers += 1

    return readers, tracking  # Trả về danh sách đầu đọc đã được chọn


def BieuDoSoSanh(tracking3_2, tracking1_6, tracking0_8):
    # Tách dữ liệu từ tracking
    COV3_2 = [item[0] for item in tracking3_2]
    readerNum3_2 = [item[1] for item in tracking3_2]
    COV1_6 = [item[0] for item in tracking1_6]
    readerNum1_6 = [item[1] for item in tracking1_6]
    COV0_8 = [item[0] for item in tracking0_8]
    readerNum0_8 = [item[1] for item in tracking0_8]
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(readerNum3_2, COV3_2, marker='^', label='Grid 3.2', color='gray', linewidth=2)
    plt.plot(readerNum1_6, COV1_6, marker='^', label='Grid 1.6', color='red', linewidth=2)
    plt.plot(readerNum0_8, COV0_8, marker='^', label='Grid 0.8', color='blue', linewidth=2)
    # Labels and legend
    plt.xlabel('Số lượng đầu đọc', fontsize=12)
    plt.ylabel('Độ bao phủ', fontsize=12)
    plt.title('So sánh độ bao phủ theo số lượng đầu đọc', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    # Show the plot
    plt.show()


def adjust_readers_location_by_virtual_force(readers, tags, max_no_change_iterations=50):
    no_change_iterations = 0
    best_fitness = -float('inf')
    best_positions = [reader.position.copy() for reader in readers]

    while no_change_iterations < max_no_change_iterations:
        for reader in readers:
            # Lực tổng hợp được khởi tạo
            total_exclusion_force = np.array([0.0, 0.0])
            total_attraction_force = np.array([0.0, 0.0])

            # 1. Lực đẩy (Exclusion Operator)
            for other_reader in readers:
                if other_reader != reader:
                    distance = np.linalg.norm(reader.position - other_reader.position)
                    if distance < 2 * RFID_RADIUS:  # Kiểm tra xem hai đầu đọc có chồng lấn không
                        # Tính toán lực đẩy
                        force_magnitude = EXCLUSION_FORCE * (2 * RFID_RADIUS - distance)
                        direction = (reader.position - other_reader.position) / distance
                        total_exclusion_force += force_magnitude * direction

            # 2. Lực hút (Attraction Operator)
            for tag in tags:
                if not tag.covered:  # Nếu thẻ chưa được bao phủ
                    distance = np.linalg.norm(reader.position - tag.position)
                    if distance <= RFID_RADIUS:  # Kiểm tra xem thẻ có nằm trong phạm vi hấp dẫn không
                        # Tính toán lực hút
                        force_magnitude = ATTRACTION_FORCE * (RFID_RADIUS - distance)
                        direction = (tag.position - reader.position) / distance
                        total_attraction_force += force_magnitude * direction

            # 3. Cập nhật vị trí đầu đọc dựa trên lực tổng hợp
            total_force = total_exclusion_force + total_attraction_force
            reader.position += total_force
        
            # Giới hạn vị trí trong không gian làm việc
            reader.position[0] = np.clip(reader.position[0], 0, GRID_X )
            reader.position[1] = np.clip(reader.position[1], 0, GRID_Y )

        # Tính tỷ lệ thẻ được bao phủ
        # Tính toán giá trị fitness
        COV = calculate_covered_tags(readers, tags)
        ITF = calculate_interference_basic(readers, tags, RFID_RADIUS)
        fitness = fitness_function_basic(COV, ITF, tags, 0.8, 0.2)
        print(Fore.LIGHTYELLOW_EX + f"Fitness: {fitness}")
        # Kiểm tra nếu giá trị fitness tốt hơn
        if fitness > best_fitness:
            best_fitness = fitness
            best_positions = [reader.position.copy() for reader in readers]
            no_change_iterations = 0
        else:
            no_change_iterations += 1

        # Khôi phục vị trí tốt nhất
        for reader, best_position in zip(readers, best_positions):
            reader.position = best_position
    print(f"Final fitness: {best_fitness}")
    return readers



def BieuDoReader(readers, tags, title, GRID_SIZE):
    """
    Vẽ biểu đồ vị trí các đầu đọc và các thẻ với mặt lưới.
 
    Parameters:
    - readers: Danh sách các đối tượng reader.
    - tags: Danh sách các đối tượng tag.
    """
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_xlim(0, GRID_X)
    ax.set_ylim(0, GRID_Y)
    ax.set_aspect('equal', 'box')

    # Hiển thị mặt lưới
    ax.set_xticks(np.arange(0, GRID_X + 1, GRID_SIZE))
    ax.set_yticks(np.arange(0, GRID_Y + 1, GRID_SIZE))
    ax.grid(color='gray', linestyle='--', linewidth=0.5)

    # Vẽ các thẻ
    tag_positions = np.array([tag.position for tag in tags])
    tag_colors = ['red' if not any(np.linalg.norm(tag.position - reader.position) <= RFID_RADIUS for reader in readers if reader.active) else 'green' for tag in tags]
    ax.scatter(tag_positions[:, 0], tag_positions[:, 1], color=tag_colors, label='Tags', s=20, marker='x')

    # Vẽ các đầu đọc
    active_reader_positions = np.array([reader.position for reader in readers if reader.active])
    ax.scatter(active_reader_positions[:, 0], active_reader_positions[:, 1], color='blue', label='Readers', marker='^')

    # Vẽ các vòng tròn phạm vi phủ sóng
    for reader in readers:
        if reader.active:
            circle = plt.Circle(reader.position, RFID_RADIUS, color='black', fill=False, linestyle='-', linewidth=1, alpha=0.5)
            ax.add_artist(circle)

    # Thêm thông tin độ phủ sóng, độ nhiễu và số lượng đầu đọc
    coverage_tag = calculate_covered_tags(readers, tags, RFID_RADIUS)/ 100 * len(tags)  # Phải định nghĩa hàm calculate_covered_tags
    interference = calculate_interference_basic(readers, tags, RFID_RADIUS)  # Phải định nghĩa hàm calculate_interference_basic
    active_reader_count = sum(reader.active for reader in readers)

    # Di chuyển các thông tin lên đầu biểu đồ
    fig.text(0.3, 0.92, f"Có {coverage_tag:.0f} thẻ bao phủ trong {len(tags)}", fontsize=12, color="black", ha='left', va='top')
    fig.text(0.5, 0.92, f"Độ nhiễu: {interference:.2f}%", fontsize=12, color="orange", ha='left', va='top')
    fig.text(0.6, 0.92, f"Số lượng đầu đọc: {active_reader_count}", fontsize=12, color="blue", ha='left', va='top')

    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))  # Đưa chú thích biểu đồ ra ngoài phía trên bên phải
    # Đặt tiêu đề cho biểu đồ
    fig.suptitle(title, fontsize=14, ha='left', va='top', fontweight='bold', x=0.01, y=0.99)
    plt.show()

def TongHopBieuDo(readers_list, tags, titles, grid_sizes, grid_shape):
    """
    Hiển thị tổng hợp nhiều biểu đồ con từ hàm BieuDoReader với `GRID_SIZE` khác nhau.

    Parameters:
    - readers_list: Danh sách các danh sách đầu đọc (phân chia theo từng biểu đồ).
    - tags: Danh sách các thẻ (dùng chung cho tất cả biểu đồ).
    - titles: Danh sách tiêu đề cho từng biểu đồ con.
    - grid_sizes: Danh sách GRID_SIZE khác nhau cho từng biểu đồ.
    - grid_shape: Tuple (rows, cols) xác định số hàng và cột trong lưới biểu đồ.
    """
    rows, cols = grid_shape
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))  # Điều chỉnh kích thước phù hợp
    axes = axes.flatten()  # Chuyển mảng 2D của axes thành 1D để duyệt dễ dàng

    for i, (readers, title, grid_size) in enumerate(zip(readers_list, titles, grid_sizes)):
        ax = axes[i]
        ax.set_xlim(0, GRID_X)
        ax.set_ylim(0, GRID_Y)
        ax.set_aspect('equal', 'box')

        # Hiển thị mặt lưới với `grid_size` cụ thể cho từng biểu đồ
        ax.set_xticks(np.arange(0, GRID_X + 1, grid_size))
        ax.set_yticks(np.arange(0, GRID_Y + 1, grid_size))
        ax.grid(color='gray', linestyle='--', linewidth=0.5)

        # Vẽ các thẻ
        tag_positions = np.array([tag.position for tag in tags])
        tag_colors = ['red' if not any(np.linalg.norm(tag.position - reader.position) <= RFID_RADIUS for reader in readers if reader.active) else 'green' for tag in tags]
        ax.scatter(tag_positions[:, 0], tag_positions[:, 1], color=tag_colors, label='Tags', s=20, marker='x')

        # Vẽ các đầu đọc
        active_reader_positions = np.array([reader.position for reader in readers if reader.active])
        ax.scatter(active_reader_positions[:, 0], active_reader_positions[:, 1], color='blue', label='Readers', marker='^')

        # Vẽ các vòng tròn phạm vi phủ sóng
        for reader in readers:
            if reader.active:
                circle = plt.Circle(reader.position, RFID_RADIUS, color='black', fill=False, linestyle='-', linewidth=1, alpha=0.5)
                ax.add_artist(circle)

        # Thêm tiêu đề cho từng biểu đồ con
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')

    # Ẩn các subplot dư thừa nếu số biểu đồ ít hơn tổng số subplot
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def mainOptimization(tags, readers, sspso, GRID_SIZE, tracking):
    readers = sspso.optimize(tags, RFID_RADIUS)
    tracking.append([calculate_covered_tags(readers, tags), len(readers)])
    #BieuDoReader(readers, tags, "Biểu đồ sau khi tối ưu hội tụ", GRID_SIZE)
    readers = adjust_readers_location_by_virtual_force(readers, tags)
    tracking.append([calculate_covered_tags(readers, tags), len(readers)])
    #BieuDoReader(readers, tags, "Biểu đồ sau khi tối ưu hóa bằng lực ảo", GRID_SIZE)
    grid_points = create_grid(GRID_SIZE, GRID_X, GRID_Y)
    tracking.append([calculate_covered_tags(readers, tags), len(readers)])
    for reader in readers:
            reader.position = snap_to_grid(reader.position, grid_points)
    #BieuDoReader(readers, tags, "Biểu đồ sau khi đưa vị trí về mắt lưới", GRID_SIZE)       
    tracking.append([calculate_covered_tags(readers, tags), len(readers)])     
    readers = Redundant_Reader_Elimination(readers, tags)
    #BieuDoReader(readers, tags, "Biểu đồ sau khi loại bỏ đầu đọc dư thừa", GRID_SIZE)
    tracking.append([calculate_covered_tags(readers, tags), len(readers)])
    return readers, tracking
    

def Redundant_Reader_Elimination(readers, tags, coverage_threshold=1, interference_threshold = 10,  fitness_threshold=0.1, w1=0.5, w2=0.3, w3=0.2):
    """
    Loại bỏ các đầu đọc dư thừa dựa trên ba tiêu chí:
    1. Giảm ít hơn 1% tỷ lệ bao phủ và tổng tỷ lệ bao phủ không giảm dưới 90%.
    2. Giảm nhiễu hơn 10%.
    3. Giảm giá trị hàm fitness ít hơn 1%.

    Parameters:
    - readers: Danh sách các đối tượng reader
    - tags: Danh sách các đối tượng tag
    - coverage_threshold: Ngưỡng tỷ lệ bao phủ tối thiểu (mặc định là 90%)
    - interference_threshold: Ngưỡng giảm nhiễu tối thiểu (mặc định là 10%)
    - fitness_threshold: Ngưỡng giảm giá trị hàm fitness tối thiểu (mặc định là 1%)
    - w1, w2: Trọng số cho các thành phần độ phủ và nhiễu

    Returns:
    - readers: Danh sách các đối tượng reader sau khi loại bỏ các đầu đọc dư thừa
    """
    initial_coverage = calculate_covered_tags(readers, tags)
    initial_interference = calculate_interference_basic(readers, tags) 
    #initial_fitness = fitness_function_basic(initial_coverage, initial_interference, tags, w1, w2)
    readers_to_remove = []

    for i, reader in enumerate(readers):
        if reader.active:
            # Tạm thời tắt đầu đọc
            reader.active = False

            # Tính toán lại các giá trị
            new_coverage = calculate_covered_tags(readers, tags)
            new_interference = calculate_interference_basic(readers, tags)
            #new_fitness = fitness_function_basic(new_coverage, new_interference, tags, w1, w2)
           
            # Kiểm tra các tiêu chí
            coverage_reduction = initial_coverage - new_coverage
            interference_reduction = initial_interference - new_interference
            #fitness_reduction = initial_fitness - new_fitness

            if coverage_reduction < coverage_threshold and new_coverage >= 80 and interference_reduction < interference_threshold :#and
                #fitness_reduction < fitness_threshold):
                print(Fore.GREEN + f"Đã loại bỏ thành công đầu đọc {i}.")
                readers_to_remove.append(reader)
            else:
                # Khôi phục đầu đọc nếu không thỏa mãn các tiêu chí
                reader.active = True

    for reader in readers_to_remove:
        readers.remove(reader)

    return readers





