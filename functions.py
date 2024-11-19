import numpy as np
from colorama import Fore, Style, init
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.cluster import KMeans
from classes import Readers
from classes import  GRID_X, GRID_Y
from utils import calculate_covered_tags, calculate_interference_basic,countReaderActive, fitness_function_basic, calculate_load_balance, RFID_RADIUS
UPDATE_INTERVAL = 500
COVER_THRESHOLD = 0.95 # Ngưỡng bao phủ
DIM = 2
EXCLUSION_FORCE = 0.2 # Hệ số lực đẩy
ATTRACTION_FORCE = 1  # Hệ số lực hút

# Định nghĩa hằng số cho lực
REPULSION_FORCE_COEF = 1.0  # Hệ số lực đẩy
ATTRACTION_FORCE_COEF = 0.5  # Hệ số lực hút
IDEAL_DISTANCE = 10.0 # Khoảng cách lý tưởng giữa các đầu đọc

def initialize_readers_with_kmeans(TAGS, n_readers):
    # Lấy vị trí của các tag
    tag_positions = np.array([tag.position for tag in TAGS])
    
    # Sử dụng thuật toán k-means để tìm các cluster center
    kmeans = KMeans(n_clusters=n_readers, random_state=0).fit(tag_positions)
    
    # Đặt các reader tại các cluster center
    reader_positions = kmeans.cluster_centers_
    
    # Khởi tạo các reader với vị trí mới
    READERS = [Readers(position=pos) for pos in reader_positions]
    
    return READERS

# Hàm Selection Mechanism for the Number of Readers
def selection_mechanism(tags, initial_num_readers):
    readers = []  # Danh sách để lưu các đầu đọc
    num_readers = initial_num_readers  # Số lượng đầu đọc ban đầu

    while True:
        # Khởi tạo các đầu đọc với vị trí ngẫu nhiên
        readers = initialize_readers_with_kmeans(tags, num_readers)
        
        # Kiểm tra độ bao phủ của tất cả các thẻ
        for tag in tags:
            tag.covered = False  # Đặt trạng thái chưa được bao phủ

            # Kiểm tra xem thẻ có nằm trong vùng phủ sóng của bất kỳ đầu đọc nào không
            for reader in readers:
                if np.linalg.norm(tag.position - reader.position) <= RFID_RADIUS:
                    tag.covered = True
                    break

        # Tính tỷ lệ thẻ được bao phủ
        coverage_ratio = calculate_covered_tags(readers, tags, RFID_RADIUS) / 100
        print(Fore.MAGENTA + Style.BRIGHT + f"Độ bao phủ: {coverage_ratio}")

        # Kiểm tra nếu tỷ lệ bao phủ đạt yêu cầu, thoát khỏi vòng lặp
        if coverage_ratio >= COVER_THRESHOLD:
            break

        # Nếu không đạt, tăng số lượng đầu đọc và lặp lại
        num_readers += 1
        #BieuDoReader(readers, tags)

    return readers  # Trả về danh sách đầu đọc đã được chọn



def adjust_readers_location_by_virtual_force(readers, tags, max_no_change_iterations=5):
    no_change_iterations = 0
    last_coverage = calculate_covered_tags(readers, tags, RFID_RADIUS) / 100

    while True:
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
            reader.position += total_exclusion_force + total_attraction_force
        
            # Giới hạn vị trí trong không gian làm việc
            reader.position[0] = np.clip(reader.position[0], RFID_RADIUS/2, GRID_X - RFID_RADIUS/2)
            reader.position[1] = np.clip(reader.position[1], RFID_RADIUS/2, GRID_Y - RFID_RADIUS/2)

        # Tính tỷ lệ thẻ được bao phủ
        coverage_ratio = calculate_covered_tags(readers, tags, RFID_RADIUS) / 100
        print(Fore.LIGHTYELLOW_EX + f"Độ bao phủ: {coverage_ratio}")
        # Kiểm tra nếu tỷ lệ bao phủ không thay đổi
        if coverage_ratio == last_coverage:
            no_change_iterations += 1
        else:
            no_change_iterations = 0

        if no_change_iterations >= max_no_change_iterations:
            print(f"Stopping early after {no_change_iterations} iterations due to no change in coverage.")
            break
        # if coverage_ratio == 80:
        #     break

        last_coverage = coverage_ratio  
        #BieuDoReader(readers, tags)
    print(f"Final coverage: {last_coverage}")
    return readers

def BieuDotags(READERS, TAGS):
    fig, ax = plt.subplots()
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_xlim(0, GRID_X)
    ax.set_ylim(0, GRID_Y)
    ax.set_aspect('equal', 'box')
    
    tag_positions = np.array([tag.position for tag in TAGS])
    reader_positions = np.array([reader.position for reader in READERS])
    
    scatter_tag = ax.scatter(tag_positions[:, 0], tag_positions[:, 1], color='blue', label='TAGS', s=10, marker='x')
    ax.scatter(reader_positions[:, 0], reader_positions[:, 1], color='red', label='Readers', marker='^')
    
    circles = [plt.Circle((x, y), RFID_RADIUS, color='red', fill=True, alpha=0.2, linestyle='--') for x, y in reader_positions]
    for circle in circles:
        ax.add_artist(circle)
    
    count_text = ax.text(0.02, 1.05, 'Tags in range: 0', transform=ax.transAxes, fontsize=12, verticalalignment='top')

    def update_tag(frame):
        for tag in TAGS:
            tag.update_position()
            tag.covered = any(np.linalg.norm(tag.position - reader.position)<= RFID_RADIUS for reader in READERS)
        
        tag_positions = np.array([tag.position for tag in TAGS])
        tag_positions = np.atleast_2d(tag_positions)

        colors = ['green' if tag.covered else 'blue' for tag in TAGS]

        scatter_tag.set_offsets(tag_positions)
        scatter_tag.set_color(colors)

        count_in_range = sum(tag.covered for tag in TAGS)
        count_text.set_text(f'Tags in range: {count_in_range}')
        print(f"{count_in_range} tags in range")

        return scatter_tag

    ani = animation.FuncAnimation(fig, update_tag, frames=999999, interval=UPDATE_INTERVAL, blit=False, repeat=False)
    plt.show()

def BieuDoReader(readers, tags):
    """
    Vẽ biểu đồ vị trí các đầu đọc và các thẻ.

    Parameters:
    - readers: Danh sách các đối tượng reader
    - tags: Danh sách các đối tượng tag
    """
    fig, ax = plt.subplots(figsize=(10, 8)) 
    ax.set_xlim(0, GRID_X)
    ax.set_ylim(0, GRID_Y)
    ax.set_aspect('equal', 'box')

    # Lấy vị trí của các tag
    tag_positions = np.array([tag.position for tag in tags])
    tag_colors = ['green' if any(np.linalg.norm(tag.position - reader.position) <= RFID_RADIUS for reader in readers if reader.active) else 'blue' for tag in tags]
    ax.scatter(tag_positions[:, 0], tag_positions[:, 1], color=tag_colors, label='Tags', s=20, marker='x')

    # Thêm thông tin về độ phủ, nhiễu và số lượng đầu đọc
    ax.text(0.02, 1.05, f'COV: {calculate_covered_tags(readers, tags, RFID_RADIUS):.2f}%', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    ax.text(0.45, 1.05, f'ITF: {calculate_interference_basic(readers, tags, RFID_RADIUS):.2f}%', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    ax.text(0.75, 1.05, f'Reader: {countReaderActive(readers)}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    
    # Lấy vị trí của các reader có active = True
    active_reader_positions = np.array([reader.position for reader in readers])
    if active_reader_positions.size > 0:
        # Đảm bảo rằng active_reader_positions là một mảng 2 chiều
        active_reader_positions = active_reader_positions.reshape(-1, 2)
        ax.scatter(active_reader_positions[:, 0], active_reader_positions[:, 1], color='red', label='Readers', marker='^')

    # Vẽ các vòng tròn phạm vi phủ sóng của các reader có active = True
    circles = [plt.Circle((x, y), RFID_RADIUS, color='red', fill=True, alpha=0.2, linestyle='--') for x, y in active_reader_positions]
    for circle in circles:
        ax.add_artist(circle)

    ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)
    plt.show()
    
def mainOptimization(tags, readers, sspso):
    while True:
        readers = sspso.optimize(tags, RFID_RADIUS)
        BieuDoReader(readers, tags)
        readers = adjust_readers_location_by_virtual_force(readers, tags)
        BieuDoReader(readers, tags)
        readers = Redundant_Reader_Elimination(readers, tags)
        BieuDoReader(readers, tags)
        if calculate_covered_tags(readers, tags, RFID_RADIUS) >= 100:
            print("Optimization completed.")
            BieuDoReader(readers, tags)
            break
    

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
    - w1, w2, w3: Trọng số cho các thành phần độ phủ, nhiễu và cân bằng tải

    Returns:
    - readers: Danh sách các đối tượng reader sau khi loại bỏ các đầu đọc dư thừa
    """
    initial_coverage = calculate_covered_tags(readers, tags)
    initial_interference = calculate_interference_basic(readers, tags) 
    initial_LoadBalance = calculate_load_balance(readers, tags)
    initial_fitness = fitness_function_basic(initial_coverage, initial_interference, initial_LoadBalance, tags, w1, w2, w3)
    readers_to_remove = []

    for i, reader in enumerate(readers):
        if reader.active:
            # Tạm thời tắt đầu đọc
            reader.active = False

            # Tính toán lại các giá trị
            new_coverage = calculate_covered_tags(readers, tags)
            new_interference = calculate_interference_basic(readers, tags)
            new_LoadBalance = calculate_load_balance(readers, tags)
            new_fitness = fitness_function_basic(new_coverage, new_interference, new_LoadBalance, tags, w1, w2, w3)
           
            # Kiểm tra các tiêu chí
            coverage_reduction = initial_coverage - new_coverage
            interference_reduction = initial_interference - new_interference
            fitness_reduction = initial_fitness - new_fitness

            if (coverage_reduction < coverage_threshold and new_coverage >= 80 and
                interference_reduction < interference_threshold and
                fitness_reduction < fitness_threshold):
                print(Fore.GREEN + f"Đã loại bỏ thành công đầu đọc {i}.")
                readers_to_remove.append(reader)
            else:
                # Khôi phục đầu đọc nếu không thỏa mãn các tiêu chí
                reader.active = True

    for reader in readers_to_remove:
        readers.remove(reader)

    return readers

def extra_mechanism(readers, uncovered_tags, initial_num_readers, coverage_threshold=1.0):
    """
    Thêm các readers sử dụng thuật toán k-means đối với những tag chưa được bao phủ
    cho đến khi đạt ngưỡng bao phủ là 1.

    Parameters:
    - readers: Danh sách các đối tượng reader hiện tại
    - uncovered_tags: Danh sách các đối tượng tag chưa được bao phủ
    - initial_num_readers: Số lượng đầu đọc hiện tại
    - coverage_threshold: Ngưỡng bao phủ (mặc định là 1.0)

    Returns:
    - readers: Danh sách các đối tượng reader đã được bổ sung
    """
    num_readers = initial_num_readers  # Số lượng đầu đọc ban đầu

    while True:
        # Sử dụng k-means để tìm vị trí mới cho các đầu đọc
        kmeans = KMeans(n_clusters=num_readers, random_state=0).fit([tag.position for tag in uncovered_tags])
        new_reader_positions = kmeans.cluster_centers_

        # Thêm các đầu đọc mới vào danh sách readers
        for pos in new_reader_positions:
            new_reader = Readers(position=pos)
            readers.append(new_reader)
            print(f"Thêm đầu đọc mới tại vị trí {new_reader.position}.")

        # Kiểm tra độ bao phủ của tất cả các thẻ
        for tag in uncovered_tags:
            tag.covered = False  # Đặt trạng thái chưa được bao phủ

            # Kiểm tra xem thẻ có nằm trong vùng phủ sóng của bất kỳ đầu đọc nào không
            for reader in readers:
                if np.linalg.norm(tag.position - reader.position) <= RFID_RADIUS:
                    tag.covered = True
                    break

        # Tính tỷ lệ thẻ được bao phủ
        coverage_ratio = calculate_covered_tags(readers, uncovered_tags, RFID_RADIUS) / 100
        print(f"Độ bao phủ: {coverage_ratio}")
        BieuDoReader(readers, uncovered_tags)
        # Kiểm tra nếu tỷ lệ bao phủ đạt yêu cầu, thoát khỏi vòng lặp
        if coverage_ratio >= coverage_threshold:
            break

        # Nếu không đạt, tăng số lượng đầu đọc và lặp lại
        num_readers += 1

    return readers

def compute_distance(reader1, reader2):
    """Tính khoảng cách Euclidean giữa hai đầu đọc."""
    return np.linalg.norm(reader1.position - reader2.position)

def compute_repulsion_force(reader1, reader2):
    """Tính lực đẩy giữa hai đầu đọc nếu khoảng cách nhỏ hơn khoảng cách lý tưởng."""
    distance = compute_distance(reader1, reader2)
    if distance < IDEAL_DISTANCE:
        # Lực đẩy tỷ lệ nghịch với khoảng cách
        force_magnitude = REPULSION_FORCE_COEF * (1 / distance - 1 / IDEAL_DISTANCE)
        direction = (reader1.position - reader2.position) / distance
        return force_magnitude * direction
    return np.array([0.0, 0.0])

def compute_attraction_force(reader1, reader2):
    """Tính lực hút giữa hai đầu đọc nếu khoảng cách lớn hơn khoảng cách lý tưởng."""
    distance = compute_distance(reader1, reader2)
    if distance > IDEAL_DISTANCE:
        # Lực hút tỷ lệ thuận với khoảng cách chênh lệch
        force_magnitude = ATTRACTION_FORCE_COEF * (distance - IDEAL_DISTANCE)
        direction = (reader2.position - reader1.position) / distance
        return force_magnitude * direction
    return np.array([0.0, 0.0])

def apply_virtual_force_algorithm(readers):
    """Hàm chính của VFA để tối ưu hóa vị trí của các đầu đọc."""
    for i, reader in enumerate(readers):
        # Tính tổng lực tác động lên mỗi đầu đọc
        total_force = np.array([0.0, 0.0])
        
        for j, other_reader in enumerate(readers):
            if i != j and other_reader.active:
                # Tính lực đẩy giữa các đầu đọc
                repulsion_force = compute_repulsion_force(reader, other_reader)
                # Tính lực hút giữa các đầu đọc
                attraction_force = compute_attraction_force(reader, other_reader)
                # Tổng hợp lực
                total_force += repulsion_force + attraction_force

        # Cập nhật vận tốc và vị trí của đầu đọc dựa trên tổng lực
        reader.velocity += total_force
        # Giới hạn vận tốc để không vượt quá vận tốc tối đa
        if np.linalg.norm(reader.velocity) > reader.max_velocity:
            reader.velocity = (reader.velocity / np.linalg.norm(reader.velocity)) * reader.max_velocity
        
        # Cập nhật vị trí của đầu đọc
        reader.position += reader.velocity

    # Trả về danh sách đầu đọc với vị trí đã được tối ưu hóa
    return readers