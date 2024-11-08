import numpy as np

RFID_RADIUS = 3.69
def calculate_inertia_weight(w_max, w_min, iter, iter_max):
    return w_max - ((w_max - w_min) / iter_max) * iter

def calculate_covered_tags(readers, tags, rfid_radius=RFID_RADIUS):
    total_tag = len(tags)
    covered_tags = 0
    for tag in tags:
        if any(np.linalg.norm(tag.position - reader.position) <= rfid_radius for reader in readers if reader.active):
            covered_tags +=1
    COV = (covered_tags / total_tag) * 100
    return COV

def countReaderActive(readers):
    count = 0
    for reader in readers:
        if reader.active:
            count += 1
    return count


def calculate_interference_basic(readers, tags, rfid_radius):
    """
    Tính toán nhiễu đơn giản, chỉ kiểm tra số lượng ăng-ten phủ sóng cho mỗi thẻ.

    Tham số:
    readers: Danh sách các ăng-ten (đối tượng Readers)
    tags: Danh sách các thẻ RFID (đối tượng tags)
    rfid_radius: Bán kính phủ sóng của mỗi ăng-ten

    Trả về:
    ITF: Giá trị nhiễu tổng cộng (số thẻ bị phủ bởi hơn 1 ăng-ten)
    """
    ITF = 0  # Tổng giá trị nhiễu
    for tag in tags:  # Duyệt qua tất cả các thẻ
        antennas_covering_tag = sum(1 for reader in readers 
                                        if np.linalg.norm(tag.position - reader.position) <= rfid_radius and reader.active)
        if antennas_covering_tag > 1:  # Nếu có hơn 1 ăng-ten phủ sóng thẻ
            ITF += (antennas_covering_tag - 1)  # Mỗi ăng-ten dư gây nhiễu
    return ITF

def fitness_function_basic(COV, ITF, LDB, w1, w2, w3, tags): # 23.optimizing_radio
    """
    Tính toán hàm mục tiêu đơn giản dựa trên độ phủ và nhiễu.
    
    Tham số:
    COV: Phần trăm độ phủ của mạng
    ITF: Giá trị nhiễu của mạng (số lượng ăng-ten gây nhiễu)
    OLP: Giá trị hình phạt chồng lấn giữa các đầu đọc.
    w1, w2, w3: Trọng số cho các thành phần độ phủ, nhiễu và hình phạt chồng lấn.
    Trả về:
    Giá trị hàm mục tiêu
    """
    fitness = COV/100 * w1 + (len(tags)/(len(tags)+ITF)) * w2 + LDB* w3
    return fitness






def constrain_velocity(velocity, upper_limit, lower_limit):
    """
    Giới hạn tốc độ của hạt trong phạm vi đã cho.

    Parameters:
    - velocity: Vận tốc hiện tại của hạt (danh sách hoặc mảng chứa các giá trị vận tốc ở từng chiều)
    - upper_limit: Giới hạn trên của không gian tìm kiếm (danh sách hoặc mảng cùng độ dài với velocity hoặc số nguyên)
    - lower_limit: Giới hạn dưới của không gian tìm kiếm (danh sách hoặc mảng cùng độ dài với velocity hoặc số nguyên)

    Returns:
    - velocity: Vận tốc đã được giới hạn
    """
    # Nếu upper_limit và lower_limit là số nguyên, chuyển đổi chúng thành danh sách
    if isinstance(upper_limit, (int, float)):
        upper_limit = [upper_limit] * len(velocity)
    if isinstance(lower_limit, (int, float)):
        lower_limit = [lower_limit] * len(velocity)

    # Tính giá trị giới hạn tốc độ delta
    delta = [(u - l) / 2 for u, l in zip(upper_limit, lower_limit)]

    # Giới hạn vận tốc
    for i in range(len(velocity)):
        if velocity[i] > delta[i]:
            velocity[i] = delta[i]
        elif velocity[i] < -delta[i]:
            velocity[i] = -delta[i]

    return velocity

# Hàm tính cân bằng tải
def calculate_load_balance(readers, tags):
    # Đếm số lượng thẻ được bao phủ bởi từng đầu đọc
    tag_counts = np.zeros(len(readers))
    for tag in tags:
        for i, reader in enumerate(readers):
            distance = np.linalg.norm(tag.position - reader.position)
            if distance <= RFID_RADIUS:
                tag_counts[i] += 1
                break  # Một thẻ chỉ cần được bao phủ bởi một đầu đọc

    # Tính độ lệch chuẩn của số lượng thẻ để đánh giá mức độ cân bằng tải
    mean_tags = np.mean(tag_counts)
    variance = np.var(tag_counts)
    load_balance = 1 / (1 + variance)  # Công thức chuẩn hóa để load balance nằm trong khoảng (0, 1)

    return load_balance