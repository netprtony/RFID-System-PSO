import numpy as np

RFID_RADIUS = 3.69
def calculate_inertia_weight(w_max, w_min, iter, iter_max): #(15)
    return w_max - ((w_max - w_min) / iter_max) * iter

def calculate_covered_tags(readers, tags, rfid_radius=RFID_RADIUS):#f1(4) tính toán phần trăm độ bao phủ
    total_tag = len(tags)
    covered_tags = 0
    for tag in tags:
        if any(np.linalg.norm(tag.position - reader.position) <= rfid_radius and reader.active for reader in readers):
            covered_tags +=1
    COV = (covered_tags / total_tag) * 100
    return COV

def countReaderActive(readers):
    count = 0
    for reader in readers:
        if reader.active:
            count += 1
    return count


def calculate_interference_basic(readers, tags, rfid_radius=RFID_RADIUS): #IFT (5) mức độ nhiễu
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
        antennas_covering_tag = sum(1 for reader in readers if np.linalg.norm(tag.position - reader.position) <= rfid_radius and reader.active)
        if antennas_covering_tag > 1:  # Nếu có hơn 1 ăng-ten phủ sóng thẻ
            ITF += (antennas_covering_tag - 1)  # Mỗi ăng-ten dư gây nhiễu
    ITF = (ITF / len(tags)) * 100  # Chuyển thành phần trăm
    return ITF

def fitness_function_basic(COV, ITF, tags, w1, w2): # Hàm fitness (7)
    """
    Tính toán hàm mục tiêu đơn giản dựa trên độ phủ và nhiễu.
    
    Tham số:
    COV: Phần trăm độ phủ của mạng
    ITF: Giá trị nhiễu của mạng (số lượng đầu đọc gây nhiễu)
    w1, w2: Trọng số cho các thành phần độ phủ, nhiễu và hình phạt chồng lấn.
    Trả về:
    Giá trị hàm mục tiêu
    """
    fitness = COV/100 * w1 + (len(tags)/(ITF+len(tags))) * w2 # Tổng nhiễu (6)
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

