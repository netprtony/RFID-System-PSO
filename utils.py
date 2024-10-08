import numpy as np

RFID_RADIUS = 3.69

def calculate_covered_students(readers, students, rfid_radius=RFID_RADIUS):
    covered_students = 0
    for student in students:
        student_covered = False
        for reader in readers:
            distance = np.linalg.norm(student.position - reader.position)
            if distance <= rfid_radius:
                student_covered = True
                break
        if student_covered:
            covered_students += 1
    return covered_students / len(students)

def calculate_overlap(readers, students, rfid_radius=RFID_RADIUS):
    total_students_covered = set()
    overlapping_students = set()

    for student in students:
        readers_covering_student = 0
        for reader in readers:
            if np.linalg.norm(student.position - reader.position) <= rfid_radius:
                readers_covering_student += 1

        if readers_covering_student > 0:
            total_students_covered.add(student)
        if readers_covering_student > 1:
            overlapping_students.add(student)

    overlap_percentage = len(overlapping_students) / len(total_students_covered) if total_students_covered else 0.0
    return overlap_percentage

def calculate_interference_basic(readers, students, rfid_radius):
    """
    Tính toán nhiễu đơn giản, chỉ kiểm tra số lượng ăng-ten phủ sóng cho mỗi thẻ.

    Tham số:
    readers: Danh sách các ăng-ten (đối tượng Readers)
    students: Danh sách các thẻ RFID (đối tượng Students)
    rfid_radius: Bán kính phủ sóng của mỗi ăng-ten

    Trả về:
    ITF: Giá trị nhiễu tổng cộng (số thẻ bị phủ bởi hơn 1 ăng-ten)
    """
    ITF = 0  # Tổng giá trị nhiễu
    for student in students:  # Duyệt qua tất cả các thẻ
        antennas_covering_student = sum(1 for reader in readers 
                                        if np.linalg.norm(student.position - reader.position) <= rfid_radius)
        if antennas_covering_student > 1:  # Nếu có hơn 1 ăng-ten phủ sóng thẻ
            ITF += (antennas_covering_student - 1)  # Mỗi ăng-ten dư gây nhiễu
    return ITF

def fitness_function_basic(COV, ITF):
    """
    Tính toán hàm mục tiêu đơn giản dựa trên độ phủ và nhiễu.
    
    Tham số:
    COV: Phần trăm độ phủ của mạng
    ITF: Giá trị nhiễu của mạng (số lượng ăng-ten gây nhiễu)
    
    Trả về:
    Giá trị hàm mục tiêu
    """
    fitness = (100 / (1 + (100 - COV) ** 2)) + (100 / (1 + ITF ** 2))
    return fitness



def fitness(students_covered, total_students, overlap_points, total_readers, readers_used):
    X = students_covered / total_students
    Y = (total_readers - readers_used) / total_readers
    Z = (total_students - overlap_points) / total_students
    return 0.6 * X + 0.2 * Y + 0.2 * Z

def calculate_coverage(readers, students, rfid_radius=RFID_RADIUS):
    """
    Tính toán độ phủ sóng dựa trên vị trí của các đầu đọc và thẻ RFID.
    """
    for student in students:
        covered = any(np.linalg.norm(student.position - reader.position) <= rfid_radius for reader in readers)
        if not covered:
            return False  # Có thẻ không được phủ sóng
    return True  # Tất cả thẻ đều được phủ sóng

def tentative_reader_elimination(readers, students, coverage_function, max_recover_generations):
    """
    Hàm Tentative Reader Elimination (TRE)

    Parameters:
    - readers: danh sách đầu đọc hiện tại trong mạng, mỗi đầu đọc có thông tin về vị trí và công suất phát.
    - students: danh sách các thẻ RFID trong khu vực, chứa thông tin về trạng thái phủ sóng của thẻ.
    - coverage_function: hàm tính toán độ phủ sóng của mạng dựa trên đầu đọc và thẻ hiện tại.
    - max_recover_generations: số thế hệ tối đa để hệ thống khôi phục độ phủ sóng nếu giảm do loại bỏ đầu đọc.

    Returns:
    - readers: danh sách đầu đọc sau khi áp dụng TRE (có thể ít hơn hoặc giữ nguyên).
    """
    # Bước 1: Kiểm tra độ phủ sóng hiện tại (bắt buộc phải có độ phủ 100%)
    full_coverage = coverage_function(readers, students)
    
    if not full_coverage:
        print("Không thể thực hiện TRE vì không có độ phủ sóng đầy đủ.")
        return readers
    
    # Bước 2: Chọn đầu đọc có số lượng thẻ ít nhất trong phạm vi
    readers_with_fewest_tags = sorted(readers, key=lambda reader: reader['covered_tags'])

    # Loại bỏ đầu đọc phủ sóng ít thẻ nhất
    reader_to_remove = readers_with_fewest_tags[0]
    readers.remove(reader_to_remove)
    print(f"Đã loại bỏ đầu đọc {reader_to_remove['id']}.")

    # Bước 3: Kiểm tra độ phủ sóng sau khi loại bỏ đầu đọc
    for generation in range(max_recover_generations):
        full_coverage = coverage_function(readers, students)
        
        if full_coverage:
            print(f"Đã khôi phục độ phủ sóng đầy đủ sau {generation + 1} thế hệ.")
            return readers  # Loại bỏ đầu đọc thành công
        
        # # Giả lập quá trình tối ưu hóa qua các thế hệ (nếu có)
        # optimize_readers(readers, students)  # Hàm này có thể là quá trình tìm kiếm giải pháp

    # Nếu không thể đạt độ phủ đầy đủ, khôi phục đầu đọc đã loại bỏ
    readers.append(reader_to_remove)
    print(f"Đã khôi phục đầu đọc {reader_to_remove['id']} vì không thể đạt độ phủ sóng sau {max_recover_generations} thế hệ.")
    
    return readers
