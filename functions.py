import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from classes import SSPSO

GRID_X, GRID_Y = 10, 10
RFID_RADIUS = 3.69
NUM_ITERATION = 50
UPDATE_INTERVAL = 500
NUM_RFID_READERS = 35
DIM = 2


def BieuDoStudents(READERS, STUDENTS):
    fig, ax = plt.subplots()
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_xlim(0, GRID_X)
    ax.set_ylim(0, GRID_Y)
    ax.set_aspect('equal', 'box')
    
    student_positions = np.array([student.position for student in STUDENTS])
    reader_positions = np.array([reader.position for reader in READERS])
    
    scatter_student = ax.scatter(student_positions[:, 0], student_positions[:, 1], color='blue', label='Students', s=10)
    ax.scatter(reader_positions[:, 0], reader_positions[:, 1], color='red', label='RFID Readers', marker='^')
    
    circles = [plt.Circle((x, y), RFID_RADIUS, color='red', fill=True, alpha=0.2, linestyle='--') for x, y in reader_positions]
    for circle in circles:
        ax.add_artist(circle)
    
    count_text = ax.text(0.02, 1.05, 'Students in range: 0', transform=ax.transAxes, fontsize=12, verticalalignment='top')

    def update_student(frame):
        for student in STUDENTS:
            student.update_position()
            student.covered = any(np.linalg.norm(student.position - reader.position) <= RFID_RADIUS for reader in READERS)
        
        student_positions = np.array([student.position for student in STUDENTS])
        student_positions = np.atleast_2d(student_positions)

        colors = ['green' if student.covered else 'blue' for student in STUDENTS]

        scatter_student.set_offsets(student_positions)
        scatter_student.set_color(colors)

        count_in_range = sum(student.covered for student in STUDENTS)
        count_text.set_text(f'Students in range: {count_in_range}')
        print(f"Iteration {frame}: {count_in_range} students in range")

        return scatter_student

    ani = animation.FuncAnimation(fig, update_student, frames=999999, interval=UPDATE_INTERVAL, blit=False, repeat=False)
    plt.show()

def BieuDoReader(READERS, STUDENTS):
    fig, ax = plt.subplots()
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_xlim(0, GRID_X)
    ax.set_ylim(0, GRID_Y)
    ax.set_aspect('equal', 'box')

    # Trích xuất vị trí sinh viên và đầu đọc RFID
    student_positions = np.array([student.position for student in STUDENTS])
    reader_positions = np.array([reader.position for reader in READERS])
    sspso = SSPSO(NUM_RFID_READERS, DIM, NUM_ITERATION)
    ax.scatter(student_positions[:, 0], student_positions[:, 1], color='blue', label='Students', s=10)
    scatter_rfid = ax.scatter(reader_positions[:, 0], reader_positions[:, 1], color='red', label='RFID Readers', marker='^')

    # Tạo các hình tròn biểu diễn vùng phủ sóng của các đầu đọc RFID
    circles = [plt.Circle((x, y), RFID_RADIUS, color='red', fill=True, alpha=0.2, linestyle='--') for x, y in reader_positions]
    for circle in circles:
        ax.add_artist(circle)

    # Text hiển thị số sinh viên trong vùng bán kính
    count_text = ax.text(0.02, 1.05, 'Students in range: 0', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    count_in_range = 0
    for student in student_positions:
        if any(np.linalg.norm(student - reader.position) <= RFID_RADIUS for reader in READERS):
            count_in_range += 1  

    count_text.set_text(f'Students in range: {count_in_range}')
    # print(f"Iteration {i}: {count_in_range} students in range")

    sspso.optimize(STUDENTS, RFID_RADIUS)
    

    # Cập nhật vị trí của các đầu đọc RFID
    reader_positions = np.array([reader.position for reader in READERS])
    
    scatter_rfid.set_offsets(reader_positions)
    
    # Cập nhật vị trí của các hình tròn vùng phủ sóng
    for i, circle in enumerate(circles):
        circle.center = reader_positions[i]
    
    return scatter_rfid, *circles
plt.show()
    