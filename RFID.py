import numpy as np

# Giả lập vị trí các đầu đọc RFID (readers)
readers = np.array([
    [0, 0],
    [10, 0],
    [0, 10],
    [10, 10]
])

# Giả lập vị trí của các thẻ RFID (tags) cần giám sát
tags = np.array([
    [2, 2],
    [8, 2],
    [2, 8],
    [8, 8]
])

# Tính khoảng cách giữa các thẻ RFID và đầu đọc
def calculate_distances(tags, readers):
    distances = np.zeros((tags.shape[0], readers.shape[0]))
    for i, tag in enumerate(tags):
        for j, reader in enumerate(readers):
            distances[i, j] = np.linalg.norm(tag - reader)
    return distances

distances = calculate_distances(tags, readers)
print("Khoảng cách giữa các thẻ và đầu đọc:")
print(distances)

