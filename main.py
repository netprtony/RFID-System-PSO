from classes import Tags, ParticleSwarmOptimizationAlgorithm, Readers, FireflyAlgorithm
from functions import BieuDoReaderTongHop, PSO_Algorithm, FA_Algorithm, selection_mechanism, initialize_readers_with_kmeans, tags, BieuDoSoSanh, BieuDoReader, tracking_PSO_time, tracking_PSO_COV, tracking_PSO_IFT, tracking_PSO_Fitness, tracking_FA_time, tracking_FA_COV, tracking_FA_IFT, tracking_FA_Fitness
import numpy as np
import time
NUM_TAGS = 265
NUM_ITERATION = 100
DIM = 2
INITIAL_NUM_RFID_READERS = 3
import random
if __name__ == "__main__":
 
    # readers = [Readers(np.random.rand(2) * [50, 30]) for _ in range(NUM_RFID_READERS)]
    # tags = [Tags(np.random.rand(2) * [50, 50]) for _ in range(NUM_TAGS)]
    #tags = random.sample(random_tag, 99)
    numTag = tags[:200]
    # readers = selection_mechanism(tag_100, INITIAL_NUM_RFID_READERS,1)
    # sspso = SSPSO(len(readers), DIM, NUM_ITERATION, readers)
    # sspso.readers = mainOptimization(tag_100, sspso, GRID_SIZE=3.2)
    GRID_SIZE = 1.3
    GRID_X  = 50
    GRID_Y = 50
    readers_PSO = selection_mechanism(numTag, INITIAL_NUM_RFID_READERS, 1)
    numreaderFA = len(readers_PSO)
    PSO_Algorithm(readers_PSO, numTag, 50 , 50, GRID_SIZE)
    
    reader_FA = [Readers(np.random.rand(2) * [50, 50]) for _ in range(numreaderFA)]
    FA_Algorithm(reader_FA, numTag, 50 , 50, GRID_SIZE)
    BieuDoReaderTongHop(readers_PSO, reader_FA, numTag, GRID_SIZE, GRID_X, GRID_Y, 3.69)

#     values_grid = [5.2, 2.6, 1.3]  # Cố định giá trị x
#     tracking_FA = []  # Lưu kết quả cho FA
#     tracking_PSO = []  # Lưu kết quả cho PSO
#     grid =  1.3
#     num_readers = 50
#     for i in range(5, 51, 5):
#         print(f"Running with {i} readers")
#         reader_PSO = initialize_readers_with_kmeans(numTag, i)
#         reader_FA = [Readers(np.random.rand(2) * [50, 50]) for _ in range(i)]
#         sspso = ParticleSwarmOptimizationAlgorithm(num_readers, DIM, NUM_ITERATION, reader_PSO)
#         #FA_Algorithm(reader_FA, numTag, 50, 50, grid)
#         #PSO_Algorithm(reader_PSO, numTag, 50, 50, grid)
#         value_FA = tracking_FA_Fitness(reader_FA, numTag, grid)
#         value_PSO = tracking_PSO_Fitness(sspso, numTag, grid)
#         tracking_FA.append([i, value_FA])  # Thêm giá trị FA
#         tracking_PSO.append([i, value_PSO])  # Thêm giá trị PSO

# # # Vẽ biểu đồ
#     BieuDoSoSanh(tracking_FA, tracking_PSO, "Số lượng đầu đọc", "Giá trị fitness", f"So sánh giá trị fitness giữa FA và PSO mắt lưới {grid}")

    # for grid in values_grid:
    #     print(f"Running with grid size {grid}")
    #     reader_PSO = initialize_readers_with_kmeans(numTag, num_readers)
    #     reader_FA = [Readers(np.random.rand(2) * [50, 50]) for _ in range(num_readers)]
    #     sspso = ParticleSwarmOptimizationAlgorithm(num_readers, DIM, NUM_ITERATION, reader_PSO)
    #     FA_Algorithm(reader_FA, numTag, 50, 50, grid)
    #     PSO_Algorithm(reader_PSO, numTag, 50, 50, grid)
    #     value_FA = tracking_FA_COV(reader_FA, numTag, grid)
    #     value_PSO = tracking_PSO_COV(sspso, numTag, grid)
    #         # Thêm giá trị vào danh sách
    #     tracking_FA.append((grid, value_FA))  # Thêm giá trị FA
    #     tracking_PSO.append((grid, value_PSO))  # Thêm giá trị PSO
    # # Vẽ biểu đồ
    # BieuDoSoSanh(tracking_FA, tracking_PSO, "Kích thước ô lưới", "Độ bao phủ (%)", "So sánh độ bao phủ giữa FA và PSO")
