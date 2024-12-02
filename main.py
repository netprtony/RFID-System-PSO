from classes import Tags, SSPSO, Readers
from functions import selection_mechanism, mainOptimization, BieuDoSoSanh, initialize_readers_with_kmeans, calculate_covered_tags,calculate_interference_basic,Reader_GRID, fitness_function_basic, tags
import numpy as np
NUM_TAGS = 265
NUM_ITERATION = 100
DIM = 2
NUM_RFID_READERS = 3
if __name__ == "__main__":
 
    readers = [Readers(np.random.rand(2) * [50, 30]) for _ in range(NUM_RFID_READERS)]
    tags = [Tags(np.random.rand(2) * [50, 50]) for _ in range(NUM_TAGS)]
    
    readers = selection_mechanism(tags, NUM_RFID_READERS, 0.95)
    sspso = SSPSO(NUM_RFID_READERS, DIM, NUM_ITERATION, readers)
    sspso.readers = mainOptimization(tags, sspso, GRID_SIZE=3.2)
    

    
    # tracking_3_2 = []
    # tracking_1_6 = []
    # tracking_0_8 = []
    # for i in range(5, 51):
    #     reader_3_2 = initialize_readers_with_kmeans(tags, i)
    #     reader_0_8 = initialize_readers_with_kmeans(tags, i)
    #     reader_1_6 = initialize_readers_with_kmeans(tags, i)

    #     sspso_3_2 = SSPSO(NUM_RFID_READERS, DIM, NUM_ITERATION, reader_3_2)
    #     sspso_1_6 = SSPSO(NUM_RFID_READERS, DIM, NUM_ITERATION, reader_1_6)
    #     sspso_0_8 = SSPSO(NUM_RFID_READERS, DIM, NUM_ITERATION, reader_0_8)

    #     reader_3_2 = sspso_3_2.optimize(tags)
    #     reader_1_6 = sspso_1_6.optimize(tags)
    #     reader_0_8 = sspso_0_8.optimize(tags)

    #     reader_3_2 = Reader_GRID(reader_3_2, 3.2)
    #     reader_1_6 = Reader_GRID(reader_1_6, 1.6)
    #     reader_0_8 = Reader_GRID(reader_0_8, 0.8)

    #     IFT_3_2 = calculate_interference_basic(reader_3_2, tags)
    #     IFT_1_6 = calculate_interference_basic(reader_1_6, tags)
    #     IFT_0_8 = calculate_interference_basic(reader_0_8, tags)
        
    #     COV_3_2 = calculate_covered_tags(reader_3_2, tags)
    #     COV_1_6 = calculate_covered_tags(reader_1_6, tags)
    #     COV_0_8 = calculate_covered_tags(reader_0_8, tags)

    #     FIT_3_2 = fitness_function_basic(COV_3_2, IFT_3_2, tags, 0.8, 0.2)
    #     FIT_1_6 = fitness_function_basic(COV_1_6, IFT_1_6, tags, 0.8, 0.2)
    #     FIT_0_8 = fitness_function_basic(COV_0_8, IFT_0_8, tags, 0.8, 0.2)
        
    #     tracking_3_2.append(np.array([FIT_3_2, i]))
    #     tracking_1_6.append(np.array([FIT_1_6, i]))
    #     tracking_0_8.append(np.array([FIT_0_8, i]))
    # BieuDoSoSanh(tracking_3_2, tracking_1_6, tracking_0_8)