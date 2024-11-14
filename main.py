from classes import Tags, SSPSO, birch_clustering, Readers, CFTree, CFNode
from functions import selection_mechanism, mainOptimization, BieuDoReader

NUM_INDIVIDUALS = 100
NUM_ITERATION = 100
DIM = 2
NUM_RFID_READERS = 3
if __name__ == "__main__":
    tags = [Tags(DIM) for _ in range(NUM_INDIVIDUALS)]
    readers = selection_mechanism(tags, NUM_RFID_READERS)
    sspso = SSPSO(NUM_RFID_READERS, DIM, NUM_ITERATION, readers)
    mainOptimization(tags, readers, sspso)
    #BieuDoReader(readers, tags)
    #clusters = birch_clustering(tags)
    #print(len(clusters))
    # Khởi tạo các reader tại các vị trí của các clusters
    #readers = [Readers(position=cluster) for cluster in clusters]
    #BieuDoReader(readers, tags)