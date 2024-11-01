from classes import Tags, Readers, SSPSO, GRID_X, GRID_Y
from functions import BieuDotags, BieuDoReader, RFID_RADIUS, generate_hexagon_centers

NUM_INDIVIDUALS = 90
NUM_ITERATION = 100
DIM = 2
positionHexagonReader , NUM_RFID_READERS = generate_hexagon_centers(GRID_X, GRID_Y)
readers =  [Readers(DIM) for _ in range(NUM_RFID_READERS)]
tags = [Tags(DIM) for _ in range(NUM_INDIVIDUALS)]
for idx, reader in enumerate(readers):
            reader.position = positionHexagonReader[idx]
sspso = SSPSO(NUM_RFID_READERS, DIM, NUM_ITERATION)

BieuDoReader(readers, tags)
readers = sspso.optimize(tags, RFID_RADIUS)
BieuDoReader(readers, tags)
#BieuDotags(readers, tags)
