from classes import Students, Readers
from functions import BieuDoStudents, BieuDoReader
NUM_RFID_READERS = 30
NUM_INDIVIDUALS = 700
DIM = 2
readers =  [Readers(DIM) for _ in range(NUM_RFID_READERS)]
students = [Students(DIM) for _ in range(NUM_INDIVIDUALS)]
BieuDoReader(readers, students)
BieuDoStudents(readers, students)
