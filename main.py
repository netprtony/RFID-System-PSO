from classes import Students, Readers
from functions import BieuDoStudents, BieuDoReader
NUM_RFID_READERS = 10
NUM_INDIVIDUALS = 100
DIM = 2
readers =  [Readers(DIM) for _ in range(NUM_RFID_READERS)]
students = [Students(DIM) for _ in range(NUM_INDIVIDUALS)]
BieuDoReader(readers, students)
BieuDoStudents(readers, students)
