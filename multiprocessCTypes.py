from multiprocessing import Process, Lock
from multiprocessing.sharedctypes import Array
from ctypes import Structure, c_double

class Point(Structure):
    _fields_ = [('x', c_double*3), ('y', c_double*3)]

def modify(A, value):
    for a in A:
        a.x = value
        a.y = value

if __name__ == '__main__':
    lock = Lock()
    pyarr = [0, 0, 0]
    seq = c_double * 3
    A = Array(Point, [(seq(*pyarr),seq(*pyarr))], lock=lock)

    pyarr = [1, 2, 3]

    p = Process(target=modify, args=(A, seq(*pyarr)))
    p.start()
    p.join()

    print([(a.x[0], a.y) for a in A])