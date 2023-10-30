"""Check that reading and writing .con files is consistent."""

from numpy import array

import ase
import ase.io

# Error tolerance.
TOL = 1e-6

# A correct .con file.
CON_FILE = """\
10000 RANDOM NUMBER SEED
0.0000 TIME
    7.2200000000    10.8760700623    14.5090868079
   84.7244899634    86.0479084995    85.7006486298
0 0
0 0 0
  1
 96
1.0
Cu
Coordinates of Component 1
 1.04833333333334  0.96500000000003  0.90250000000000  0 1
 3.02000000000000  2.77000000000000  0.90250000000000  0 2
 2.97833333333334  1.09000000000003  2.70750000000000  0 3
 1.34000000000000  2.89500000000000  2.70750000000000  0 4
 4.65833333333334  0.96500000000003  0.90250000000000  0 5
 6.63000000000000  2.77000000000000  0.90250000000000  0 6
 6.58833333333334  1.09000000000003  2.70750000000000  0 7
 4.95000000000000  2.89500000000000  2.70750000000000  0 8
 1.38166666666666  4.57499999999997  0.90250000000000  0 9
 3.35333333333334  6.38000000000003  0.90250000000000  0 10
 3.31166666666666  4.69999999999997  2.70750000000000  0 11
 1.67333333333334  6.50500000000003  2.70750000000000  0 12
 4.99166666666666  4.57499999999997  0.90250000000000  0 13
 6.96333333333334  6.38000000000003  0.90250000000000  0 14
 6.92166666666666  4.69999999999997  2.70750000000000  0 15
 5.28333333333334  6.50500000000003  2.70750000000000  0 16
 1.71500000000000  8.18500000000000  0.90250000000000  0 17
 3.68666666666666  9.98999999999997  0.90250000000000  0 18
 3.64500000000000  8.31000000000000  2.70750000000000  0 19
 2.00666666666666 10.11499999999997  2.70750000000000  0 20
 5.32500000000000  8.18500000000000  0.90250000000000  0 21
 7.29666666666666  9.98999999999997  0.90250000000000  0 22
 7.25500000000000  8.31000000000000  2.70750000000000  0 23
 5.61666666666666 10.11499999999997  2.70750000000000  0 24
 1.29833333333334  1.21500000000003  4.51250000000000  0 25
 3.27000000000000  3.02000000000000  4.51250000000000  0 26
 3.22833333333334  1.34000000000003  6.31750000000000  0 27
 1.59000000000000  3.14500000000000  6.31750000000000  0 28
 4.90833333333334  1.21500000000003  4.51250000000000  0 29
 6.88000000000000  3.02000000000000  4.51250000000000  0 30
 6.83833333333334  1.34000000000003  6.31750000000000  0 31
 5.20000000000000  3.14500000000000  6.31750000000000  0 32
 1.63166666666666  4.82499999999997  4.51250000000000  0 33
 3.60333333333334  6.63000000000003  4.51250000000000  0 34
 3.56166666666666  4.94999999999997  6.31750000000000  0 35
 1.92333333333334  6.75500000000003  6.31750000000000  0 36
 5.24166666666666  4.82499999999997  4.51250000000000  0 37
 7.21333333333334  6.63000000000003  4.51250000000000  0 38
 7.17166666666666  4.94999999999997  6.31750000000000  0 39
 5.53333333333334  6.75500000000003  6.31750000000000  0 40
 1.96500000000000  8.43500000000000  4.51250000000000  0 41
 3.93666666666666 10.23999999999997  4.51250000000000  0 42
 3.89500000000000  8.56000000000000  6.31750000000000  0 43
 2.25666666666666 10.36499999999997  6.31750000000000  0 44
 5.57500000000000  8.43500000000000  4.51250000000000  0 45
 7.54666666666666 10.23999999999997  4.51250000000000  0 46
 7.50500000000000  8.56000000000000  6.31750000000000  0 47
 5.86666666666666 10.36499999999997  6.31750000000000  0 48
 1.54833333333334  1.46500000000003  8.12250000000000  0 49
 3.52000000000000  3.27000000000000  8.12250000000000  0 50
 3.47833333333334  1.59000000000003  9.92750000000000  0 51
 1.84000000000000  3.39500000000000  9.92750000000000  0 52
 5.15833333333334  1.46500000000003  8.12250000000000  0 53
 7.13000000000000  3.27000000000000  8.12250000000000  0 54
 7.08833333333334  1.59000000000003  9.92750000000000  0 55
 5.45000000000000  3.39500000000000  9.92750000000000  0 56
 1.88166666666666  5.07499999999997  8.12250000000000  0 57
 3.85333333333334  6.88000000000003  8.12250000000000  0 58
 3.81166666666666  5.19999999999997  9.92750000000000  0 59
 2.17333333333334  7.00500000000003  9.92750000000000  0 60
 5.49166666666666  5.07499999999997  8.12250000000000  0 61
 7.46333333333334  6.88000000000003  8.12250000000000  0 62
 7.42166666666666  5.19999999999997  9.92750000000000  0 63
 5.78333333333334  7.00500000000003  9.92750000000000  0 64
 2.21500000000000  8.68500000000000  8.12250000000000  0 65
 4.18666666666666 10.48999999999997  8.12250000000000  0 66
 4.14500000000000  8.81000000000000  9.92750000000000  0 67
 2.50666666666666 10.61499999999997  9.92750000000000  0 68
 5.82500000000000  8.68500000000000  8.12250000000000  0 69
 7.79666666666666 10.48999999999997  8.12250000000000  0 70
 7.75500000000000  8.81000000000000  9.92750000000000  0 71
 6.11666666666666 10.61499999999997  9.92750000000000  0 72
 1.79833333333334  1.71500000000003 11.73250000000000  0 73
 3.77000000000000  3.52000000000000 11.73250000000000  0 74
 3.72833333333334  1.84000000000003 13.53750000000000  0 75
 2.09000000000000  3.64500000000000 13.53750000000000  0 76
 5.40833333333334  1.71500000000003 11.73250000000000  0 77
 7.38000000000000  3.52000000000000 11.73250000000000  0 78
 7.33833333333334  1.84000000000003 13.53750000000000  0 79
 5.70000000000000  3.64500000000000 13.53750000000000  0 80
 2.13166666666666  5.32499999999997 11.73250000000000  0 81
 4.10333333333334  7.13000000000003 11.73250000000000  0 82
 4.06166666666666  5.44999999999997 13.53750000000000  0 83
 2.42333333333334  7.25500000000003 13.53750000000000  0 84
 5.74166666666666  5.32499999999997 11.73250000000000  0 85
 7.71333333333334  7.13000000000003 11.73250000000000  0 86
 7.67166666666666  5.44999999999997 13.53750000000000  0 87
 6.03333333333334  7.25500000000003 13.53750000000000  0 88
 2.46500000000000  8.93500000000000 11.73250000000000  0 89
 4.43666666666666 10.73999999999997 11.73250000000000  0 90
 4.39500000000000  9.06000000000000 13.53750000000000  0 91
 2.75666666666666 10.86499999999997 13.53750000000000  0 92
 6.07500000000000  8.93500000000000 11.73250000000000  0 93
 8.04666666666666 10.73999999999997 11.73250000000000  0 94
 8.00500000000000  9.06000000000000 13.53750000000000  0 95
 6.36666666666666 10.86499999999997 13.53750000000000  0 96
"""
# The corresponding data as an ASE Atoms object.
data = ase.Atoms('Cu96',
                 cell=array([[7.22, 0, 0],
                             [1, 10.83, 0],
                             [1, 1, 14.44]]),
                 positions=array([[1.04833333, 0.965, 0.9025],
                                  [3.02, 2.77, 0.9025],
                                  [2.97833333, 1.09, 2.7075],
                                  [1.34, 2.895, 2.7075],
                                  [4.65833333, 0.965, 0.9025],
                                  [6.63, 2.77, 0.9025],
                                  [6.58833333, 1.09, 2.7075],
                                  [4.95, 2.895, 2.7075],
                                  [1.38166667, 4.575, 0.9025],
                                  [3.35333333, 6.38, 0.9025],
                                  [3.31166667, 4.7, 2.7075],
                                  [1.67333333, 6.505, 2.7075],
                                  [4.99166667, 4.575, 0.9025],
                                  [6.96333333, 6.38, 0.9025],
                                  [6.92166667, 4.7, 2.7075],
                                  [5.28333333, 6.505, 2.7075],
                                  [1.715, 8.185, 0.9025],
                                  [3.68666667, 9.99, 0.9025],
                                  [3.645, 8.31, 2.7075],
                                  [2.00666667, 10.115, 2.7075],
                                  [5.325, 8.185, 0.9025],
                                  [7.29666667, 9.99, 0.9025],
                                  [7.255, 8.31, 2.7075],
                                  [5.61666667, 10.115, 2.7075],
                                  [1.29833333, 1.215, 4.5125],
                                  [3.27, 3.02, 4.5125],
                                  [3.22833333, 1.34, 6.3175],
                                  [1.59, 3.145, 6.3175],
                                  [4.90833333, 1.215, 4.5125],
                                  [6.88, 3.02, 4.5125],
                                  [6.83833333, 1.34, 6.3175],
                                  [5.2, 3.145, 6.3175],
                                  [1.63166667, 4.825, 4.5125],
                                  [3.60333333, 6.63, 4.5125],
                                  [3.56166667, 4.95, 6.3175],
                                  [1.92333333, 6.755, 6.3175],
                                  [5.24166667, 4.825, 4.5125],
                                  [7.21333333, 6.63, 4.5125],
                                  [7.17166667, 4.95, 6.3175],
                                  [5.53333333, 6.755, 6.3175],
                                  [1.965, 8.435, 4.5125],
                                  [3.93666667, 10.24, 4.5125],
                                  [3.895, 8.56, 6.3175],
                                  [2.25666667, 10.365, 6.3175],
                                  [5.575, 8.435, 4.5125],
                                  [7.54666667, 10.24, 4.5125],
                                  [7.505, 8.56, 6.3175],
                                  [5.86666667, 10.365, 6.3175],
                                  [1.54833333, 1.465, 8.1225],
                                  [3.52, 3.27, 8.1225],
                                  [3.47833333, 1.59, 9.9275],
                                  [1.84, 3.395, 9.9275],
                                  [5.15833333, 1.465, 8.1225],
                                  [7.13, 3.27, 8.1225],
                                  [7.08833333, 1.59, 9.9275],
                                  [5.45, 3.395, 9.9275],
                                  [1.88166667, 5.075, 8.1225],
                                  [3.85333333, 6.88, 8.1225],
                                  [3.81166667, 5.2, 9.9275],
                                  [2.17333333, 7.005, 9.9275],
                                  [5.49166667, 5.075, 8.1225],
                                  [7.46333333, 6.88, 8.1225],
                                  [7.42166667, 5.2, 9.9275],
                                  [5.78333333, 7.005, 9.9275],
                                  [2.215, 8.685, 8.1225],
                                  [4.18666667, 10.49, 8.1225],
                                  [4.145, 8.81, 9.9275],
                                  [2.50666667, 10.615, 9.9275],
                                  [5.825, 8.685, 8.1225],
                                  [7.79666667, 10.49, 8.1225],
                                  [7.755, 8.81, 9.9275],
                                  [6.11666667, 10.615, 9.9275],
                                  [1.79833333, 1.715, 11.7325],
                                  [3.77, 3.52, 11.7325],
                                  [3.72833333, 1.84, 13.5375],
                                  [2.09, 3.645, 13.5375],
                                  [5.40833333, 1.715, 11.7325],
                                  [7.38, 3.52, 11.7325],
                                  [7.33833333, 1.84, 13.5375],
                                  [5.7, 3.645, 13.5375],
                                  [2.13166667, 5.325, 11.7325],
                                  [4.10333333, 7.13, 11.7325],
                                  [4.06166667, 5.45, 13.5375],
                                  [2.42333333, 7.255, 13.5375],
                                  [5.74166667, 5.325, 11.7325],
                                  [7.71333333, 7.13, 11.7325],
                                  [7.67166667, 5.45, 13.5375],
                                  [6.03333333, 7.255, 13.5375],
                                  [2.465, 8.935, 11.7325],
                                  [4.43666667, 10.74, 11.7325],
                                  [4.395, 9.06, 13.5375],
                                  [2.75666667, 10.865, 13.5375],
                                  [6.075, 8.935, 11.7325],
                                  [8.04666667, 10.74, 11.7325],
                                  [8.005, 9.06, 13.5375],
                                  [6.36666667, 10.865, 13.5375]]),
                 pbc=(True, True, True))


def test_eon_readwrite():
    # First, write a correct .con file and try to read it.
    con_file = 'pos.con'
    with open(con_file, 'w') as fd:
        fd.write(CON_FILE)
    box = ase.io.read(con_file, format='eon')
    # Check cell vectors.
    assert (abs(box.cell - data.cell)).sum() < TOL  # read: cell vector check
    # Check atom positions.
    # read: position check
    assert (abs(box.positions - data.positions)).sum() < TOL

    # Now that we know that reading a .con file works, we will write
    # one and read it back in.
    out_file = 'out.con'
    ase.io.write(out_file, data, format='eon')
    data2 = ase.io.read(out_file, format='eon')
    # Check cell vectors.
    # write: cell vector check
    assert (abs(data2.cell - data.cell)).sum() < TOL
    # Check atom positions.
    # write: position check
    assert (abs(data2.positions - data.positions)).sum() < TOL
