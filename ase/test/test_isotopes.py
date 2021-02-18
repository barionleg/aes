from ase.data.isotopes import parse_isotope_data


def test_isotopes():
    raw_data = ['<html>',
                '__________________________________________________________',
                '1   H   1    1.00782503223(9)   0.999885(70)'
                '  [1.00784,1.00811]  m',
                '    D   2    2.01410177812(12)  0.000115(70)',
                '    T   3    3.0160492779(24)               ',
                '    H   4    4.02643(11)                    ',
                '        5    5.035311(96)                   ',
                '        6    6.04496(27)                    ',
                '        7    7.0527(11#)                    ',
                '___________________________________________________________',
                '2   He  3    3.0160293201(25)   0.00000134(3)  4.002602(2)'
                '      g,r',
                '        4    4.00260325413(6)   0.99999866(3)',
                '        5    5.012057(21)                   ',
                '        6    6.018885891(57)                ',
                '        7    7.0279907(81)                  ',
                '        8    8.033934390(95)                ',
                '        9    9.043946(50)                   ',
                '        10   10.05279(11)                   ',
                '___________________________________________________________']

    isotopes = parse_isotope_data(raw_data)

    assert isotopes[1][2]['mass'] == 2.01410177812
    assert isotopes[2][4]['composition'] == 0.99999866
