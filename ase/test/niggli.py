def test():
    # Convert a selection of unit cells, both reasonable and unreasonable,
    # into their Niggli unit cell, and compare against the pre-computed values.
    # The tests and pre-computed values come from the program cctbx, in which
    # this algorithm was originally implemented.

    import numpy as np

    from ase import Atoms
    from ase.build import niggli_reduce

    cells_in = np.array([
        [[+1.38924439894498e+01, +0.00000000000000e+00, +0.00000000000000e+00],
         [+3.59907875374346e-01, +1.38877811878372e+01, +0.00000000000000e+00],
         [+6.94622199472490e+00, +6.76853982134488e+00, +1.11326936851271e+01]],
        [[+1.00000000000000e+01, +0.00000000000000e+00, +0.00000000000000e+00],
         [-5.00000000000000e+00, +8.66025403784439e+00, +0.00000000000000e+00],
         [+1.41421356237310e+01, +8.16496580927726e+00, +1.15470053837925e+01]],
        [[+1.00000000000000e+01, +0.00000000000000e+00, +0.00000000000000e+00],
         [-1.00000000000000e+01, +1.73205080756888e+01, +0.00000000000000e+00],
         [+1.50000000000000e+01, -8.66025403784438e+00, +2.44948974278318e+01]],
        [[+1.08166538263920e+01, +0.00000000000000e+00, +0.00000000000000e+00],
         [+5.40832691319598e+00, +1.27180973419769e+01, +0.00000000000000e+00],
         [+5.40832691319598e+00, +5.20911251255623e+00, +1.16023767751065e+01]],
        [[+1.01488915650922e+01, +0.00000000000000e+00, +0.00000000000000e+00],
         [-4.51609252491968e+00, +1.25938440639213e+01, +0.00000000000000e+00],
         [-4.12196081365396e+00, -5.71298877345999e+00, +1.13741460481665e+01]],
        [[+1.97989898732233e+01, +0.00000000000000e+00, +0.00000000000000e+00],
         [-1.62230498655085e+02, +1.64752132933454e+02, +0.00000000000000e+00],
         [-5.05076272276107e-01, -1.43302471019530e+01, +6.23631266175214e-01]],
        [[+1.03923048454133e+01, +0.00000000000000e+00, +0.00000000000000e+00],
         [-3.84900179459751e+00, +1.26168611463068e+01, +0.00000000000000e+00],
         [-3.27165152540788e+00, -6.30843057315338e+00, +1.11130553854464e+01]],
        [[+1.60468065358812e+01, +0.00000000000000e+00, +0.00000000000000e+00],
         [-5.92018105207268e-01, +1.33285225949130e+01, +0.00000000000000e+00],
         [-8.05612005796522e+01, -1.80304581562370e+02, +8.00942125147844e+00]],
        [[+1.04880884817015e+01, +0.00000000000000e+00, +0.00000000000000e+00],
         [-4.79909503253615e+00, +1.29602734102598e+01, +0.00000000000000e+00],
         [-3.34506458393662e+00, -6.26040929795398e+00, +1.18582384168722e+01]],
        [[+1.00498756211209e+01, +0.00000000000000e+00, +0.00000000000000e+00],
         [-3.83918515889354e+00, +1.26198517152830e+01, +0.00000000000000e+00],
         [-1.69985519994207e+00, -7.00161889241639e+00, +1.10493359612507e+01]],
        [[+1.00498756211209e+01, +0.00000000000000e+00, +0.00000000000000e+00],
         [-4.47766735594495e+00, +1.26866266221366e+01, +0.00000000000000e+00],
         [-3.68163760377696e+00, -5.94997793843316e+00, +1.14910098375475e+01]],
        [[+1.13578166916005e+01, +0.00000000000000e+00, +0.00000000000000e+00],
         [-3.36772471669551e+00, +1.32158401258701e+01, +0.00000000000000e+00],
         [-3.36772471669551e+00, -6.98718877407442e+00, +1.12177369940646e+01]],
        [[+1.18321595661992e+01, +0.00000000000000e+00, +0.00000000000000e+00],
         [-4.71877792223422e+00, +1.29511827614560e+01, +0.00000000000000e+00],
         [-3.55669082198251e+00, -6.47559138072800e+00, +1.16368667031408e+01]],
        [[+6.90590144772860e+01, +0.00000000000000e+00, +0.00000000000000e+00],
         [-8.02073428510396e+00, +4.80089958494375e+01, +0.00000000000000e+00],
         [+1.34099960000000e-08, +4.16233443900000e-07, +4.81947969343710e-03]],
        [[+8.08161863921814e+01, +0.00000000000000e+00, +0.00000000000000e+00],
         [-4.03305037431393e+01, +7.02701915501634e+01, +0.00000000000000e+00],
         [+1.95267511987431e-01, +1.40678305273598e+02, +3.93001827573170e-03]],
        [[+1.27366000000000e+01, +0.00000000000000e+00, +0.00000000000000e+00],
         [-4.95315299468855e+00, +2.88072764316797e+01, +0.00000000000000e+00],
         [-9.46867174719139e-01, -5.76708582259125e-01, +4.90035053895005e+00]],
        [[+1.27806000000000e+01, +0.00000000000000e+00, +0.00000000000000e+00],
         [+1.17491405990366e+01, +4.91718158542779e+00, +0.00000000000000e+00],
         [-6.91158909142352e+00, -1.19373435268607e+00, +2.86097847514890e+01]],
        [[+1.00000000000000e+00, +0.00000000000000e+00, +0.00000000000000e+00],
         [+5.00000000000000e-01, +8.66025403484439e-01, +0.00000000000000e+00],
         [+0.00000000000000e+00, +0.00000000000000e+00, +1.00000000000000e+00]],
        [[+1.00000000000000e+00, +0.00000000000000e+00, +0.00000000000000e+00],
         [-5.00000000000000e-01, +8.66025403484439e-01, +0.00000000000000e+00],
         [+0.00000000000000e+00, +0.00000000000000e+00, +1.00000000000000e+00]],
        [[+0.00000000000000e+00, +1.50000000000000e-02, +0.00000000000000e+00],
         [+2.30000000000000e-02, +7.50000000000000e-03, +0.00000000000000e+00],
         [+0.00000000000000e+00, -4.13728692927329e-05, +8.31877949084600e-01]]])

    cells_out = np.array([
        [[+1.38924439894498e+01, +0.00000000000000e+00, +0.00000000000000e+00],
         [+3.59907875374344e-01, +1.38877811878372e+01, +0.00000000000000e+00],
         [+6.94622199472490e+00, +6.76853982134488e+00, +1.11326936851271e+01]],
        [[+1.00000000000000e+01, +0.00000000000000e+00, +0.00000000000000e+00],
         [+5.00000000000000e+00, +8.66025403784439e+00, +0.00000000000000e+00],
         [+8.57864376268997e-01, +4.95288228567129e-01, +1.15470053837925e+01]],
        [[+1.00000000000000e+01, +0.00000000000000e+00, +0.00000000000000e+00],
         [+1.06057523872491e-15, +1.73205080756888e+01, +0.00000000000000e+00],
         [-5.00000000000000e+00, -8.66025403784442e+00, +2.44948974278318e+01]],
        [[+1.08166538263920e+01, +0.00000000000000e+00, +0.00000000000000e+00],
         [+5.40832691319598e+00, +1.27180973419769e+01, +0.00000000000000e+00],
         [+5.40832691319598e+00, +5.20911251255623e+00, +1.16023767751065e+01]],
        [[+1.01488915650922e+01, +0.00000000000000e+00, +0.00000000000000e+00],
         [-4.51609252491968e+00, +1.25938440639213e+01, +0.00000000000000e+00],
         [-4.12196081365396e+00, -5.71298877345999e+00, +1.13741460481665e+01]],
        [[+1.36381816969869e+01, +0.00000000000000e+00, +0.00000000000000e+00],
         [+6.81909084849243e+00, +1.26293309403154e+01, +0.00000000000000e+00],
         [+6.81909084849065e+00, +4.47371284092803e+00, +1.18104146166409e+01]],
        [[+1.03923048454133e+01, +0.00000000000000e+00, +0.00000000000000e+00],
         [-3.84900179459751e+00, +1.26168611463068e+01, +0.00000000000000e+00],
         [-3.27165152540788e+00, -6.30843057315338e+00, +1.11130553854464e+01]],
        [[+1.26095202129182e+01, +0.00000000000000e+00, +0.00000000000000e+00],
         [+6.30476010645935e+00, +1.17579760163048e+01, +0.00000000000000e+00],
         [+3.15238005323008e+00, +5.87898800815218e+00, +1.15542200082912e+01]],
        [[+1.04880884817015e+01, +0.00000000000000e+00, +0.00000000000000e+00],
         [-4.79909503253615e+00, +1.29602734102598e+01, +0.00000000000000e+00],
         [-3.34506458393662e+00, -6.26040929795398e+00, +1.18582384168722e+01]],
        [[+1.00498756211209e+01, +0.00000000000000e+00, +0.00000000000000e+00],
         [-4.51083526228529e+00, +1.23956591287645e+01, +0.00000000000000e+00],
         [-3.83918515889354e+00, -5.71984630990568e+00, +1.12491784369700e+01]],
        [[+1.00498756211209e+01, +0.00000000000000e+00, +0.00000000000000e+00],
         [-4.47766735594495e+00, +1.26866266221366e+01, +0.00000000000000e+00],
         [-3.68163760377696e+00, -5.94997793843316e+00, +1.14910098375475e+01]],
        [[+1.13578166916005e+01, +0.00000000000000e+00, +0.00000000000000e+00],
         [-4.62236725820948e+00, +1.28309672640153e+01, +0.00000000000000e+00],
         [-3.36772471669551e+00, -6.41548363200768e+00, +1.15542200082913e+01]],
        [[+1.18321595661992e+01, +0.00000000000000e+00, +0.00000000000000e+00],
         [-4.71877792223422e+00, +1.29511827614560e+01, +0.00000000000000e+00],
         [-3.55669082198251e+00, -6.47559138072800e+00, +1.16368667031408e+01]],
        [[+4.81947971142972e-03, +0.00000000000000e+00, +0.00000000000000e+00],
         [+4.12397039845618e-03, +4.86743859122682e+01, +0.00000000000000e+00],
         [+4.62732595971025e-03, +1.13797841621313e+01, +6.81149615940608e+01]],
        [[+1.43683914413843e-01, +0.00000000000000e+00, +0.00000000000000e+00],
         [+4.73841211849216e-02, +8.02075186538656e+00, +0.00000000000000e+00],
         [+9.29303317118020e-03, +8.28854375915883e-01, +1.93660401476964e+01]],
        [[+5.02420000000000e+00, +0.00000000000000e+00, +0.00000000000000e+00],
         [-2.40035596861745e+00, +1.25083680303996e+01, +0.00000000000000e+00],
         [-2.37319883118274e+00, -5.49894680458153e+00, +2.86098306766757e+01]],
        [[+5.02419976114664e+00, +0.00000000000000e+00, +0.00000000000000e+00],
         [-2.40036499209593e+00, +1.25083662987906e+01, +0.00000000000000e+00],
         [-2.37320481266200e+00, -5.49892622854049e+00, +2.86097847514890e+01]],
        [[+1.00000000000000e+00, +0.00000000000000e+00, +0.00000000000000e+00],
         [-5.00000000000000e-01, +8.66025403484439e-01, +0.00000000000000e+00],
         [+0.00000000000000e+00, +0.00000000000000e+00, +1.00000000000000e+00]],
        [[+1.00000000000000e+00, +0.00000000000000e+00, +0.00000000000000e+00],
         [-5.00000000000000e-01, +8.66025403484439e-01, +0.00000000000000e+00],
         [+0.00000000000000e+00, +0.00000000000000e+00, +1.00000000000000e+00]],
        [[+1.50000000000000e-02, +0.00000000000000e+00, +0.00000000000000e+00],
         [+7.50000000000000e-03, +2.30000000000000e-02, +0.00000000000000e+00],
         [+4.13728692926938e-05, +1.79760110872209e-16, +8.31877949084600e-01]]])

    conf = Atoms(pbc=True)

    for i, cell in enumerate(cells_in):
        conf.set_cell(cell)
        niggli_reduce(conf)
        cell = conf.get_cell()
        diff = np.linalg.norm(cell - cells_out[i])
        assert diff < 1e-5, \
            'Difference between unit cells is too large! ({0})'.format(diff)
