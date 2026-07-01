import sys
import numpy as np

np.int = int
np.float = float
np.bool = bool
np.complex = complex
np.object = object
np.str = str
np.typeDict = np.sctypeDict

import matplotlib.pyplot as plt
import PyDSTool as dst

I1 = 3.996
I2 = 3.997
gamma1 = 2.996
gamma2 = 2.998
w11 = 4.994
w22 = 4.989

DSargs = dst.args(name='toggle_switch')
DSargs.pars = {'c': 0.0, 'I1': I1, 'I2': I2, 'g1': gamma1, 'g2': gamma2, 'w11': w11, 'w22': w22}
DSargs.pdomain = {'c': [-15.0, 5.0]} 

DSargs.fnspecs = {'relu': (['x'], 'x * heav(x)')}
DSargs.varspecs = {
    'x1': 'w11 * (relu(x1)**4)/(1.0 + relu(x1)**4) + c * (relu(x2)**4)/(1.0 + relu(x2)**4) + I1 - g1*x1',
    'x2': 'c * (relu(x1)**4)/(1.0 + relu(x1)**4) + w22 * (relu(x2)**4)/(1.0 + relu(x2)**4) + I2 - g2*x2'
}

DSargs.ics = {'x1': 2.5, 'x2': 2.5} 
DSargs.tdomain = [0, 100]

ode = dst.Generator.Vode_ODEsystem(DSargs)
PC = dst.ContClass(ode)

# Curve 1: The central branch (starts from c=0, where it's monostable)
PCargs = dst.args(name='EQ1', type='EP-C')
PCargs.freepars = ['c']
PCargs.StepSize = -0.1
PCargs.MaxNumPoints = 400
PCargs.MaxStepSize = 0.5
PCargs.MinStepSize = 1e-4
PCargs.LocBifPoints = 'all'
PCargs.SaveEigen = True
PCargs.StopAtPoints = 'B'

PC.newCurve(PCargs)
PC['EQ1'].forward()
PC['EQ1'].backward()

# Curve 2: The upper stable branch (starts from strong repression c=-15)
ode.set(pars={'c': -15.0}, ics={'x1': 3.0, 'x2': -3.6})
PCargs_up = dst.args(name='EQ_up', type='EP-C')
PCargs_up.freepars = ['c']
PCargs_up.StepSize = 0.1  # Go right towards c=0
PCargs_up.MaxNumPoints = 400
PCargs_up.MaxStepSize = 0.5
PCargs_up.MinStepSize = 1e-4
PCargs_up.LocBifPoints = 'all'
PCargs_up.SaveEigen = True
PCargs_up.StopAtPoints = 'B'

PC.newCurve(PCargs_up)
PC['EQ_up'].forward()

# Curve 3: The lower stable branch (starts from strong repression c=-15)
ode.set(pars={'c': -15.0}, ics={'x1': -3.6, 'x2': 3.0})
PCargs_down = dst.args(name='EQ_down', type='EP-C')
PCargs_down.freepars = ['c']
PCargs_down.StepSize = 0.1  # Go right towards c=0
PCargs_down.MaxNumPoints = 400
PCargs_down.MaxStepSize = 0.5
PCargs_down.MinStepSize = 1e-4
PCargs_down.LocBifPoints = 'all'
PCargs_down.SaveEigen = True
PCargs_down.StopAtPoints = 'B'

PC.newCurve(PCargs_down)
PC['EQ_down'].forward()

try:
    plt.figure(figsize=(10,6))
    # PC.display plots all successfully calculated branches!
    PC.display(['c', 'x1'], stability=True, figure=1)
    plt.savefig('pydstool_bifurcation.png', dpi=300, bbox_inches='tight')
    print("SUCCESS")
except Exception as e:
    print(f"FAILED PLOT: {e}")
