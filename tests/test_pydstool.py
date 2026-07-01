import numpy as np
np.int = int
np.float = float
np.bool = bool
np.complex = complex
np.object = object
np.str = str
np.typeDict = np.sctypeDict

import PyDSTool as dst

def test_pydstool():
    pars = {'c': 1.0}
    icdict = {'x': 1.0}
    
    x_str = '-x*x*x + c*x'
    
    DSargs = dst.args(name='test')
    DSargs.varspecs = {'x': x_str}
    DSargs.pars = pars
    DSargs.ics = icdict
    DSargs.tdata = [0, 10]
    
    ODE = dst.Generator.Vode_ODEsystem(DSargs)
    
    PC = dst.ContClass(ODE)
    PCargs = dst.args(name='EQ1', type='EP-C')
    PCargs.freepars = ['c']
    PCargs.StepSize = 0.05
    PCargs.MaxNumPoints = 50
    PCargs.MaxStepSize = 0.1
    PCargs.LocBifPoints = 'all'
    PCargs.SaveEigen = True
    
    PC.newCurve(PCargs)
    print("Computing curve...")
    PC['EQ1'].forward()
    print("Successfully ran PyCont!")
    
if __name__ == '__main__':
    test_pydstool()
