from PyMPDATA.options import Options
# options = Options(n_iters=2, infinite_gauge=True, flux_corrected_transport=True)

from PyMPDATA.stepper import Stepper
# stepper = Stepper(options=options, grid=(ny, nx))

from PyMPDATA.solver import Solver
# solver = Solver(stepper=stepper, advectee=advectee, advector=advector)

class mpdata_wrapper:
    def __init__(self, advector, advectee, shape, options):
        self.options = options
#         self.stepper = Stepper(options=options, grid=(ny, nx))
        self.stepper = Stepper(options=self.options, grid=shape)
        self.solver = Solver(stepper=self.stepper, advectee=advectee, advector=advector)
        # print('shape: ', advectee.grid)
    
    def __call__(self, advector, advectee):
        """
        advector: arr_shape@(ny + 1, nx), arr_shape@(ny, nx + 1)
        """
        #curr -> advectee
        #g_phys -> advector
        self.solver.advectee.get()[:] = advectee
        if type(advector) is tuple:
            for i, _advector_axis in enumerate(advector):
                self.solver.advector.get_component(i)[:] = _advector_axis
        else:
            self.solver.advector.get_component(0)[:] = advector


        self.solver.advance(nt=1) #1 -> flux movement
        return self.solver.advectee.get() 
        
        

def MPDATA(advector, advectee):
    """
    advector: arr_shape@(ny + 1, nx), arr_shape@(ny, nx + 1)
    """
    solver.curr.get()[:] = advectee
#     solver.GC_phys.get_component(0)[:] = advector[0]
#     solver.GC_phys.get_component(1)[:] = advector[1]
    for i, _advector_axis in enumerate(advector):
        solver.GC_phys.get_component(i)[:] = advector[i]
        
    solver.advance(nt=1) #1 -> flux movement
    return solver.curr.get() 
