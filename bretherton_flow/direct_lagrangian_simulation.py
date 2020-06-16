import numpy as np
from dedalus import public as de

class LagrangianEquations:
    def __init__(self,
                 na = 128,
                 nb = 32,
                  g = 1.0,
                  k = 10.0,
                  h = 1.0,
                  L = 20.0,
                  A = 1e-4,
                  surface_pressure = "2 * A / g * cos(k * a - σ * t) * exp(-a**2 / 2)",
                  timestepper = 'RK222'
                ):

        self.na = na
        self.nb = nb

        self.g = k
        self.k = k
        self.σ = np.sqrt(g * k * np.tanh(k * h))
        self.A = A
        self.h = h
        self.L = L

        a_basis = de.Fourier('a', na, interval=(-L/2, 2 * L), dealias=3/2)
        b_basis = de.Chebyshev('b', nb, interval=(-h, 0), dealias=3/2)

        self.domain = domain = de.Domain([a_basis, b_basis], grid_dtype=np.float64)

        self.problem = problem = de.IVP(domain, variables=['x', 'y', 'u', 'v', 'p'])

        problem.parameters['k'] = k
        problem.parameters['g'] = g
        problem.parameters['A'] = A
        problem.parameters['σ'] = self.σ

        ## Jacobian
        problem.substitutions['J(r, s)'] = "da(r) * db(s) - db(r) * da(s)"

        ## Identities
        problem.add_equation("dt(x) - u = 0")
        problem.add_equation("dt(y) - v = 0")

        ## Mass
        problem.add_equation("da(x) + db(y) = - J(x, y)")

        ## Momentum
        problem.add_equation("dt(u) + da(p) + g * da(y) = - J(p, y)")
        problem.add_equation("dt(v) + db(p) - g * da(x) = - J(x, p)")

        problem.add_bc("left(y) = 0")
        problem.add_bc(f"right(p) = {surface_pressure}")

        timestepper = getattr(de.timesteppers, timestepper)
        self.solver = solver = problem.build_solver(timestepper)

        self.x = solver.state['x']
        self.y = solver.state['y']
        self.u = solver.state['u']
        self.v = solver.state['v']
        self.p = solver.state['p']

        self.initialize_wave_mean_decomposition()

    def run(self, stop_time, dt=None):

        if dt is None:
            dt = 0.001 * 2 * np.pi / self.σ

        solver = self.solver
        solver.stop_iteration = int(np.round(stop_time / dt))

        while solver.ok:
            solver.step(dt)

            if solver.iteration % 100 == 0:
                t = solver.sim_time
                print("iteration: ", solver.iteration,
                      "time: {:.2f} periods".format(self.σ * t / (2  * np.pi)))

    def add_analysis(self):
        solver = self.solver
        problem = self.problem

        σ = problem.parameters['σ']
        T = 2 * np.pi / σ

        analysis = solver.evaluator.add_file_handler('analysis', sim_dt=T)
        analysis.add_task('x')
        analysis.add_task('y')
        analysis.add_task('u')
        analysis.add_task('v')
        analysis.add_task('p')

    def initialize_wave_mean_decomposition(self):
        self.ka = self.domain.elements(0)
        self.kb = self.domain.elements(1)

        self.filter = self.ka < self.k / 2

        self.mean_x = self.domain.new_field(name='mean_x')
        self.mean_y = self.domain.new_field(name='mean_y')
        self.mean_u = self.domain.new_field(name='mean_u')
        self.mean_v = self.domain.new_field(name='mean_v')
        self.mean_p = self.domain.new_field(name='mean_p')

        self.wave_x = self.domain.new_field(name='wave_x')
        self.wave_y = self.domain.new_field(name='wave_y')
        self.wave_u = self.domain.new_field(name='wave_u')
        self.wave_v = self.domain.new_field(name='wave_v')
        self.wave_p = self.domain.new_field(name='wave_p')

    def wave_mean_decomposition(self, scale=1):

        solver = self.solver
        domain = self.domain
        problem = self.problem

        self.x.set_scales(scale)
        self.y.set_scales(scale)
        self.u.set_scales(scale)
        self.v.set_scales(scale)
        self.p.set_scales(scale)

        self.mean_x.set_scales(scale)
        self.mean_y.set_scales(scale)
        self.mean_u.set_scales(scale)
        self.mean_v.set_scales(scale)
        self.mean_p.set_scales(scale)

        self.wave_x.set_scales(scale)
        self.wave_y.set_scales(scale)
        self.wave_u.set_scales(scale)
        self.wave_v.set_scales(scale)
        self.wave_p.set_scales(scale)

        self.mean_x['c'] = self.x['c'] * self.filter
        self.mean_y['c'] = self.y['c'] * self.filter
        self.mean_u['c'] = self.u['c'] * self.filter
        self.mean_v['c'] = self.v['c'] * self.filter
        self.mean_p['c'] = self.p['c'] * self.filter

        self.wave_x['c'] = self.x['c'] - self.mean_x['c']
        self.wave_y['c'] = self.y['c'] - self.mean_y['c']
        self.wave_u['c'] = self.u['c'] - self.mean_u['c']
        self.wave_v['c'] = self.v['c'] - self.mean_v['c']
        self.wave_p['c'] = self.p['c'] - self.mean_p['c']
