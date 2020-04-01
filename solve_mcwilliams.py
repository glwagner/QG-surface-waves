import os
import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.extras import plot_tools

import logging
logger = logging.getLogger(__name__)

from mcwilliams_problem import simple_domain, mcwilliams_preconditioner, mcwilliams_problem, make_plot

nx = 32
nz = 32

lx = 1/4
ly = 1/8
f = 1
h = 0.1
A = 0.1

US = A * f * lx

domain = simple_domain(nx, nz)

preconditioner = mcwilliams_preconditioner(domain, h=h, lx=lx, ly=ly, f=f, US=US)
preconditioner.solve()

solver = mcwilliams_problem(domain, h=h, lx=lx, ly=ly, f=f, US=US)

solver.state['ψ']['g'] = preconditioner.state['ψ']['g']
solver.state['ψz']['g'] = preconditioner.state['ψz']['g']

ψ = solver.state['ψ']
ψ.require_grid_space()
ψ0 = ψ['g'].copy()
dψ = np.inf

u_op = - de.operators.differentiate(ψ, y=1)
v_op =   de.operators.differentiate(ψ, x=1)

dt = 1e-1 / max(nx, nz)**2

analysis = solver.evaluator.add_file_handler('mcwilliams_flow', sim_dt=1000*dt)
analysis.add_task('ψ', layout='g', name='ψ')
analysis.add_task('-dy(ψ)', layout='g', name='u')
analysis.add_task('dx(ψ)', layout='g', name='v')

scale = 4

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
make_plot(axs, solver, u_op, v_op)
plt.savefig('bretherton_flow.png', dpi=480)

for i in range(10000):

    solver.step(dt)

    if (solver.iteration-1) % 100 == 0:

        make_plot(axs, solver, u_op, v_op)

        # Save solution
        plt.savefig('mcwilliams_flow_{}.png'.format(solver.iteration), dpi=480)
        
        # Compute change
        dψ = np.sum(np.abs(ψ0 - ψ['g'])) / np.sum(np.abs(ψ['g']))
        ψ0 = ψ['g'].copy()

        logger.info('Iteration: %i, Time: %e, dt: %e, dψ: %e' %(solver.iteration, solver.sim_time, dt, dψ))

