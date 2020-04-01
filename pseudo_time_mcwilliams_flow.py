import os
import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.extras import plot_tools

import logging
logger = logging.getLogger(__name__)

from mcwilliams_preconditioner import solve_mcwilliams_preconditioner_problem

# We want
#
# (f L / N H) = 1 
#
# => N = H / f L
#
# and
#
# US / f * L = 1

nx = 64
nz = 16

lx = 1/4
ly = 1/8
f = 1
h = 0.1

preconditioner = solve_mcwilliams_preconditioner_problem(nx=nx, nz=nz, h=h, lx=lx, ly=ly, f=f)
domain = preconditioner.domain
problem = de.IVP(domain, variables=['ψ', 'ψz'])

problem.parameters['h'] = h
problem.parameters['lx'] = lx
problem.parameters['ly'] = ly
problem.parameters['f'] = f
problem.parameters['N'] = h / (f * lx)
problem.parameters['US'] = 0.5 * f * lx

problem.substitutions['uS'] = "               US * exp(z/h - x**2 / (2 * lx**2) - y**2 / (2 * ly**2))"
problem.substitutions['uSy'] = "- y / ly**2 * US * exp(z/h - x**2 / (2 * lx**2) - y**2 / (2 * ly**2))"
problem.substitutions['uSz'] = "  1 / h     * US * exp(z/h - x**2 / (2 * lx**2) - y**2 / (2 * ly**2))"

problem.add_equation(
        "dt(ψ) - dx(dx(ψ)) - dy(dy(ψ)) - f**2 / N**2 * dz(ψz) = f * uSz / N**2 * (dy(ψz) + dy(ψ) / h) + uSy",
        condition="(nx != 0) or (ny != 0)")

problem.add_equation("ψ = 0", condition="(nx == 0) and (ny == 0)")
problem.add_equation("ψz = 0", condition="(nx == 0) and (ny == 0)")
problem.add_equation("dz(ψ) - ψz = 0", condition="(nx != 0) or (ny != 0)")

problem.add_bc("right(f * ψz) = right(uSz * dy(ψ))", condition="(nx != 0) or (ny != 0)")
problem.add_bc("left(ψz) = 0", condition="(nx != 0) or (ny != 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.SBDF3)

solver.state['ψ']['g'] = preconditioner.state['ψ']['g']
solver.state['ψz']['g'] = preconditioner.state['ψz']['g']

ψ = solver.state['ψ']
ψ.require_grid_space()
ψ0 = ψ['g'].copy()
dψ = np.inf

u_op = - de.operators.differentiate(ψ, y=1)
v_op =   de.operators.differentiate(ψ, x=1)

dt = 1e-2 * 1 / max(nx, nz)**2

analysis = solver.evaluator.add_file_handler('mcwilliams_flow', sim_dt=1000*dt)
analysis.add_task('ψ')
analysis.add_task('-dy(ψ)')
analysis.add_task('dx(ψ)')

scale = 4

fig, axs = plt.subplots(ncols=3, figsize=(24, 8))

for i in range(1000):
    solver.step(dt)

    if (solver.iteration-1) % 10 == 0:

        # Plot solution
        ψ = solver.state['ψ']
        ψ.require_grid_space()

        u = u_op.evaluate()
        u.set_scales(scale)
        u.require_grid_space()

        v = v_op.evaluate()
        v.set_scales(scale)
        v.require_grid_space()

        plt.sca(axs[0]) 
        plt.cla()
        plot_tools.plot_bot_3d(u, "z", scale*nz-1, axes=axs[0], even_scale=True)

        plt.sca(axs[1]) 
        plt.cla()
        plot_tools.plot_bot_3d(v, "z", scale*nz-1, axes=axs[1], even_scale=True)

        plt.sca(axs[2]) 
        plt.cla()
        plot_tools.plot_bot_3d(u, "y", int(scale*nx/2), axes=axs[2], even_scale=True)

        plt.pause(0.1)

        # Save solution
        plt.savefig('mcwilliams_flow.png', dpi=480)

        # Compute change
        dψ = np.sum(np.abs(ψ0 - ψ['g'])) / np.sum(np.abs(ψ['g']))
        ψ0 = ψ['g'].copy()

        logger.info('Iteration: %i, Time: %e, dt: %e, dψ: %e' %(solver.iteration, solver.sim_time, dt, dψ))
