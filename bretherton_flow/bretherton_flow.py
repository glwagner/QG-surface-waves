"""
Simulation script for 2D Poisson equation.

This script is an example of how to do 2D linear boundary value problems.
It is best ran serially, and produces a plot of the solution using the included
plotting helpers.

On a single process, this should take just a few seconds to run.

"""

import os
import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.extras import plot_tools

# We want
#
# (f L / N H) = 1 
#
# => N = H / f L
#
# and
#
# US / f * L = 1

h = 0.1
lx = 1/2
ly = 1/4
f = 1
nx = 32
nz = 32

# Create bases and domain
x_basis = de.Fourier('x', nx, interval=(-np.pi, np.pi))
y_basis = de.Fourier('y', nx, interval=(-np.pi, np.pi))
z_basis = de.Chebyshev('z', nz, interval=(-1, 0))
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)

# Poisson equation
problem = de.LBVP(domain, variables=['ψ', 'ψz'])

problem.parameters["h"] = h
problem.parameters["lx"] = lx
problem.parameters["ly"] = ly
problem.parameters["f"] = f
problem.parameters["N"] = h / (f * lx)
problem.parameters["US"] = f * lx

problem.substitutions["uS"] = "US * exp(z/h - x**2 / (2 * lx**2) - y**2 / (2 * ly**2))"
#problem.substitutions["uSz"] = "dz(uS)"
problem.substitutions["uSy"] = "- y * US / ly**2 * exp(z/h - x**2 / (2 * lx**2) - y**2 / (2 * ly**2))"

problem.add_equation("dx(dx(ψ)) + dy(dy(ψ)) + f**2 / N**2 * dz(ψz) = - uSy",
                     condition="(nx != 0) or (ny != 0)")

problem.add_equation("ψ = 0", condition="(nx == 0) and (ny == 0)")
problem.add_equation("ψz = 0", condition="(nx == 0) and (ny == 0)")
problem.add_equation("dz(ψ) - ψz = 0", condition="(nx != 0) or (ny != 0)")

problem.add_bc("left(ψz) = 0", condition="(nx != 0) or (ny != 0)")
problem.add_bc("right(ψz) = 0", condition="(nx != 0) or (ny != 0)")

# Build solver
solver = problem.build_solver()
solver.solve()

ψ = solver.state['ψ']
uS = domain.new_field(name='uS')

u_op = - de.operators.differentiate(ψ, y=1)
v_op =   de.operators.differentiate(ψ, x=1)

# Plot solution
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 6))
scale = 4

ψ.set_scales(scale)
ψ.require_grid_space()

u = u_op.evaluate()
u.set_scales(scale)
u.require_grid_space()

v = v_op.evaluate()
v.set_scales(scale)
v.require_grid_space()

x = domain.grid(0, scales=scale)
y = domain.grid(1, scales=scale)
z = domain.grid(2, scales=scale)

X, Y = np.meshgrid(x, y)

u_surface_op = de.operators.interpolate(u, z=0)
v_surface_op = de.operators.interpolate(v, z=0)

u_surface = u_surface_op.evaluate()
u_surface.set_scales(scale)

plt.sca(axs[0, 0]) 
plt.pcolormesh(X, Y, u_surface['g'][:, :, -1])
#plot_tools.plot_bot_3d(u, "z", scale*nz-1, axes=axs[0, 0], even_scale=True)

plt.sca(axs[0, 1]) 
plot_tools.plot_bot_3d(v, "z", scale*nz-1, axes=axs[0, 1], even_scale=True)

plt.sca(axs[1, 0]) 
plot_tools.plot_bot_3d(u, "y", int(scale*nx/2), axes=axs[1, 0], even_scale=True)

plt.sca(axs[1, 1]) 
plot_tools.plot_bot_3d(u, "x", int(scale*nx/2), axes=axs[1, 1], even_scale=True)

plt.pause(0.1)

plt.savefig('qg_wave_induced_flow.png')
