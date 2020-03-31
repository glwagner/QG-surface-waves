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


# Create bases and domain
x_basis = de.Fourier('x', 64, interval=(-np.pi, np.pi))
y_basis = de.Fourier('y', 64, interval=(-np.pi, np.pi))
z_basis = de.Chebyshev('z', 64, interval=(-1, 0))
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)

# Poisson equation
problem = de.LBVP(domain, variables=['ψ', 'ψz'])

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

problem.parameters["h"] = h
problem.parameters["lx"] = lx
problem.parameters["ly"] = ly
problem.parameters["f"] = f
problem.parameters["N"] = h / (f * lx)
problem.parameters["US"] = f * lx

#problem.substitutions["uS(x, y, z)"] = "US * exp(z/h - x**2 / (2 * lx**2) - y**2 / (2 * ly**2))"
#problem.substitutions["uSz"] = "dz(uS)"
problem.substitutions["uSy"] = "- y * US / ly**2 * exp(z/h - x**2 / (2 * lx**2) - y**2 / (2 * ly**2))"
problem.substitutions["uSz"] = "US / h * exp(z/h - x**2 / (2 * lx**2) - y**2 / (2 * ly**2))"

problem.add_equation("dx(dx(ψ)) + dy(dy(ψ)) + f**2 / N**2 * dz(ψz) = uSy",
                     condition="(nx != 0) or (ny != 0)")

problem.add_equation("ψ = 0", condition="(nx == 0) and (ny == 0)")
problem.add_equation("ψz = 0", condition="(nx == 0) and (ny == 0)")
problem.add_equation("dz(ψ) - ψz = 0", condition="(nx != 0) or (ny != 0)")

problem.add_bc("left(ψz) = 0", condition="(nx != 0) or (ny != 0)")
problem.add_bc("right(f * ψz) = uSz * dy(ψ)", condition="(nx != 0) or (ny != 0)")

# Build solver
solver = problem.build_solver()
solver.solve()

# Plot solution
ψ = solver.state['ψ']
ψ.require_grid_space()
plot_tools.plot_bot_3d(ψ, "z", 63)
plt.savefig('wave_induced_flow_near_surface.png')
