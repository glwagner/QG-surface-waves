import os
import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.extras import plot_tools

from mcwilliams_preconditioner import solve_mcwilliams_preconditioner_problem

n = 32
h = 0.1
lx = 1/2
ly = 1/4
f = 1

# Create bases and domain
x_basis = de.Fourier('x', n, interval=(-np.pi, np.pi))
y_basis = de.Fourier('y', n, interval=(-np.pi, np.pi))
z_basis = de.Chebyshev('z', n, interval=(-1, 0))
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)

problem = de.IVP(domain, variables=['ψ', 'ψz'])

# We want
#
# (f L / N H) = 1 
#
# => N = H / f L
#
# and
#
# US / f * L = 1

problem.parameters['h'] = h
problem.parameters['lx'] = lx
problem.parameters['ly'] = ly
problem.parameters['f'] = f
problem.parameters['N'] = h / (f * lx)
problem.parameters['US'] = f * lx

problem.substitutions['uS'] = "US * exp(z/h - x**2 / (2 * lx**2) - y**2 / (2 * ly**2))"
problem.substitutions['uSy'] = "- y * US / ly**2 * exp(z/h - x**2 / (2 * lx**2) - y**2 / (2 * ly**2))"
problem.substitutions['uSz'] = "US / h * exp(z/h - x**2 / (2 * lx**2) - y**2 / (2 * ly**2))"

problem.add_equation("dt(ψ) - dx(dx(ψ)) - dy(dy(ψ)) - f**2 / N**2 * dz(ψz) = - f * uS / (N**2 * h) * dy(ψz) - f * uS / (N**2 * h**2) * dy(ψ) - uSy",
                     condition="(nx != 0) or (ny != 0)")

problem.add_equation("ψ = 0", condition="(nx == 0) and (ny == 0)")
problem.add_equation("ψz = 0", condition="(nx == 0) and (ny == 0)")
problem.add_equation("dz(ψ) - ψz = 0", condition="(nx != 0) or (ny != 0)")

problem.add_bc("left(ψz) = 0", condition="(nx != 0) or (ny != 0)")
problem.add_bc("right(f * ψz) = right(uSz * dy(ψ))", condition="(nx != 0) or (ny != 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.RK443)

preconditioner = solve_mcwilliams_preconditioner_problem(n=n, h=h, lx=lx, ly=ly, f=f)

solver.state['ψ']['g'] = preconditioner.state['ψ']['g']
solver.state['ψz']['g'] = preconditioner.state['ψz']['g']

dt = 1e-6

for i in range(10):
    solver.step(dt)
