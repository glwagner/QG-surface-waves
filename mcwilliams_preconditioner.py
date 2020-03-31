import os
import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.extras import plot_tools


def solve_mcwilliams_preconditioner_problem(
        buoyancy_bc="right(ψz) = 0",
         h = 0.1,
        lx = 1/2,
        ly = 1/4,
         f = 1,
         n = 32,
        ):

    # Create bases and domain
    x_basis = de.Fourier('x', n, interval=(-np.pi, np.pi))
    y_basis = de.Fourier('y', n, interval=(-np.pi, np.pi))
    z_basis = de.Chebyshev('z', n, interval=(-1, 0))
    domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)

    # Poisson equation
    problem = de.LBVP(domain, variables=['ψ', 'ψz'])
    
    problem.parameters['h'] = h
    problem.parameters['lx'] = lx
    problem.parameters['ly'] = ly
    problem.parameters['f'] = f
    problem.parameters['N'] = h / (f * lx)
    problem.parameters['US'] = f * lx

    problem.substitutions['uSy'] = "- y * US / ly**2 * exp(z/h - x**2 / (2 * lx**2) - y**2 / (2 * ly**2))"
    problem.substitutions['uSz'] = "US / h * exp(z/h - x**2 / (2 * lx**2) - y**2 / (2 * ly**2))"

    problem.add_equation("dx(dx(ψ)) + dy(dy(ψ)) + f**2 / N**2 * dz(ψz) = uSy",
                         condition="(nx != 0) or (ny != 0)")

    problem.add_equation("ψ = 0", condition="(nx == 0) and (ny == 0)")
    problem.add_equation("ψz = 0", condition="(nx == 0) and (ny == 0)")
    problem.add_equation("dz(ψ) - ψz = 0", condition="(nx != 0) or (ny != 0)")

    problem.add_bc("left(ψz) = 0", condition="(nx != 0) or (ny != 0)")
    problem.add_bc(buoyancy_bc, condition="(nx != 0) or (ny != 0)")

    # Build solver
    solver = problem.build_solver()
    solver.solve()

    return solver
