import os
import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.extras import plot_tools

def simple_domain(nx, nz, Lx=2*np.pi, Ly=2*np.pi, Lz=1):

    x_basis = de.Fourier('x', nx, interval=(-Lx/2, Lx/2))
    y_basis = de.Fourier('y', nx, interval=(-Ly/2, Ly/2))
    z_basis = de.Chebyshev('z', nz, interval=(-Lz, 0))

    domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)

    return domain

def mcwilliams_preconditioner(domain, h=0.1, lx=1/4, ly=1/8, f=1, US=1/16):

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

    problem.add_equation("dx(dx(ψ)) + dy(dy(ψ)) + f**2 / N**2 * dz(ψz) = -uSy",
                         condition="(nx != 0) or (ny != 0)")

    problem.add_equation("ψ = 0", condition="(nx == 0) and (ny == 0)")
    problem.add_equation("ψz = 0", condition="(nx == 0) and (ny == 0)")
    problem.add_equation("dz(ψ) - ψz = 0", condition="(nx != 0) or (ny != 0)")

    problem.add_bc("left(ψz) = 0", condition="(nx != 0) or (ny != 0)")
    problem.add_bc("right(ψz) = 0", condition="(nx != 0) or (ny != 0)")

    solver = problem.build_solver()

    return solver


def mcwilliams_problem(domain, h=0.1, lx=1/4, ly=1/8, f=1, US=1/16):
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

    return solver

def make_plot(axs, solver, u_op, v_op, nz=16, nx=32, scale=4):

    # Plot solution
    ψ = solver.state['ψ']
    ψ.require_grid_space()

    u = u_op.evaluate()
    u.set_scales(scale)
    u.require_grid_space()

    v = v_op.evaluate()
    v.set_scales(scale)
    v.require_grid_space()

    plt.sca(axs[0, 0]) 
    plt.cla()
    plot_tools.plot_bot_3d(u, "z", scale*nz-1, axes=axs[0, 0], even_scale=True)

    plt.sca(axs[0, 1]) 
    plt.cla()
    plot_tools.plot_bot_3d(v, "z", scale*nz-1, axes=axs[0, 1], even_scale=True)

    plt.sca(axs[1, 0]) 
    plt.cla()
    plot_tools.plot_bot_3d(u, "y", int(scale*nx/2), axes=axs[1, 0], even_scale=True)

    plt.sca(axs[1, 1]) 
    plt.cla()
    plot_tools.plot_bot_3d(u, "x", int(scale*nx/2), axes=axs[1, 1], even_scale=True)

    plt.pause(0.1)
