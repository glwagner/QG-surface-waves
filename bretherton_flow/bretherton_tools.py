import numpy as np

from dedalus import public as de
from dedalus.extras import plot_tools

def create_domain(nx=32, nz=32, L=20, H=10):

    # Create bases and domain
    x_basis = de.Fourier('x',   nx, interval=(-L/2, L/2))
    y_basis = de.Fourier('y',   nx, interval=(-L/2, L/2))
    z_basis = de.Chebyshev('z', nz, interval=(-H, 0))
    
    return de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)

def make_solver(domain, h=1, lx=2, ly=1, f=1, N=20, US=1):
    problem = de.LBVP(domain, variables=['ψ', 'ψz'])

    problem.parameters["h"] = h
    problem.parameters["lx"] = lx
    problem.parameters["ly"] = ly
    problem.parameters["f"] = f
    problem.parameters["N"] = N
    problem.parameters["US"] = US

    problem.substitutions["uSy"] = "- y / ly**2 * US * exp(z/h - x**2 / (2 * lx**2) - y**2 / (2 * ly**2))"

    problem.add_equation("dx(dx(ψ)) + dy(dy(ψ)) + f**2 / N**2 * dz(ψz) = - uSy",
                         condition="(nx != 0) or (ny != 0)")

    problem.add_equation("ψ = 0", condition="(nx == 0) and (ny == 0)")
    problem.add_equation("ψz = 0", condition="(nx == 0) and (ny == 0)")
    problem.add_equation("dz(ψ) - ψz = 0", condition="(nx != 0) or (ny != 0)")

    problem.add_bc("left(ψz) = 0", condition="(nx != 0) or (ny != 0)")
    problem.add_bc("right(ψz) = 0", condition="(nx != 0) or (ny != 0)")

    # Build solver
    return problem.build_solver()

def x_velocities(solver):
    problem = solver.problem
    domain = solver.domain

    ψ = solver.state['ψ']
    u_op = - de.operators.differentiate(ψ, y=1)
    u = u_op.evaluate()
    u.require_grid_space()

    x = domain.grid(0)
    y = domain.grid(1)
    z = domain.grid(2)

    uS = domain.new_field(name='uS')
    uM = domain.new_field(name='uM')

    US = problem.parameters['US']
    lx = problem.parameters['lx']
    ly = problem.parameters['ly']
    h = problem.parameters['h']

    uS['g'] = US * np.exp(z/h - x**2 / (2 * lx**2) - y**2 / (2 * ly**2))
    uM['g'] = uS['g'] - u['g']

    return u, uS, uM

def surface_grid(solver, scale=4):
    domain = solver.domain

    x = domain.grid(0)
    y = domain.grid(1)
    z = domain.grid(2)
    
    X, Y = np.meshgrid(x, y)

    return X, Y

def meridional_grid(solver, scale=4):
    domain = solver.domain

    x = domain.grid(0)
    y = domain.grid(1)
    z = domain.grid(2)
    
    Y, Z = np.meshgrid(y, z)

    return Y, Z

def interpolated_x_velocities(solver, scale=4, **slice):

    uL, uS, uM = x_velocities(solver)
    ψL = solver.state['ψ']

    ψL_interp_op = de.operators.interpolate(ψL, **slice)
    uL_interp_op = de.operators.interpolate(uL, **slice)
    uM_interp_op = de.operators.interpolate(uM, **slice)
    uS_interp_op = de.operators.interpolate(uS, **slice)

    ψL_interp = ψL_interp_op.evaluate()
    uL_interp = uL_interp_op.evaluate()
    uM_interp = uM_interp_op.evaluate()
    uS_interp = uS_interp_op.evaluate()

    ψL_interp.set_scales(scale)
    uL_interp.set_scales(scale)
    uM_interp.set_scales(scale)
    uS_interp.set_scales(scale)

    return ψL, uL_interp, uM_interp, uS_interp

