import numpy as np
from mpi4py import MPI

from dedalus.extras.flow_tools import CFL
from dedalus.core.future import FutureField

from utils import add_first_derivative_substitutions, add_parameters

nx = 32
ny = 32
nz = 32
Lx = 1.0
Ly = 1.0
Lz = 1.0
f = 0.0
κ = 1.0
ν = 1.0
Nsq = 0.0

    problem.substitutions['uS'] = 
    problem.substitutions['uSy'] = "- y / ly**2 * US * exp(z/h - x**2 / (2 * lx**2) - y**2 / (2 * ly**2))"
    problem.substitutions['uSz'] = "  1 / h     * US * exp(z/h - x**2 / (2 * lx**2) - y**2 / (2 * ly**2))"

substitutions = {
        'envelope' : "exp(z/h - (x - cx*t)**2 / (2 * lx**2) - (y - cy*t)**2 / (2 * ly**2))"
           'uS' : "US * envelope"
           'vS' : "VS * envelope"
        'dz_uS' : "US / h * envelope"
        'dz_vS' : "VS / h * envelope"
        'dt_uS' : "US / h * envelope"
        'dt_vS' : "VS / h * envelope"
        }

# Create bases and domain
self.xbasis = xbasis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
self.ybasis = ybasis = de.Fourier('y', ny, interval=(0, Ly), dealias=3/2)
self.zbasis = zbasis = de.Chebyshev('z', nz, interval=(-Lz, 0), dealias=3/2)
self.domain = domain = de.Domain([xbasis, ybasis, zbasis], grid_dtype=np.float64)

variables = ['p', 'b', 'u', 'v', 'w', 'bz', 'uz', 'vz', 'wz']
self.problem = problem = de.IVP(domain, variables=variables, time='t')

add_parameters(problem, f=f, ν=ν, κ=κ, Nsq=Nsq, h=h, cx=cx, cy=cy, lx=lx, ly=ly, US=US, VS=VS)
add_first_derivative_substitutions(problem, ['u', 'v', 'w', 'b'], ['x', 'y'])
add_substitutions(problem, substitutions)

# Equations
problem.add_equation(f"dt(u) - ν * (dx(ux) + dy(uy) + dz(uz)) + dx(p) - f*v = - u*ux - v*uy - w*uz + dt_uS")
problem.add_equation(f"dt(v) - ν * (dx(vx) + dy(vy) + dz(vz)) + dy(p) + f*u = - u*vx - v*vy - w*vz + dt_vS")

# Hydrostatic
problem.add_equation(f"dz(p) - b = - u * dz_uS - v * dz_vS")

problem.add_equation(f"dt(b) - κ * (dx(bx) + dy(by) + dz(bz)) + Nsq*w = - u*bx - v*by - w*bz")

problem.add_equation("ux + vy + wz = 0")

# First-order equivalencies
problem.add_equation("bz - dz(b) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("vz - dz(v) = 0")
problem.add_equation("wz - dz(w) = 0")

# Pressure gauge condition
problem.add_bc("left(p) = 0", condition="(nx == 0) and (ny == 0)")

problem.add_bc("right(uz) = 0", condition="(nx == 0) and (ny == 0)")
problem.add_bc("left(uz) = 0", condition="(nx == 0) and (ny == 0)")

problem.add_bc("right(vz) = 0", condition="(nx == 0) and (ny == 0)")
problem.add_bc("left(vz) = 0", condition="(nx == 0) and (ny == 0)")

problem.add_bc("right(bz) = 0", condition="(nx == 0) and (ny == 0)")
problem.add_bc("left(bz) = 0", condition="(nx == 0) and (ny == 0)")
