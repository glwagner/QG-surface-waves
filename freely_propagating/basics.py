import numpy as np
from dedalus import public as de

class GaussianPacket():
    """Contains surface wave packet parameters."""
    def __init__(self, wavenumber=1, depth=1, length=10, g=1):
        self.wavenumber = k = wavenumber
        self.depth = h = depth

        self.frequency = sigma = np.sqrt(g * k * np.tanh(k * h))

        self.group_velocity = cg = g / (2 * sigma) * (k * h * np.sech(k * h)**2 + np.tanh(k * h))
        self.S = g * k**2 / sigma**2 * cg * (1 + np.tanh(k * h)**2)

    def a(self, x):
        return np.exp(-x**2 / (2 * self.length**2)

class McIntyreProblem():
    """The circulation-round-a-packet problem discussed by McIntyre (1981)"""
    def __init__(self, wave_packet, L=100, nx=32, nz=32):
        self.wave_packet = wave_packet
        self.h = h = wave_packet.depth

        self.xbasis = xbasis = de.Fourier('x', nx, interval=(-L/2, L/2))
        self.ybasis = ybasis = de.Chebyshev('z', nz, interval=(-h, 0))

        self.domain = domain = de.Domain([xbasis, zbasis], grid_dtype=np.float64)

        self.problem = problem = de.LBVP(domain, variables=['Φ', 'Φz'])

        problem.parameters["k"] = wave_packet.wavenumber
        problem.parameters["h"] = h
        problem.parameters["S"] = wave_packet.S
        problem.parameters["l"] = wave_packet.length
        
        problem.substitutions["a_squared"] = "exp(-x**2 / l**2)"
        
        problem.add_equation("dx(dx(Φ)) + dz(Φz) = 0", condition="(nx != 0)")
        
        problem.add_equation("Φ = 0", condition="(nx == 0)")
        problem.add_equation("Φz = 0", condition="(nx == 0)")
        problem.add_equation("dz(Φ) - Φz = 0", condition="(nx != 0)")
        
        problem.add_bc("left(Φz) = 0", condition="(nx != 0)")
        problem.add_bc("right(Φz) = S * dx(a_squared) / 2", condition="(nx != 0)")

        self.solver = solver = problem.build_solver()
        solver.solve()

        self.streamfunction_problem = streamfunction_problem = de.LBVP(domain, variables=['ψ', 'ψz'])

        Φz = solver.state['Φz']

        streamfunction_problem.parameters["Φz"] = Φz

        streamfunction_problem.add_equation("dx(dx(ψ)) + dz(ψz) = 0", condition="(nx != 0)")

        streamfunction_problem.add_equation("ψ = 0", condition="(nx == 0)")
        streamfunction_problem.add_equation("ψz = 0", condition="(nx == 0)")
        streamfunction_problem.add_equation("dz(ψ) - ψz = 0", condition="(nx != 0)")

        streamfunction_problem.add_bc("left(dx(ψ)) = 0", condition="(nx != 0)")
        streamfunction_problem.add_bc("right(dx(ψ)) = interp(Φz, z=0)", condition="(nx != 0)")

        # Build solver
        self.streamfunction_solver = streamfunction_problem.build_solver()
        self.streamfunction_solver.solve()
