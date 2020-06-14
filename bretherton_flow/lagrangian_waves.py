import os
import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.extras import plot_tools
from dedalus.extras.plot_tools import plot_bot_2d

# Wavenumber
g = 1.0 # non-dimensional for now
k = 5

h = 1
L = 40
ϵ = 0.001

σ = np.sqrt(g * k * np.tanh(k * h))

a_basis = de.Fourier('a',    128, interval=(-L/4, L))
b_basis = de.Chebyshev('b',   32, interval=(-h, 0))

domain = de.Domain([a_basis, b_basis], grid_dtype=np.complex128)

problem = de.IVP(domain, variables=['xw', 'yw', 'uw', 'vw', 'pw']) #,
                                    #'xm', 'ym', 'um', 'vm', 'pm'])

problem.parameters["k"] = k
problem.parameters["g"] = g
problem.parameters["σ"] = σ
problem.parameters["ep"] = ϵ

problem.substitutions["J1(r, s)"] = (
    "   ( da(conj(r)) - 1j * k * conj(r) ) * db(s)       " +
    " - ( da(s)       + 1j * k * s       ) * db(conj(r)) "
)

problem.substitutions["J2(r, s)"] = (
    " - ( da(conj(s)) - 1j * k * conj(s) ) * db(r)       " +
    " + ( da(r)       + 1j * k * r       ) * db(conj(s)) "
)

problem.substitutions["J(r, s)"] = "J1(r, s) + J2(r, s)"

# Waves

## Identities
problem.add_equation("dt(xw) - uw = 0")
problem.add_equation("dt(yw) - vw = 0")

## Mass conservation to ϵ^2
problem.add_equation("da(xw) + 1j * k * xw + db(yw) = 0")

## Momentum conservation
problem.add_equation("dt(uw) - 1j * σ * uw + da(pw) + 1j * k * pw = 0")
problem.add_equation("dt(vw) - 1j * σ * vw + db(pw) + g * db(yw)  = 0")

problem.add_bc("left(yw) = 0")
problem.add_bc("right(pw) = 0 * g * ep * exp(-a**2 / 2) * exp(- σ * t / 10)")

# Mean

## Identities
#problem.add_equation("dt(xm) - um = 0")
#problem.add_equation("dt(vm) - vm = 0")

## Mass conservation to ϵ^2
#problem.add_equation("da(xm) + db(ym) = - J(xw, yw)")

## Momentum conservation
#problem.add_equation("dt(um) + da(pm)              = -J(pw, yw)")
#problem.add_equation("dt(vm) + db(pm) - g * da(xm) = -J(xw, pw)")

#problem.add_bc("left(vm) = 0")
#problem.add_bc("right(pm) = 0")

timestepper = de.timesteppers.RK443
solver = problem.build_solver(timestepper)

x = solver.state['xw']
y = solver.state['yw']
u = solver.state['uw']
v = solver.state['vw']

a = domain.grid(0)
b = domain.grid(1)
x['g'] = np.exp(-a**2 / 2 + k * b)
y['g'] = - 1j * x['g']

u['g'] = σ * y['g']
v['g'] = - σ * x['g']

solver.stop_iteration = 100000
dt = 0.001 * 2 * np.pi / σ

iters_per_period = int(np.round((2 * np.pi / σ) / dt))

scale = 10
a = domain.grid(0, scales=scale)
b = domain.grid(1, scales=scale)
A, B = np.meshgrid(a, b)

while solver.ok:
    solver.step(dt)

    if solver.iteration % iters_per_period == 0:
        print("iteration: ", solver.iteration)
        t = solver.sim_time
        print("time: ", σ * t / (2 * np.pi))
        #U = np.abs(solver.state['uw']['g'])
        solver.state['yw'].set_scales(scale)
        solver.state['pw'].set_scales(scale)
        P = np.abs(solver.state['pw']['g'])
        Y = np.abs(solver.state['yw']['g'])

        y = np.exp(1j * k * a - 1j * σ * t) * solver.state['yw']['g']
        y += np.conj(y)
        y = np.real(y)

        #plt.contourf(A, B, Y.T)
        η = np.real(Y[:, -1])

        plt.cla()
        plt.plot(a, η)
        plt.pause(0.1)
