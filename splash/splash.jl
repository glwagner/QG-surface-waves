using FFTW, PyPlot, LinearAlgebra

struct WaveTank{T}
    g :: T
    h :: T
end

WaveTank(T=Float64; h, g=9.81) = WaveTank{T}(g, h)

struct Grid{T, K, X, F, B}
            n :: Int
            L :: T
            x :: X
            k :: K
     rfftplan :: F
    irfftplan :: B
end

function Grid(T=Float64; n, L)
    L = T(L)
    k = collect(rfftfreq(n, 2π/L * n))
    dx = L / n
    x = collect(range(-L/2, stop=L/2-dx, length=n))

     rfftplan = plan_rfft(0 * x, flags=FFTW.MEASURE)
    irfftplan = plan_irfft(k .+ 0im, n, flags=FFTW.MEASURE)

    return Grid(n, L, x, k, rfftplan, irfftplan)
end

struct Pebble{P, PH}
    Π :: P      # splash distribution in physical space
    Π̂ :: PH     # splash distribution in fourier space
end

function Pebble(grid; Π)
    Π = Π.(grid.x)
    Π̂ = rfft(Π)
    return Pebble(Π, Π̂)
end

struct Splash{G, T, P, S, SH}
       grid :: G
       tank :: T
     pebble :: P
          s :: S
          ŝ :: SH
end

function Splash(grid, tank, pebble)
    ŝ = 0 * grid.k .+ 0im
    s = 0 * grid.x
    return Splash(grid, tank, pebble, s, ŝ)
end

σ(k, g, h) = sqrt(g * k) * tanh(k * h)

function free_surface!(splash, t)
    pebble, tank, grid = splash.pebble, splash.tank, splash.grid
    g, h = tank.g, tank.h
    @. splash.ŝ = pebble.Π̂ + cos(σ(grid.k, g, h) * t) * pebble.Π̂

    mul!(splash.s, grid.irfftplan, splash.ŝ)

    return nothing
end

tank = WaveTank(h=10)
grid = Grid(n=2^12, L=1000)
pebble = Pebble(grid; Π = x -> exp(-x^2 / 2))
splash = Splash(grid, tank, pebble)

free_surface!(splash, 0)

close("all")
fig, axs = subplots(figsize=(10, 3))

s₀ = deepcopy(splash.s)

dt = 0.1
for i = 1:450
    free_surface!(splash, i * dt)
    cla()
    # plot(grid.x, s₀, "k--")
    plot(grid.x, splash.s, linestyle="-", color="xkcd:indigo", alpha=0.6)
    ylim(-0.5, 1.0)
    pause(0.05)
end
