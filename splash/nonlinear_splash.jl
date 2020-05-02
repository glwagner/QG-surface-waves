include("WaveTanks.jl")

using PyPlot

using .WaveTanks

function set_forcing!(forcing, Π)
    forcing.physical.Π .= Π.(forcing.grid.x)
    forward_transform!(forcing, :Π)
    @. forcing.spectral.Πx = im * forcing.grid.k * forcing.spectral.Π
    inverse_transform!(forcing, :Πx)
    return nothing
end

σ(k, g, h) = sqrt(g * k) * tanh(k * h)

function calculate_first_order_fields!(fields, forcing, tank, t)
    grid = fields.grid
    g, h = tank.g, tank.h
    k = grid.k

    @. fields.spectral.Φ   = - forcing.spectral.Π * sin(σ(k, g, h) * t) / σ(k, g, h)
    @. fields.spectral.Φt  = - forcing.spectral.Π * cos(σ(k, g, h) * t)
    @. fields.spectral.ϕz  =   forcing.spectral.Π * sin(σ(k, g, h) * t) / g
    @. fields.spectral.ϕzt =   forcing.spectral.Π * σ(k, g, h) * cos(σ(k, g, h) * t) / g
    @. fields.spectral.χzt = - forcing.spectral.Π * σ(k, g, h) * sin(σ(k, g, h) * t) / (g * sinh(k * h)^2)

    @inbounds fields.spectral.χzt[1] = 0
    @inbounds fields.spectral.Φ[1] = 0

    @. fields.spectral.Φx  = im * k * fields.spectral.Φ
    @. fields.spectral.Φxt = im * k * fields.spectral.Φt

    inverse_transform!(fields)

    return nothing
end

function calculate_thorn!(second, first, forcing, tank, t)
    calculate_first_order_fields!(first, forcing, tank, t)

    second.physical.Pt .= @. - (  2 * first.physical.Φx  * first.physical.Φxt
                                + 2 * first.physical.ϕz  * first.physical.ϕzt
                                +     first.physical.χzt * first.physical.Φt   / tank.g
                                -     first.physical.χzt * forcing.physical.Π  / tank.g
                                +     first.physical.Φx  * forcing.physical.Πx)


    forward_transform!(second, :Pt)

    return nothing
end

#G(k, t, g, h) = k == 0 ? zero(typeof(k)) : sin(σ(k, g, h) * t) / σ(k, g, h)
G(k, t, g, h) = cos(σ(k, g, h) * t)

function calculate_second_order_fields!(second, first, forcing, tank, greens, t)
    grid = second.grid
    k = grid.k
    g = tank.g
    h = tank.h

    # Stretch nodes ∈ [-1, 1] and weights to [0, t]
    @. greens.τ = t * (greens.nodes + 1) / 2
    @. greens.dτ = t * greens.weights / 2

    # Calculate integral kernel on time-wavenumber (nt, nk) grid
    for (i, τi) in enumerate(greens.τ)
        calculate_thorn!(second, first, forcing, tank, τi)
        #@. @views greens.kernel[i, :] = G(k, t-τi, g, h) * second.spectral.Pt * greens.dτ[i]
        @. @views greens.kernel[i, :] = G(k, t-τi, g, h) * second.spectral.Pt * greens.dτ[i]
    end

    second.spectral.Φt .= sum(greens.kernel, dims=1)[:]

    calculate_thorn!(second, first, forcing, tank, t)

    inverse_transform!(second)

    return nothing
end

struct Splash{G, T, F, O1, O2, D}
                   grid :: G
                   tank :: T
                forcing :: F
     first_order_fields :: O1
    second_order_fields :: O2
        greens_function :: D
end

function Splash(;
                     n = 2^8,
                     L = 100,
                     h = 10,
                     g = 9.81,
                nnodes = n
               )

    grid = Grid(n=n, L=L)
    forcing = FourierFields(grid, :Π, :Πx)
    greens_function = GreensQuadrature(grid; nnodes=nnodes)
    
     first_order_fields = FourierFields(grid, :Φ, :Φxt, :Φx, :Φt, :ϕz, :ϕzt, :χzt)
    second_order_fields = FourierFields(grid, :Φt, :Pt)
    
    tank = WaveTank(g=g, h=h)

    return Splash(grid, tank, forcing, first_order_fields, second_order_fields, greens_function)
end

calculate_first_order_fields!(splash, t) =
    calculate_first_order_fields!(splash.first_order_fields, splash.forcing, splash.tank, t)

calculate_second_order_fields!(splash, t) =
    calculate_second_order_fields!(splash.second_order_fields, 
                                   splash.first_order_fields, 
                                   splash.forcing, splash.tank, splash.greens_function, t)

function calculate_surface_displacement(splash, t)
    calculate_second_order_fields!(splash, t)
    calculate_first_order_fields!(splash, t)

    s₁ = @. - (splash.first_order_fields.physical.Φt + splash.forcing.physical.Π) / splash.tank.g
    s₂ = @. - (splash.second_order_fields.physical.Φt + s₁ * splash.first_order_fields.physical.ϕzt ) / splash.tank.g

    return s₁, s₂
end

splash = Splash(n=2^10, L=500, h=10)
set_forcing!(splash.forcing, x -> splash.tank.g * exp(-x^2 / 32))

close("all")
fig, axs = subplots(figsize=(20, 6))

dt = 0.05
for i = 1:1500
    t = i * dt
    @time s₁, s₂ = calculate_surface_displacement(splash, t)

    sca(axs); cla()
    plot(splash.grid.x, s₁ .+ s₂, linestyle="-", linewidth=2, color="xkcd:royal blue", label=L"s_1 + s_2", alpha=0.4)
    plot(splash.grid.x, s₁,       linestyle="-", linewidth=1, color="xkcd:dark grey",  label=L"s_1",       alpha=0.6)
    ylim(-1.5, 0.5)
    legend(loc="lower left")


    pause(0.05)
end
