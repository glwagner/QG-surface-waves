module WaveTanks

export
    WaveTank,
    Grid,
    FourierFields,
    GreensQuadrature,
    inverse_transform!,
    forward_transform!

using FFTW, LinearAlgebra, FastGaussQuadrature

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

struct FourierFields{G, C, F}
        grid :: G
    physical :: C
    spectral :: F

    function FourierFields(grid, names...)
        physical = NamedTuple{names}(Tuple(0 * grid.x        for name in names))
        spectral = NamedTuple{names}(Tuple(0 * grid.k .+ 0im for name in names))
        return new{typeof(grid), typeof(physical), typeof(spectral)}(grid, physical, spectral)
    end
end

function inverse_transform!(fields, name)
    f̂ = fields.spectral[name]
    f = fields.physical[name]
    mul!(f, fields.grid.irfftplan, f̂)
    return nothing
end

function forward_transform!(fields, name)
    f̂ = fields.spectral[name]
    f = fields.physical[name]
    mul!(f̂, fields.grid.rfftplan, f)
    return nothing
end

function inverse_transform!(fields)
    for name in propertynames(fields.physical)
        inverse_transform!(fields, name)
    end
    return nothing
end

function forward_transform!(fields)
    for name in propertynames(fields.physical)
        forward_transform!(fields, name)
    end
    return nothing
end

struct GreensQuadrature{G, K, A}
       grid :: G
     kernel :: K
      nodes :: A
    weights :: A
          τ :: A
         dτ :: A
end

function GreensQuadrature(grid; nnodes)
    nodes, weights = gausslegendre(nnodes)
     τ = deepcopy(nodes)
    dτ = deepcopy(weights)
    kernel = zeros(nnodes) .* reshape(grid.k .+ 0im, 1, length(grid.k)) # nweights, 
    return GreensQuadrature(grid, kernel, nodes, weights, τ, dτ)
end

end # module
