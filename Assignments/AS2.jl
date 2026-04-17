using Random, LinearAlgebra, Plots, Parameters, NLsolve

@with_kw mutable struct Params
    β::Float64 = 0.99
    γ::Float64 = 2.0
    α::Float64 = 0.3
    ϕ::Float64 = 0.0           # capital adjustment cost 
    δ::Float64 = NaN           # calibrated in steady_state
    χ::Float64 = NaN           # calibrated in steady_state
end

"""
    steady_state(params; n_target, i_over_y)

Solve for the deterministic SS of the planner problem (a=0, ϕ=0) and
calibrate δ, χ so that i/y = `i_over_y` and n = `n_target`.
Mutates `params.δ` and `params.χ` in place.
"""
function steady_state(params::Params; n_target = 0.3, i_over_y = 0.16)
    @unpack β, γ, α = params

    δ        = i_over_y * (1/β - 1) / (α - i_over_y)
    y_over_k = (1/β - 1 + δ) / α

    n = n_target
    k = n * y_over_k^(-1/(1 - α))
    y = y_over_k * k
    i = δ * k
    c = y - i
    χ = (1 - α) * y / (c * n^(1 + γ))

    params.δ = δ
    params.χ = χ
    return (; k, y, c, i, n, δ, χ)
end

function transition_residuals!(F, x, p::Params, k0, ss, T)
    @unpack α, β, δ, χ, γ = p

    kpath = @view x[1:T]         # k_1,…,k_T
    npath = @view x[T+1:2T]      # n_0,…,n_{T-1}

    k = vcat(k0, kpath, ss.k)    # length T+2, k_0..k_{T+1}=k̄
    n = vcat(npath, ss.n)        # length T+1, n_0..n_T=n̄

    c = [k[t]^α*n[t]^(1-α) + (1-δ)*k[t] - k[t+1] for t in 1:T+1]

    @inbounds for t in 1:T
        F[t]   = χ*n[t]^γ*c[t] - (1-α)*k[t]^α*n[t]^(-α)           # labor FOC
        R_tp1  = α*k[t+1]^(α-1)*n[t+1]^(1-α) + 1 - δ
        F[T+t] = 1/c[t] - β*R_tp1/c[t+1]                          # Euler
    end
    return F
end

"""
    transition_path(params; k0, T=200)

Solve for the perfect-foresight transition from `k0` to steady state,
using a T-period truncation.  Returns NamedTuple of length-(T+1) paths
for (k, n, c, i, y).
"""
function transition_path(params::Params; k0::Float64, T::Int = 200)
    ss = steady_state(params)

    kpath0 = collect(range(k0, ss.k, length = T+2))[2:T+1]   # linear guess for k_1..k_T
    npath0 = fill(ss.n, T)                                    # SS guess for labor
    x0     = vcat(kpath0, npath0)

    result = nlsolve((F,x) -> transition_residuals!(F, x, params, k0, ss, T),
                     x0; autodiff = :forward, ftol = 1e-10)
    converged(result) || error("Transition did not converge")

    x     = result.zero
    k     = vcat(k0, x[1:T], ss.k)
    n     = vcat(x[T+1:2T], ss.n)
    @unpack α, δ = params
    y     = [k[t]^α*n[t]^(1-α)              for t in 1:T+1]
    c     = [y[t] + (1-δ)*k[t] - k[t+1]     for t in 1:T+1]
    inv   = k[2:T+2] .- (1-δ).*k[1:T+1]

    return (k = k[1:T+1], n = n, c = c, i = inv, y = y)
end


p  = Params()
ss = steady_state(p)
tp = transition_path(p; k0 = 0.5*ss.k, T = 200)
plot(0:200, [tp.k tp.n tp.c tp.y], label = ["k" "n" "c" "y"], layout = 4)