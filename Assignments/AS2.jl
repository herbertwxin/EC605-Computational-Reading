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
    @unpack α, β, δ, χ, γ, ϕ = p

    kpath = @view x[1:T]
    npath = @view x[T+1:2T]

    k = vcat(k0, kpath, ss.k)    # length T+2
    n = vcat(npath, ss.n)        # length T+1

    inv_t = [k[t+1] - (1-δ)*k[t]          for t in 1:T+1]   # i_t
    ψ     = [inv_t[t]/k[t] - δ             for t in 1:T+1]   # i_t/k_t - δ
    c     = [k[t]^α*n[t]^(1-α) - inv_t[t] - ϕ*ψ[t]^2*k[t]  for t in 1:T+1]

    @inbounds for t in 1:T
        # labor FOC: χ n^γ c = (1-α) y / n  (unchanged)
        F[t]   = χ*n[t]^γ*c[t] - (1-α)*k[t]^α*n[t]^(-α)
        # Euler with adjustment costs (reduces to standard when ϕ=0)
        # Envelope: V'(k') = (1/c')[MPK' + (1-δ) + ϕ ψ'(2+ψ')]
        R_tp1  = α*k[t+1]^(α-1)*n[t+1]^(1-α) + (1-δ) + ϕ*ψ[t+1]*(2 + ψ[t+1])
        F[T+t] = (1 + 2ϕ*ψ[t])/c[t] - β*R_tp1/c[t+1]
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

"""
    value_function_iteration(params, kgrid, ngrid, T, ϵ; a)

Solve the planner's Bellman by VFI.  Loops over all (k_i, k'_j, n_l)
triples to find the best (k', n) for each state k.

Arguments
---------
- `kgrid` : capital state/choice grid (length Nk)
- `ngrid` : labor choice grid (length Nn)
- `T`     : maximum number of iterations
- `ϵ`     : sup-norm convergence tolerance
- `a`     : TFP path, length T (default zeros — non-stochastic Q3 case).
            For Q5+ pass the Rouwenhorst grid points as a separate agrid
            and call this once per TFP node (or restructure to 2-D V).
"""
function value_function_iteration(params::Params, kgrid::Vector{Float64}, ngrid::Vector{Float64},
                                   T::Int, ϵ::Float64)
    @unpack α, β, δ, χ, γ, ϕ = params
    Nk = length(kgrid)
    Nn = length(ngrid)

    V     = zeros(Nk)      # value function (over k states)
    V_new = zeros(Nk)
    kpol  = ones(Int, Nk) # index into kgrid for k' policy
    npol  = ones(Int, Nk) # index into ngrid for n policy

    # helpers: no name collision with loop variables
    inv_val(kp, k)      = kp - (1-δ)*k
    cons(iv, k, n)  = k^α*n^(1-α) - iv - ϕ*(iv/k - δ)^2*k

    for t in 1:T               

        for i in 1:Nk                # --- loop over states k_i ---
            best = -Inf
            bj, bl = 1, 1

            for j in 1:Nk            # --- loop over choices k'_j ---
                iv = inv_val(kgrid[j], kgrid[i])

                for l in 1:Nn        # --- loop over choices n_l ---
                    cv = cons(iv, kgrid[i], ngrid[l])
                    cv <= 0 && continue          # infeasible: skip
                    v  = log(cv) - χ*ngrid[l]^(1+γ)/(1+γ) + β*V[j]  # Bug 3,5,8 fixed
                    if v > best
                        best, bj, bl = v, j, l
                    end
                end
            end

            V_new[i] = best   # Bug 6 fixed: policy indexed by state i
            kpol[i]  = bj
            npol[i]  = bl
        end                          # Bug 7 fixed: no duplicate block here

        err = norm(V_new .- V, Inf)
        V  .= V_new                  # in-place update avoids extra allocation
        if err < ϵ
            println("VFI converged in $t iterations (err=$err)")
            break
        end
    end

    return V, kpol, npol
end



# -------------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------------
p  = Params()
ss = steady_state(p)

# -------------------------------------------------------------------------
# Q2: transition path
# -------------------------------------------------------------------------
tp = transition_path(p; k0 = 0.5*ss.k, T = 200)
plot(0:200, [tp.k tp.n tp.c tp.y], label = ["k" "n" "c" "y"], layout = 4)

# -------------------------------------------------------------------------
# Q3: VFI (non-stochastic, ϕ=0)
# -------------------------------------------------------------------------
Nk   = 500
Nn   = 50
# Center the capital grid around SS; ±40% is wide enough for a 50% shock
kgrid = collect(range(0.6*ss.k, 1.4*ss.k, length = Nk))
# Labor grid: biologically plausible range around SS n*=0.3
ngrid = collect(range(0.1, 0.9, length = Nn))

V, kpol, npol = value_function_iteration(p, kgrid, ngrid, 5_000, 1e-7)

# Recover policy rules as levels
k_policy = kgrid[kpol]   # k'(k)
n_policy = ngrid[npol]   # n(k)

# Implied investment, consumption, output at each grid point
@unpack α, δ = p
y_grid = kgrid.^α .* n_policy.^(1-α)
i_grid = k_policy .- (1-δ).*kgrid
c_grid = y_grid .- i_grid

# Check: policy rule should map k* → k* at steady state
k_ss_idx = argmin(abs.(kgrid .- ss.k))
println("SS check  k*(k̄) = $(k_policy[k_ss_idx]) ≈ $(ss.k)")
println("SS check  n*(k̄) = $(n_policy[k_ss_idx]) ≈ $(ss.n)")

# Plot policy rules
plt_vfi = plot(kgrid, [k_policy kgrid],  label = ["k'(k)" "45° line"],
               xlabel = "k",  title  = "Capital policy",  lw = 2)
plt_n   = plot(kgrid,  n_policy,          label = "n(k)",
               xlabel = "k",  title  = "Labor policy",    lw = 2)
plt_c   = plot(kgrid,  c_grid,            label = "c(k)",
               xlabel = "k",  title  = "Consumption",     lw = 2)
plt_V   = plot(kgrid,  V,                 label = "V(k)",
               xlabel = "k",  title  = "Value function",  lw = 2)
plot(plt_vfi, plt_n, plt_c, plt_V, layout = (2,2))

# -------------------------------------------------------------------------
# Q4: repeat Q2 and Q3 with ϕ=10
# -------------------------------------------------------------------------
# SS is identical (at SS: i/k = δ so adjustment cost = 0 regardless of ϕ)
p4    = Params(p; ϕ = 10.0)      # copy calibrated p, just flip ϕ

# Q4a: transition path
tp4 = transition_path(p4; k0 = 0.5*ss.k, T = 200)

# Q4b: VFI
V4, kpol4, npol4 = value_function_iteration(p4, kgrid, ngrid, 5_000, 1e-7)

k_policy4 = kgrid[kpol4]
n_policy4 = ngrid[npol4]
y_grid4   = kgrid.^α .* n_policy4.^(1-α)
i_grid4   = k_policy4 .- (1-δ).*kgrid
ψ4        = i_grid4 ./ kgrid .- δ
c_grid4   = y_grid4 .- i_grid4 .- p4.ϕ .* ψ4.^2 .* kgrid

# --- comparison plots: ϕ=0 vs ϕ=10 ---
# Transition paths
p_tp_k = plot(0:200, [tp.k tp4.k], label = ["ϕ=0" "ϕ=10"],
              xlabel = "t", title = "Capital path", lw = 2)
p_tp_i = plot(0:200, [tp.i tp4.i], label = ["ϕ=0" "ϕ=10"],
              xlabel = "t", title = "Investment path", lw = 2)
p_tp_c = plot(0:200, [tp.c tp4.c], label = ["ϕ=0" "ϕ=10"],
              xlabel = "t", title = "Consumption path", lw = 2)
p_tp_n = plot(0:200, [tp.n tp4.n], label = ["ϕ=0" "ϕ=10"],
              xlabel = "t", title = "Hours path", lw = 2)
plot(p_tp_k, p_tp_i, p_tp_c, p_tp_n, layout = (2,2))

# Policy rules
p_k4 = plot(kgrid, [k_policy k_policy4 kgrid],
            label = ["k'(k) ϕ=0" "k'(k) ϕ=10" "45°"],
            xlabel = "k", title = "Capital policy", lw = 2)
p_n4 = plot(kgrid, [n_policy n_policy4],
            label = ["n(k) ϕ=0" "n(k) ϕ=10"],
            xlabel = "k", title = "Labor policy", lw = 2)
p_i4 = plot(kgrid, [i_grid i_grid4],
            label = ["i(k) ϕ=0" "i(k) ϕ=10"],
            xlabel = "k", title = "Investment policy", lw = 2)
p_c4 = plot(kgrid, [c_grid c_grid4],
            label = ["c(k) ϕ=0" "c(k) ϕ=10"],
            xlabel = "k", title = "Consumption policy", lw = 2)
plot(p_k4, p_n4, p_i4, p_c4, layout = (2,2))

