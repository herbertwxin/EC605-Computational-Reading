using Random, LinearAlgebra, Plots, Parameters, NLsolve, QuantEcon, CSV, DataFrames, Statistics

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

function value_function_iteration_stochastic(params::Params, kgrid::Vector{Float64}, ngrid::Vector{Float64},
    T::Int, ϵ::Float64, Π::Matrix{Float64}, agrid::Vector{Float64})
    @unpack α, β, δ, χ, γ, ϕ = params
    Nk = length(kgrid)
    Nn = length(ngrid) 
    Na = length(agrid)
    V     = zeros(Nk, Na)      # value function (over k states)
    V_new = zeros(Nk, Na)
    kpol  = ones(Int, Nk, Na) # index into kgrid for k' policy
    npol  = ones(Int, Nk, Na) # index into ngrid for n policy 
    # helpers: no name collision with loop variables
    inv_val(kp, k)  = kp - (1-δ)*k
    cons(iv, k, n, a)  = exp(a)*k^α*n^(1-α) - iv - ϕ*(iv/k - δ)^2*k

    for t in 1:T       
        EV = V * Π'        
        for s in 1:Na            # --- loop over states a_s ---
            for i in 1:Nk                # --- loop over states k_i ---
                best = -Inf
                bj, bl = 1, 1

                for j in 1:Nk            # --- loop over choices k'_j ---
                
                    iv = inv_val(kgrid[j], kgrid[i])

                    for l in 1:Nn        # --- loop over choices n_l ---
                        cv = cons(iv, kgrid[i], ngrid[l], agrid[s])
                        cv <= 0 && continue          # infeasible: skip
                        v  = log(cv) - χ*ngrid[l]^(1+γ)/(1+γ) + β*EV[j,s]  
                        if v > best
                            best, bj, bl = v, j, l
                        end
                    end
                end

                V_new[i,s] = best   # Bug 6 fixed: policy indexed by state i
                kpol[i,s]  = bj
                npol[i,s]  = bl
            end                          # Bug 7 fixed: no duplicate block here

        end

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
Nk   = 50
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


# -------------------------------------------------------------------------
# Q5: Stochastic model
# -------------------------------------------------------------------------
data = CSV.read("EC605-Computational-Reading/Assignments/logtfp_detrended.csv", DataFrame)
a    = data[:, 2]            
T    = length(a)

a_lag = a[1:T-1]
a_cur = a[2:T]
ρ_hat = (a_lag ⋅ a_cur) / (a_lag ⋅ a_lag)          
resid = a_cur .- ρ_hat .* a_lag
σ_hat = std(resid)


@show ρ_hat σ_hat

N  = 11
mc = rouwenhorst(N, ρ_hat, σ_hat)
agrid = collect(mc.state_values)   # N-point grid
Π  = mc.p                       # N×N transition matrix


Vs, kpols, npols = value_function_iteration_stochastic(p, kgrid, ngrid, 5_000, 1e-7, Π, agrid)

# -------------------------------------------------------------------------
# Q6: simulate 10_000 periods, drop first 3_000 burn-in
# -------------------------------------------------------------------------
function simulate(params::Params, kpol, npol, kgrid, agrid, Π;
                  T_sim = 10_000, T_burn = 3_000, seed = 42)
    @unpack α, δ, ϕ = params
    Random.seed!(seed)
    Na = length(agrid)

    # Cumulative transition probabilities for fast TFP draws
    Π_cdf = cumsum(Π, dims = 2)

    # Initialise at median TFP state and SS-nearest capital
    s = (Na + 1) ÷ 2
    i = size(kpol, 1) ÷ 2

    k_sim = zeros(T_sim)
    n_sim = zeros(T_sim)
    a_sim = zeros(T_sim)
    y_sim = zeros(T_sim)
    c_sim = zeros(T_sim)
    inv_sim = zeros(T_sim)

    for t in 1:T_sim
        k_sim[t]   = kgrid[i]
        a_sim[t]   = agrid[s]
        n_sim[t]   = ngrid[npol[i, s]]
        kp         = kgrid[kpol[i, s]]
        y_sim[t]   = exp(agrid[s]) * kgrid[i]^α * n_sim[t]^(1-α)
        iv         = kp - (1-δ)*kgrid[i]
        ψ          = iv/kgrid[i] - δ
        inv_sim[t] = iv
        c_sim[t]   = y_sim[t] - iv - ϕ*ψ^2*kgrid[i]

        # Draw next TFP state from row s of Π
        u = rand()
        s = searchsortedfirst(Π_cdf[s, :], u)

        # Next capital state: find nearest grid point to kp
        i = argmin(abs.(kgrid .- kp))
    end

    # Drop burn-in
    r = T_burn+1:T_sim
    return (k = k_sim[r], n = n_sim[r], a = a_sim[r],
            y = y_sim[r], c = c_sim[r], i = inv_sim[r])
end

sim = simulate(p, kpols, npols, kgrid, agrid, Π)

# Summary statistics on the remaining 7_000 observations
function summary_stats(x; name = "")
    σ   = std(x)
    ac1 = cor(x[1:end-1], x[2:end])
    println("$name  std=$(round(σ,digits=4))  ac1=$(round(ac1,digits=4))")
    return (; σ, ac1)
end

println("\n--- Q6 business cycle moments (ϕ=0) ---")
for (name, x) in zip(["y","c","i","n","k"], [sim.y, sim.c, sim.i, sim.n, sim.k])
    summary_stats(x; name = name)
end

# Cross-correlations with output
println("\nCorrelations with output:")
for (name, x) in zip(["c","i","n","k"], [sim.c, sim.i, sim.n, sim.k])
    println("  cor(y,$name) = $(round(cor(sim.y, x), digits=4))")
end

# -------------------------------------------------------------------------
# Q7: repeat with ϕ=10
# -------------------------------------------------------------------------
Vs7, kpols7, npols7 = value_function_iteration_stochastic(p4, kgrid, ngrid, 5_000, 1e-7, Π, agrid)
sim7 = simulate(p4, kpols7, npols7, kgrid, agrid, Π)

println("\n--- Q7 business cycle moments (ϕ=10) ---")
for (name, x) in zip(["y","c","i","n","k"], [sim7.y, sim7.c, sim7.i, sim7.n, sim7.k])
    summary_stats(x; name = name)
end
println("\nCorrelations with output:")
for (name, x) in zip(["c","i","n","k"], [sim7.c, sim7.i, sim7.n, sim7.k])
    println("  cor(y,$name) = $(round(cor(sim7.y, x), digits=4))")
end

