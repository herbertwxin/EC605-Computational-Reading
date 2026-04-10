using Random, LinearAlgebra, Plots, Parameters

# First-order MA process
function MA1(θ, N)
    ϵ = randn(N)
    y = zeros(N)
    for t in 2:N
        y[t] = ϵ[t] - θ * ϵ[t-1]
    end
    return y, ϵ
end
  
Θ = 2;
T = 100;
# Simulation
Y, x = MA1(Θ,T);

# Since thtea > 1, the process is non-invertible and as the coefficient in front of y explodes.



@with_kw mutable struct KalmanFilter
    A::Matrix{Float64} = [0. 0.; 1. 0.]      # 2x2: shifts ε_t → ε_{t-1}
    G::Matrix{Float64} = [1. -Θ]             # 1x2: y_t = ε_t - Θε_{t-1}
    C::Matrix{Float64} = [1.; 0.][:,:]       # 2x1: new shock enters first component only
    # R::Matrix{Float64} = [0.][:,:]           # 1x1: no measurement noise

    x̂0::Vector{Float64} = [0., 0.]          # prior mean: E[ε_0]=0, E[ε_{-1}]=0
    Σ0::Matrix{Float64} = [1. 0.; 0. 1.]    # prior variance: identity (both ~ N(0,1))
end


function updateBeliefs(KF::KalmanFilter,y,x̂,Σ)
    @unpack A,G,C = KF
    a = y - G*x̂
    K = A*Σ*G'*inv(G*Σ*G')
    x̂′= A*x̂ + K*a
    Σ′ = C*C' + (A-K*G)*Σ*(A' - G'*K')
    
    return x̂′,Σ′
end



function applyFilter(KF::KalmanFilter,Y)
    @unpack x̂0,Σ0 = KF

    T = size(Y,2) #how many rows are Y
    x̂ = zeros(length(x̂0),T+1)
    Σ = zeros(length(x̂0),length(x̂0),T+1) #note 3 dimensional array
    x̂[:,1] .= x̂0
    Σ[:,:,1] .= Σ0
    for t in 1:T
        x̂[:,t+1],Σ[:,:,t+1] = updateBeliefs(KF,Y[:,t],x̂[:,t],Σ[:,:,t])
    end

    return x̂,Σ
end

KF = KalmanFilter()
x̂,Σ = applyFilter(KF,Y')
plot(1:T,x,xlabel="Time",ylabel="x",label="True State",legend=true)
plot!(1:T,x̂[2, 2:T+1],label="Filter")

#The standard gain is Σ*G'*inv(G*Σ*G') (no A). With A multiplied in, 
#your x̂′ becomes the next predicted state x̂_{t+1|t} rather than the current filtered state x̂_{t|t}. 
#This can be an intentional "predict-and-update in one step" formulation, but it also changes 
#what a = y - G*x̂ means — x̂ must then be the predicted state x̂_{t|t-1}, not the filtered one. 
#Make sure x̂0 is your prior predicted state and that Σ is the predicted (not filtered) covariance throughout.