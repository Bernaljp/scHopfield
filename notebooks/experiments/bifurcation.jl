using Pkg
# Ensure we use the global environment or the active one
using JSON
using CSV
using DataFrames
using BifurcationKit
using Setfield
using LinearAlgebra

# 1. Read parameters
params_file = length(ARGS) > 0 ? ARGS[1] : "params.json"
params = JSON.parsefile(params_file)

I1 = Float64(params["I1"])
I2 = Float64(params["I2"])
w11 = Float64(params["w11"])
w22 = Float64(params["w22"])
gamma1 = Float64(params["gamma1"])
gamma2 = Float64(params["gamma2"])
c_start = Float64(params["c_start"])

# 2. Define the biological activation functions
function sigma(x)
    x_clamped = max(x, 0.0)
    return (x_clamped^4) / (1.0 + x_clamped^4)
end

function sigma_prime(x)
    if x <= 0.0
        return 0.0
    end
    return (4.0 * (x^3)) / ((1.0 + x^4)^2)
end

# 3. Define the Bifurcation Problem
function F(X, p)
    c = p.c
    x1, x2 = X[1], X[2]
    dx1 = w11 * sigma(x1) + c * sigma(x2) + I1 - gamma1 * x1
    dx2 = c * sigma(x1) + w22 * sigma(x2) + I2 - gamma2 * x2
    return [dx1, dx2]
end

function J(X, p)
    c = p.c
    x1, x2 = X[1], X[2]
    J11 = w11 * sigma_prime(x1) - gamma1
    J12 = c * sigma_prime(x2)
    J21 = c * sigma_prime(x1)
    J22 = w22 * sigma_prime(x2) - gamma2
    return [J11 J12; J21 J22]
end

u0_list = params["u0_list"]

for (i, u0_in) in enumerate(u0_list)
    u0 = Float64.(u0_in)
    p = (c = c_start,)
    
    bifprob = BifurcationProblem(
        F, u0, p, (@optic _.c); 
        J=J, 
        record_from_solution = (x, p; k...) -> (x1=x[1], x2=x[2])
    )

    optn = NewtonPar(verbose=false)
    br = continuation(
        bifprob, 
        PALC(), 
        ContinuationPar(p_min=-15.0, p_max=1.0, newton_options=optn, max_steps=2000),
        verbosity=0,
        normC=norm
    )

    c_vals = Float64[]
    x1_vals = Float64[]
    x2_vals = Float64[]

    for pt in br.branch
        push!(c_vals, Float64(pt.param))
        push!(x1_vals, Float64(pt.x1))
        push!(x2_vals, Float64(pt.x2))
    end

    df = DataFrame(c=c_vals, x1=x1_vals, x2=x2_vals)
    CSV.write("/tmp/branch_results_$i.csv", df)
    println("Branch $i continuation complete. Saved $(nrow(df)) points.")
end
