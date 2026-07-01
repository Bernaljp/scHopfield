using Pkg
using JSON
using BifurcationKit
using Setfield
using LinearAlgebra
using Plots

# ---------------------------------------------------------
# 1. Parameter Initialization
# ---------------------------------------------------------
params_file = length(ARGS) > 0 ? ARGS[1] : "params.json"
params = JSON.parsefile(params_file)

p_model = (
    I1 = Float64(params["I1"]),
    I2 = Float64(params["I2"]),
    w11 = Float64(params["w11"]),
    w22 = Float64(params["w22"]),
    gamma1 = Float64(params["gamma1"]),
    gamma2 = Float64(params["gamma2"]),
    c = Float64(params["c_start"])
)

u0_list = params["u0_list"]

# ---------------------------------------------------------
# 2. Biological Activation Functions
# ---------------------------------------------------------
function sigma(x)
    # The "Smooth Clamp"
    # Prevents unphysical negative activation while remaining perfectly smooth 
    # at the origin so the Newton solver can safely round the extreme folds.
    x_c = max(x, 0.0)
    return (x_c^4) / (1.0 + x_c^4)
end

# ---------------------------------------------------------
# 3. Vector Field Definition
# ---------------------------------------------------------
function F(X, p)
    c, w11, w22 = p.c, p.w11, p.w22
    I1, I2 = p.I1, p.I2
    gamma1, gamma2 = p.gamma1, p.gamma2
    
    x1, x2 = X[1], X[2]
    dx1 = w11 * sigma(x1) + c * sigma(x2) + I1 - gamma1 * x1
    dx2 = c * sigma(x1) + w22 * sigma(x2) + I2 - gamma2 * x2
    return [dx1, dx2]
end

# ---------------------------------------------------------
# 4. Continuation Options
# ---------------------------------------------------------
optn = NewtonPar(verbose=false, tol=1e-9, max_iterations=30)
opts_br = ContinuationPar(
    p_min = -6.0, 
    p_max = 1.0, 
    newton_options = optn, 
    max_steps = 10000, 
    ds = 0.01, 
    dsmax = 0.02,    # Slightly tightened to ensure it safely navigates the tight lower fold
    dsmin = 1e-6, 
    detect_bifurcation = 3, 
    n_inversion = 8,
    max_bisection_steps = 30
)

# ---------------------------------------------------------
# 5. Branch Switching & Computation
# ---------------------------------------------------------
sym_idx = findfirst(u -> abs(u[1] - u[2]) < 1e-3, u0_list)
sym_idx = isnothing(sym_idx) ? 1 : sym_idx 
u0_main = Float64.(u0_list[sym_idx])

bifprob = BifurcationProblem(
    F, u0_main, p_model, (@optic _.c); 
    record_from_solution = (x, p; k...) -> (x1=x[1], x2=x[2])
)

println("Computing main symmetric branch...")
br_main = continuation(
    bifprob, PALC(), opts_br;
    verbosity = 0, normC = norm,
    bothside = true  
)

bif_idx = findfirst(pt -> pt.type != :endpoint, br_main.specialpoint)

if isnothing(bif_idx)
    error("Solver failed to detect a true bifurcation point.")
else
    println("Detected bifurcation at c ≈ $(round(br_main.specialpoint[bif_idx].param, digits=3)).")
end

println("Computing bifurcating (asymmetric) branches via branch switching...")
br_bif = continuation(
    br_main, bif_idx, setproperties(opts_br; max_steps = 10000);
    verbosity = 0, normC = norm,
    bothside = true, 
    record_from_solution = (x, p; k...) -> (x1=x[1], x2=x[2])
)

# ---------------------------------------------------------
# 6. Robust Custom Plotting & Formatting
# ---------------------------------------------------------
p1 = plot(
    title="Mutual Inhibition Toggle Switch", 
    xlabel="Mutual Inhibition Strength (c)", 
    ylabel="Steady State Gene 1 (x1)", 
    grid=true, gridalpha=0.4,           
    framestyle=:box,
    xlims=(-6.0, 1.0),
    legend=:bottomright,               # Strictly placed in the empty quadrant
    background_color_legend=RGBA(1, 1, 1, 0.9), 
    foreground_color_legend=:silver,   
    legendfontsize=9
)

function plot_branch_custom!(plt, br, branch_color)
    c_seg = Float64[]
    x1_seg = Float64[]
    cur_stable = br.branch[1].stable
    
    for i in 1:length(br.branch)
        pt = br.branch[i]
        is_jump = i > 1 && abs(pt.param - br.branch[i-1].param) > 0.1
        is_stab_change = i > 1 && (pt.stable != cur_stable)
        
        if is_jump || is_stab_change
            style = cur_stable ? :solid : :dash
            plot!(plt, c_seg, x1_seg, color=branch_color, linewidth=2.5, linestyle=style, label="")
            
            c_seg = Float64[]
            x1_seg = Float64[]
            cur_stable = pt.stable
            
            if !is_jump
                push!(c_seg, Float64(br.branch[i-1].param))
                push!(x1_seg, Float64(br.branch[i-1].x1))
            end
        end
        
        push!(c_seg, Float64(pt.param))
        push!(x1_seg, Float64(pt.x1))
    end
    
    style = cur_stable ? :solid : :dash
    plot!(plt, c_seg, x1_seg, color=branch_color, linewidth=2.5, linestyle=style, label="")
end

# Apply the branch colors
plot_branch_custom!(p1, br_main, :orange)
plot_branch_custom!(p1, br_bif, :blue)

# Clean, descriptive 4-part legend
plot!(p1, [NaN], [NaN], color=:orange, linewidth=2.5, label="Symmetric State")
plot!(p1, [NaN], [NaN], color=:blue,   linewidth=2.5, label="Asymmetric State")
plot!(p1, [NaN], [NaN], color=:gray,   linewidth=2.5, linestyle=:solid, label="Stable Branch")
plot!(p1, [NaN], [NaN], color=:gray,   linewidth=2.5, linestyle=:dash,  label="Unstable Branch")

savefig(p1, "native_bifurcation_plot.png")
println("Saved perfectly styled plot to native_bifurcation_plot_perfect.png")