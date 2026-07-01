using Pkg
using JSON
using BifurcationKit
using Setfield
using LinearAlgebra
using Plots
using Plots.PlotMeasures

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
    dsmax = 0.02,    
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
br_main = continuation(bifprob, PALC(), opts_br; verbosity=0, normC=norm, bothside=true)

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
# 6. Multi-View 3D Plotting (c on Z-axis)
# ---------------------------------------------------------
function plot_branch_custom_3d!(plt, br, branch_color)
    c_seg, x1_seg, x2_seg = Float64[], Float64[], Float64[]
    cur_stable = br.branch[1].stable
    
    for i in 1:length(br.branch)
        pt = br.branch[i]
        is_jump = i > 1 && abs(pt.param - br.branch[i-1].param) > 0.1
        is_stab_change = i > 1 && (pt.stable != cur_stable)
        
        if is_jump || is_stab_change
            style = cur_stable ? :solid : :dash
            
            # Swapped mapping: x=x1, y=x2, z=c
            plot!(plt, x1_seg, x2_seg, c_seg, color=branch_color, linewidth=2.5, linestyle=style, label="")
            
            c_seg, x1_seg, x2_seg = Float64[], Float64[], Float64[]
            cur_stable = pt.stable
            
            if !is_jump
                push!(c_seg, Float64(br.branch[i-1].param))
                push!(x1_seg, Float64(br.branch[i-1].x1))
                push!(x2_seg, Float64(br.branch[i-1].x2))
            end
        end
        
        push!(c_seg, Float64(pt.param))
        push!(x1_seg, Float64(pt.x1))
        push!(x2_seg, Float64(pt.x2))
    end
    
    style = cur_stable ? :solid : :dash
    # Swapped mapping: x=x1, y=x2, z=c
    plot!(plt, x1_seg, x2_seg, c_seg, color=branch_color, linewidth=2.5, linestyle=style, label="")
end

# A 360-degree orbital tour of the 3D geometry
views = [
    (45, 30),   # Front-Right Isometric
    (135, 30),  # Back-Right
    (225, 45),  # Back-Left (Higher angle to look down into the fold)
    (315, 15)   # Front-Left (Lower angle to see the profile)
]

titles = [
    "Azimuth: 45°, Elevation: 30°",
    "Azimuth: 135°, Elevation: 30°",
    "Azimuth: 225°, Elevation: 45°",
    "Azimuth: 315°, Elevation: 15°"
]

plts = []

for i in 1:4
    leg_opt = (i == 1) ? :topleft : :none
    
    plt = plot(
        title=titles[i], 
        # Updated axis labels and limits to reflect the new orientation
        xlabel="Gene 1 (x1)", ylabel="Gene 2 (x2)", zlabel="Inhibition (c)",
        grid=true, gridalpha=0.4,           
        framestyle=:box,
        camera=views[i],
        xlims=(0, 3.5), ylims=(0, 3.5), zlims=(-6.0, 1.0),
        legend=leg_opt,
        background_color_legend=RGBA(1, 1, 1, 0.85), 
        foreground_color_legend=:silver,   
        legendfontsize=8,
        titlefontsize=10,
        margin=4mm
    )
    
    plot_branch_custom_3d!(plt, br_main, :orange)
    plot_branch_custom_3d!(plt, br_bif, :blue)
    
    # Legend manually assigned to the first plot only
    if i == 1
        nan_arr = Float64[NaN]
        plot!(plt, nan_arr, nan_arr, nan_arr, color=:orange, linewidth=2.5, label="Symmetric State")
        plot!(plt, nan_arr, nan_arr, nan_arr, color=:blue,   linewidth=2.5, label="Asymmetric State")
        plot!(plt, nan_arr, nan_arr, nan_arr, color=:gray,   linewidth=2.5, linestyle=:solid, label="Stable")
        plot!(plt, nan_arr, nan_arr, nan_arr, color=:gray,   linewidth=2.5, linestyle=:dash,  label="Unstable")
    end
    
    push!(plts, plt)
end

# Combine all 4 plots into a 2x2 grid
final_plot = plot(plts..., layout=(2,2), size=(1000, 800))

savefig(final_plot, "native_bifurcation_plot_3d_multiview.png")
println("Saved orbital 3D plots to native_bifurcation_plot_3d_orbital.png")