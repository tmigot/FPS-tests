using Pkg
Pkg.activate(".")
Pkg.instantiate()
using ADNLPModels, JSOSolvers, NLPModels, SolverBenchmark
using OptimizationProblems, OptimizationProblems.ADNLPProblems
using Stopping, StoppingInterface, NLPModelsIpopt
using FletcherPenaltyNLPSolver, Dates, JLD2
using Percival, DCISolver

n = 100

df = OptimizationProblems.meta
problems = df[df.has_equalities_only .&& df.nvar .> 1 .&& .!df.has_bounds, :name]
problems = [eval(Symbol(problem))(n = n) for problem ∈ problems]

atol, rtol = 1e-5, 1e-7

solvers = Dict(
  :ipopt => model -> ipopt(model, print_level = 0, dual_inf_tol = Inf, constr_viol_tol = Inf, compl_inf_tol = Inf, tol = rtol, max_cpu_time = max_time),
  :percival => model -> percival(model, atol = atol, rtol = rtol, max_time = max_time),
  :dci => model -> dci(model, atol = atol, rtol = rtol, linear_solver = :ldlfact),
  :fps_ipopt1_ldlt => model -> fps_solve(model, atol = atol, rtol = rtol, max_time = max_time, unconstrained_solver = StoppingInterface.ipopt, hessian_approx = Val(1), qds_solver = :ldlt),
  :fps_tron1_ldlt => model -> fps_solve(model, atol = atol, rtol = rtol, max_time = max_time, unconstrained_solver = StoppingInterface.tron, hessian_approx = Val(1), qds_solver = :ldlt),
  :fps_ipopt2_ldlt => model -> fps_solve(model, atol = atol, rtol = rtol, max_time = max_time, unconstrained_solver = StoppingInterface.ipopt, hessian_approx = Val(2), qds_solver = :ldlt),
  :fps_tron2_ldlt => model -> fps_solve(model, atol = atol, rtol = rtol, max_time = max_time, unconstrained_solver = StoppingInterface.tron, hessian_approx = Val(2), qds_solver = :ldlt),
  :fps_ipopt1_it => model -> fps_solve(model, atol = atol, rtol = rtol, max_time = max_time, unconstrained_solver = StoppingInterface.ipopt, hessian_approx = Val(1), qds_solver = :iterative),
  :fps_tron1_it => model -> fps_solve(model, atol = atol, rtol = rtol, max_time = max_time, unconstrained_solver = StoppingInterface.tron, hessian_approx = Val(1), qds_solver = :iterative),
  :fps_ipopt2_it => model -> fps_solve(model, atol = atol, rtol = rtol, max_time = max_time, unconstrained_solver = StoppingInterface.ipopt, hessian_approx = Val(2), qds_solver = :iterative),
  :fps_tron2_it => model -> fps_solve(model, atol = atol, rtol = rtol, max_time = max_time, unconstrained_solver = StoppingInterface.tron, hessian_approx = Val(2), qds_solver = :iterative),
)

if StoppingInterface.is_knitro_installed
  solvers[:knitro] = model -> knitro(model, atol = atol, rtol = rtol, maxtime_real = max_time)
  solvers[:fps_knitro1_it] = model -> fps_solve(model, atol = atol, rtol = rtol, unconstrained_solver = StoppingInterface.knitro, hessian_approx = Val(1), qds_solver = :iterative)
  solvers[:fps_knitro1_ldlt] = model -> fps_solve(model, atol = atol, rtol = rtol, unconstrained_solver = StoppingInterface.knitro, hessian_approx = Val(1), qds_solver = :ldlt)
  solvers[:fps_knitro2_it] = model -> fps_solve(model, atol = atol, rtol = rtol, unconstrained_solver = StoppingInterface.knitro, hessian_approx = Val(2), qds_solver = :iterative)
  solvers[:fps_knitro2_ldlt] = model -> fps_solve(model, atol = atol, rtol = rtol, unconstrained_solver = StoppingInterface.knitro, hessian_approx = Val(2), qds_solver = :ldlt)
end

stats = bmark_solvers(solvers, problems)

name = "$(today())_OP_equality_$(n)_$(string(length(problems)))"
@save "$(name).jld2" stats

cols = [:id, :name, :nvar, :objective, :dual_feas, :neval_obj, :neval_grad, :neval_hess, :iter, :elapsed_time, :status]
header = Dict(
  :nvar => "n",
  :objective => "f(x)",
  :dual_feas => "‖∇f(x)‖",
  :neval_obj => "# f",
  :neval_grad => "# ∇f",
  :neval_hess => "# ∇²f",
  :elapsed_time => "t",
)

first_order(df) = df.status .== :first_order
unbounded(df) = df.status .== :unbounded
solved(df) = first_order(df) .| unbounded(df)
costnames = ["time", "obj + grad + hess"]
costs = [
  df -> .!solved(df) .* Inf .+ df.elapsed_time,
  df -> .!solved(df) .* Inf .+ df.neval_obj .+ df.neval_grad .+ df.neval_hess,
]

using Plots
gr()

p = profile_solvers(stats, costs, costnames)
png("$(name)_profile_wall")
p1 = performance_profile(
  stats,
  costs[1],
  legend=:bottomright,
  title = "Performance profile on performance profile bounds w.r.t. $(costnames[1])",
)
png("$(name)_pp_time")
p2 = performance_profile(
  stats,
  costs[2],
  legend = :bottomright,
  title = "Performance profile on performance profile bounds w.r.t. $(costnames[2])",
)
png("$(name)_pp_sum")
