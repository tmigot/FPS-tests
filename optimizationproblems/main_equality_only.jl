using Pkg
Pkg.activate(".")
Pkg.instantiate()
using ADNLPModels, JSOSolvers, NLPModels, SolverBenchmark
using OptimizationProblems, OptimizationProblems.ADNLPProblems, NLPModelsJuMP
using Stopping, StoppingInterface, NLPModelsIpopt
using DataFrames
using FletcherPenaltyNLPSolver, Dates, JLD2
using Percival, DCISolver, NLPModelsKnitro


n = 100

df = OptimizationProblems.meta
problems = df[df.has_equalities_only .& (df.nvar .> 10) .& (.!df.has_bounds), :name]
problems = [MathOptNLPModel(OptimizationProblems.PureJuMP.eval(Symbol(problem))(n = n), name = string(problem)) for problem ∈ problems]
jth_hess_implemented = false

atol, rtol = 1e-4, 1e-5
max_time = 1200.

solvers = Dict(
  :ipopt => model -> ipopt(model, print_level = 0, dual_inf_tol = Inf, constr_viol_tol = Inf, compl_inf_tol = Inf, tol = rtol, max_cpu_time = max_time),
  :percival => model -> percival(model, atol = atol, rtol = rtol, max_time = max_time),
  :dci => model -> dci(model, atol = atol, rtol = rtol, linear_solver = :ldlfact),
  :fps_ipopt2_ldlt => model -> fps_solve(model, atol = atol, rtol = rtol, max_time = max_time, subproblem_solver = StoppingInterface.ipopt, hessian_approx = Val(2), qds_solver = :ldlt),
  :fps_tron2_ldlt => model -> fps_solve(model, atol = atol, rtol = rtol, max_time = max_time, subproblem_solver = StoppingInterface.tron, hessian_approx = Val(2), qds_solver = :ldlt),
  :fps_trunk2_ldlt => model -> fps_solve(model, atol = atol, rtol = rtol, max_time = max_time, subproblem_solver = StoppingInterface.trunk, hessian_approx = Val(2), qds_solver = :ldlt),
  :fps_ipopt2_it => model -> fps_solve(model, atol = atol, rtol = rtol, max_time = max_time, subproblem_solver = StoppingInterface.ipopt, hessian_approx = Val(2), qds_solver = :iterative),
  :fps_tron2_it => model -> fps_solve(model, atol = atol, rtol = rtol, max_time = max_time, subproblem_solver = StoppingInterface.tron, hessian_approx = Val(2), qds_solver = :iterative),
  :fps_trunk2_it => model -> fps_solve(model, atol = atol, rtol = rtol, max_time = max_time, subproblem_solver = StoppingInterface.trunk, hessian_approx = Val(2), qds_solver = :iterative),
)

if jth_hess_implemented
  solvers[:fps_ipopt1_ldlt] = model -> fps_solve(model, atol = atol, rtol = rtol, max_time = max_time, subproblem_solver = StoppingInterface.ipopt, hessian_approx = Val(1), qds_solver = :ldlt)
  solvers[:fps_tron1_ldlt] = model -> fps_solve(model, atol = atol, rtol = rtol, max_time = max_time, subproblem_solver = StoppingInterface.tron, hessian_approx = Val(1), qds_solver = :ldlt)
  solvers[:fps_trunk1_ldlt] = model -> fps_solve(model, atol = atol, rtol = rtol, max_time = max_time, subproblem_solver = StoppingInterface.trunk, hessian_approx = Val(1), qds_solver = :ldlt)
  solvers[:fps_ipopt1_it] = model -> fps_solve(model, atol = atol, rtol = rtol, max_time = max_time, subproblem_solver = StoppingInterface.ipopt, hessian_approx = Val(1), qds_solver = :iterative)
  solvers[:fps_tron1_it] = model -> fps_solve(model, atol = atol, rtol = rtol, max_time = max_time, subproblem_solver = StoppingInterface.tron, hessian_approx = Val(1), qds_solver = :iterative)
  solvers[:fps_trunk1_it] = model -> fps_solve(model, atol = atol, rtol = rtol, max_time = max_time, subproblem_solver = StoppingInterface.trunk, hessian_approx = Val(1), qds_solver = :iterative)
end

if false # StoppingInterface.is_knitro_installed
  solvers[:knitro] = model -> knitro(model, atol = atol, rtol = rtol, maxtime_real = max_time)
  solvers[:fps_knitro1_it] = model -> fps_solve(model, atol = atol, rtol = rtol, subproblem_solver = StoppingInterface.knitro, hessian_approx = Val(1), qds_solver = :iterative)
  solvers[:fps_knitro1_ldlt] = model -> fps_solve(model, atol = atol, rtol = rtol, subproblem_solver = StoppingInterface.knitro, hessian_approx = Val(1), qds_solver = :ldlt)
  solvers[:fps_knitro2_it] = model -> fps_solve(model, atol = atol, rtol = rtol, subproblem_solver = StoppingInterface.knitro, hessian_approx = Val(2), qds_solver = :iterative)
  solvers[:fps_knitro2_ldlt] = model -> fps_solve(model, atol = atol, rtol = rtol, subproblem_solver = StoppingInterface.knitro, hessian_approx = Val(2), qds_solver = :ldlt)
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

cols_short = [:objective, :dual_feas, :neval_obj, :neval_grad, :neval_hess, :neval_hprod, :neval_cons, :neval_jac, :neval_jprod, :neval_jtprod, :iter, :elapsed_time, :status]
for (k, problem) in zip(1:length(problems), problems)
  @show (problem.meta.name, problem.meta.nvar, problem.meta.ncon, problem.meta.nnln, problem.meta.nlin)
  df = DataFrame(names = collect(keys(solvers)))
  for col in cols_short
    setproperty!(df, col, [stats[solver][!, col][k] for solver ∈ keys(solvers)])
  end
  pretty_stats(df)
end

#=
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
=#
