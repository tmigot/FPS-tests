using Pkg; Pkg.activate("")
Pkg.instantiate()
using CUTEst, NLPModels, SolverBenchmark, StoppingInterface
using NLPModelsIpopt, DCISolver, Percival, FletcherPenaltyNLPSolver

use_knitro = false

nmax = 300
problems = readlines("list_problems_eq_$nmax.dat")
cutest_problems = (CUTEstModel(p) for p in problems)

max_time = 1200.0 #20 minutes
tol = 1e-5
atol = tol
rtol = tol

solvers = Dict(
  :ipopt => nlp -> ipopt(
      nlp,
      print_level = 0,
      dual_inf_tol = Inf,
      constr_viol_tol = Inf,
      compl_inf_tol = Inf,
      acceptable_iter = 0,
      max_cpu_time = max_time,
      tol = tol,
  ),
  :dcildl => nlp -> dci(
      nlp, # uses x0 = nlp.meta.x0 by default
      linear_solver = :ldlfact,
      max_time = max_time,
      max_iter = typemax(Int64),
      max_eval = typemax(Int64),
      atol = tol,
      ctol = tol,
      rtol = tol,
  ),
  :percival => nlp -> percival(
    nlp,
    #μ::Real = T(10.0),
    max_iter = typemax(Int64),
    max_time = max_time,
    max_eval = typemax(Int64),
    atol = tol,
    ctol = tol,
    rtol = tol,
    #subsolver_logger::AbstractLogger = NullLogger(),
    #inity = nothing,
    #subproblem_modifier = identity,
    #subsolver_max_eval = max_eval,
    #subsolver_kwargs = Dict(:max_cgiter => nlp.meta.nvar),
  ),
  :fps_tron => nlp -> fps_solve(
    nlp,
    max_iter = typemax(Int64),
    max_time = max_time,
    max_eval = typemax(Int64),
    unconstrained_solver = StoppingInterface.tron,
    atol = tol,
    rtol = tol,
  ),
  :fps_lbfgs => nlp -> fps_solve(
    nlp,
    max_iter = typemax(Int64),
    max_time = max_time,
    max_eval = typemax(Int64),
    unconstrained_solver = StoppingInterface.lbfgs,
    atol = tol,
    rtol = tol,
  ),
  :fps_ipopt => nlp -> fps_solve(
    nlp,
    max_iter = typemax(Int64),
    max_time = max_time,
    max_eval = typemax(Int64),
    unconstrained_solver = StoppingInterface.ipopt,
    atol = tol,
    rtol = tol,
  )
)

if use_knitro
  using NLPModelsKnitro
  solvers[:knitro] = model -> knitro(model, atol = atol, rtol = rtol, maxtime_real = max_time)
  solvers[:fps_knitro1_it] = model -> fps_solve(model, atol = atol, rtol = rtol, max_time = max_time, unconstrained_solver = StoppingInterface.knitro, hessian_approx = Val(1), qds_solver = :iterative)
  solvers[:fps_knitro1_ldlt] = model -> fps_solve(model, atol = atol, rtol = rtol, max_time = max_time, unconstrained_solver = StoppingInterface.knitro, hessian_approx = Val(1), qds_solver = :ldlt)
  solvers[:fps_knitro2_it] = model -> fps_solve(model, atol = atol, rtol = rtol, max_time = max_time, unconstrained_solver = StoppingInterface.knitro, hessian_approx = Val(2), qds_solver = :iterative)
  solvers[:fps_knitro2_ldlt] = model -> fps_solve(model, atol = atol, rtol = rtol, max_time = max_time, unconstrained_solver = StoppingInterface.knitro, hessian_approx = Val(2), qds_solver = :ldlt)
end

stats = bmark_solvers(solvers, cutest_problems)

using JLD2, Dates
@save "$(today())_ipopt_dcildl_percival_fpsK_$(string(length(problems)))_$nmax.jld2" stats
