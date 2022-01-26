using Pkg; Pkg.activate("")
Pkg.instantiate()
using CUTEst, NLPModels, SolverBenchmark
using NLPModelsIpopt, DCISolver, Percival, FletcherPenaltyNLPSolver
nmax = 10000
problems = readlines("list_problems_eq_$nmax.dat")
cutest_problems = (CUTEstModel(p) for p in problems)

max_time = 1200.0 #20 minutes
tol = 1e-5

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
  :fps => nlp -> fps_solve(
    nlp,
    nlp.meta.x0,
    σ_0 = 1000.,
    max_iter = typemax(Int64),
    max_time = max_time,
    max_eval = typemax(Int64),
    unconstrained_solver = knitro,
    atol = tol,
    rtol = tol,
  )
)
stats = bmark_solvers(solvers, cutest_problems)

using JLD2
@save "ipopt_dcildl_percival_fpsK_$(string(length(problems))).jld2" stats
