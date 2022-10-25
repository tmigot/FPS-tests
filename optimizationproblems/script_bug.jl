using Pkg
Pkg.activate(".")
using OptimizationProblems, NLPModels, NLPModelsJuMP, FletcherPenaltySolver, JSOSolvers, StoppingInterface
nlp = MathOptNLPModel(OptimizationProblems.PureJuMP.hovercraft1d(n = 100))
fps_solve(nlp, subproblem_solver = StoppingInterface.trunk, hessian_approx = Val(2), qds_solver = :iterative)
@show neval_jac(nlp) neval_hess(nlp)
