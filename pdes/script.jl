using Pkg
Pkg.activate(".")
# Models
using NLPModels, PDENLPModels, PDEOptimizationProblems
# Solvers
using NLPModelsIpopt, DCISolver, Percival, FletcherPenaltySolver

PDEOptimizationProblems.problems

name = :apinene
n = 400
PDEOptimizationProblems.eval(Symbol("get_", name, "_meta"))(n)
@time nlp = apinene(n = 400)
@assert equality_constrained(nlp) && !has_bounds(nlp)
fps_solve(nlp, hessian_approx = Val(2), subproblem_solver = StoppingInterface.tron, qds_solver = :iterative)