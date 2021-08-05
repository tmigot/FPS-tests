using DelimitedFiles, LinearAlgebra, Printf, SparseArrays
using BenchmarkTools, DataFrames, Dates, JLD2, Plots
#JSO packages
using CUTEst, NLPModels, NLPModelsIpopt, BenchmarkProfiles, SolverBenchmark, SolverCore
#This package
using NLPModelsKnitro

function runtest(problems, solvers; today::String = string(today()))
  list = ""
  for solver in keys(solvers)
    list = string(list, "_$(solver)")
  end
  stats = bmark_solvers(solvers, problems)

  @save "$(today)_$(list)_$(string(length(pnames))).jld2" stats

  return stats
end

#=
nmax = 300
_pnames = CUTEst.select(
  max_var = nmax,
  min_con = 1,
  max_con = nmax,
  only_free_var = true,
  only_equ_con = true,
  objtype = 3:6,
)

#Remove all the problems ending by NE as Ipopt cannot handle them.
pnamesNE = _pnames[findall(x -> occursin(r"NE\b", x), _pnames)]
pnames = setdiff(_pnames, pnamesNE)
cutest_problems = (CUTEstModel(p) for p in pnames)
=#

jump_problems = MathOptNLPModel[]
for prob in names(OptimizationProblems)
  prob == :OptimizationProblems && continue
  prob_fn = eval(prob)
  nlp = MathOptNLPModel(prob_fn())
  if nlp.meta.nlin > 0
    push!(jump_problems, nlp)
  end
end

#Same time limit for all the solvers
max_time = 60.0
solvers = Dict(
  :knitro =>
    nlp -> knitro(
      nlp,
      out_hints = 0,
      outlev = 0,
      feastol = 1e-5,
      feastol_abs = 1e-5,
      opttol = 1e-5,
      opttol_abs = 1e-5,
      maxtime_cpu = max_time,
      x0 = nlp.meta.x0,
    ),
  :DCILDL =>
    nlp -> dci(
      nlp,
      nlp.meta.x0,
      linear_solver = :ldlfact,
      max_time = max_time,
      max_iter = typemax(Int64),
      max_eval = typemax(Int64),
    ),
)

const SUITE = BenchmarkGroup()
#=
SUITE[:cutest_knitro_dcildl_benchmark] =
  @benchmarkable runtest(cutest_problems, solvers) samples = 5
=#
SUITE[:jump_knitro_dcildl_benchmark] =
  @benchmarkable runtest(jump_problems, solvers) samples = 5