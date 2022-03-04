using Pkg; Pkg.activate(".")
using JLD2, Plots, SolverBenchmark #, SolverCore

name = "2022-02-28_OP_bounds_100_9"
# 2022-02-28_OP_equality_100_25 # none is scalable
# 2022-02-28_OP_mixed_100_62 # none is scalable
# 2022-02-28_OP_unconstrained_100_90
@load "$name.jld2" stats
solved(df) = (df.status .== :first_order)

for solver in keys(stats)
  @show size(stats[solver][solved(stats[solver]), [:name]], 1)
end

costs = [
  df -> .!solved(df) * Inf + df.elapsed_time,
  df -> .!solved(df) * Inf + df.neval_obj + df.neval_cons,
]
costnames = ["Time", "Evaluations of obj + cons"]
p = profile_solvers(stats, costs, costnames)
png(p, "$name")
# Plots.svg(p, "ipopt_dcildl_82")

open("stats_$name.dat", "w") do io
  print(io, stats[:fps][!, [:name, :nvar, :ncon, :status, :objective, :elapsed_time, :iter, :primal_feas, :dual_feas]])
end
