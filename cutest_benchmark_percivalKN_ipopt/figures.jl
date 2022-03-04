using Pkg; Pkg.activate(".")
using JLD2, Plots, SolverBenchmark #, SolverCore

nb = "45"
name = "2022-03-05_ipopt_dcildl_percival_fpsK_45_300"
@load "$name.jld2" stats
solved(df) = (df.status .== :first_order)

for solver in keys(stats)
  # Number of problems solved by ipopt
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

open("2022-03-05_stats_fpsK_$nb.dat", "w") do io
  print(io, stats[:fps][!, [:name, :nvar, :ncon, :status, :objective, :elapsed_time, :iter, :primal_feas, :dual_feas]])
end
