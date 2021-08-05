# Test linear constraints in NLPModelsKnitro.jl
# https://github.com/jump-dev/KNITRO.jl/blob/c81586bd406cbf6fbde2c136feda28bf8dee31b7/src/kn_constraints.jl#L210
using Pkg
Pkg.activate(".")
Pkg.add(url="https://github.com/tmigot/NLPModelsKnitro.jl#linear-cons")
using NLPModelsKnitro
using OptimizationProblems, NLPModelsJuMP

#=
problems = MathOptNLPModel[]
for prob in names(OptimizationProblems)
  prob == :OptimizationProblems && continue
  prob_fn = eval(prob)
  nlp = MathOptNLPModel(prob_fn())
  if nlp.meta.nlin > 0
    println(prob)
    push!(problems, nlp)
  end
end
=#

prob = :hs105
nlp = MathOptNLPModel(eval(prob)())
solver = KnitroSolver(nlp, outlev = 0)
stats = knitro!(nlp, solver)
@show stats.solution
@show stats.status
# @test isapprox(stats.solution, [-1.4; 2.4], rtol = 1e-6)
# @test stats.iter == 1
# @test stats.status == :first_order

# check constraint violation: https://github.com/jump-dev/KNITRO.jl/blob/c81586bd406cbf6fbde2c136feda28bf8dee31b7/src/kn_constraints.jl#L103
index = Cint.nlp.meta.lin # Vector{Cint}(undef, nlp.meta.ncon)
infeas, viols = KN_get_con_viols(solver.kc, index)
@show viols

gx = KNITRO.KN_get_objgrad_values(solver.kc)[2]
@show gx
# @test isapprox(gx, [-4.8; -4.8], rtol = 1e-6)
cx = KNITRO.KN_get_con_values(solver.kc)
@show cx
# @test isapprox(norm(cx), 0, atol = 1e-6)
Jx = KNITRO.KN_get_jacobian_values(solver.kc)
@show Jx
# @test Jx[1] == [0; 0]
# @test Jx[2] == [0; 1]
# @test Jx[3] == [1; 1]
# finalize(solver)