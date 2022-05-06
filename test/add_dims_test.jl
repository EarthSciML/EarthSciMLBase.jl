using EarthSciMLBase
using ModelingToolkit, Catalyst

@parameters x y t k
@variables u(t) q(t)
Dt = Differential(t)

exp = 3k * u + 1
eq = Dt(u) ~ 3k * u + q + 1

@testset "Expression" begin
    r1 = EarthSciMLBase.add_dims(exp, [u, q], x, y, t)
    @test sprint(print, r1) == "1 + 3k*u(x, y, t)"
end

@testset "Equation" begin
    r2 = EarthSciMLBase.add_dims(eq, [u, q], x, y, t)
    @test sprint(print, r2) == "Differential(t)(u(x, y, t)) ~ 1 + 3k*u(x, y, t) + q(x, y, t)"
end

@testset "ODESystem" begin
    @named sys = ODESystem([eq])
    r5 = sys + AddDims(x, y, t)
    @test sprint(print, r5) == "Symbolics.Equation[Differential(t)(u(x, y, t)) ~ 1 + 3k*u(x, y, t) + q(x, y, t)]"
    @test sys + AddDims(x, y, t) == AddDims(x, y, t) + sys
end

@testset "ReactionSystem" begin
    rn = @reaction_network begin
        β, m₁ --> m₂
    end β
    r6 = rn + AddDims(x, y, t)
    @test sprint(print, r6) == "Symbolics.Equation[Differential(t)(m₁(x, y, t)) ~ -β*m₁(x, y, t), Differential(t)(m₂(x, y, t)) ~ β*m₁(x, y, t)]"
    @test rn + AddDims(x, y, t) == AddDims(x, y, t) + rn
end