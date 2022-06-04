using EarthSciMLBase
using ModelingToolkit, Catalyst

@parameters x y t k
@variables u(t) q(t)
Dt = Differential(t)
eq = Dt(u) ~ 3k * u + q + 1
eq2 = Dt(u) ~ 3k * u + 1

wanteq = let 
    @parameters x y t k
    @variables u(..) q(..)
    Dt = Differential(t)
    [Dt(u(x, y, t)) ~ 3k * u(x, y, t) + q(x, y, t) + 1]
end

@testset "Expression" begin
    exp = 3k * u + 1

    wantexp = let 
        @parameters x y t k
        @variables u(..)
        3k * u(x, y, t) + 1
    end

    result = EarthSciMLBase.add_dims(exp, [u], x, y, t)
    @test isequal(result, wantexp)
end

@testset "Equation" begin
    result = EarthSciMLBase.add_dims(eq, [u, q], x, y, t)
    @test isequal(result, wanteq[1])
end

@testset "Equation 2" begin
    result = EarthSciMLBase.add_dims(eq2, [u], x, y, t)

    wanteq = let 
        @parameters x y t k
        @variables u(..)
        Dt = Differential(t)
        Dt(u(x, y, t)) ~ 3k * u(x, y, t) + 1
    end

    @test isequal(result, wanteq)
end

@testset "ODESystem" begin
    @named sys = ODESystem([eq])
    result = sys + AddDims(x, y, t)
    @test isequal(result, wanteq)
    @test isequal(AddDims(x, y, t) + sys, wanteq)
end

@testset "ReactionSystem" begin
    rn = @reaction_network begin
        β, m₁ --> m₂
    end β
    result = rn + AddDims(x, y, t)

    wantrn = let 
        @parameters β x y t
        @variables m₁(..) m₂(..)
        Dt = Differential(t)
        [
            Dt(m₁(x, y, t)) ~ -β*m₁(x, y, t), 
            Dt(m₂(x, y, t)) ~ β*m₁(x, y, t),
        ]
    end
    @test isequal(result, wantrn)
    @test isequal(AddDims(x, y, t) + rn, wantrn)
end