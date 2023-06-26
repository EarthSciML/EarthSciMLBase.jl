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

    result = EarthSciMLBase.add_dims(exp, [u], [x, y, t])
    @test isequal(result, wantexp)
end

@testset "Equation" begin
    result = EarthSciMLBase.add_dims(eq, [u, q], [x, y, t])
    @test isequal(result, wanteq[1])
end

@testset "Equation 2" begin
    result = EarthSciMLBase.add_dims(eq2, [u], [x, y, t])

    wanteq = let
        @parameters x y t k
        @variables u(..)
        Dt = Differential(t)
        Dt(u(x, y, t)) ~ 3k * u(x, y, t) + 1
    end

    @test isequal(result, wanteq)
end