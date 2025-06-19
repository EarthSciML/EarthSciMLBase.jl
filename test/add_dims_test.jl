using EarthSciMLBase
using ModelingToolkit
t = ModelingToolkit.t_nounits
D = ModelingToolkit.D_nounits

@parameters x y k
@variables u(t) q(t)
eq = D(u) ~ 3k * u + q + 1
eq2 = D(u) ~ 3k * u + 1

wanteq = let
    @parameters x y k
    @variables u(..) q(..)
    [D(u(x, y, t)) ~ 3k * u(x, y, t) + q(x, y, t) + 1]
end

@testset "Expression" begin
    exp = 3k * u + 1

    wantexp = let
        @parameters x y k
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
        @parameters x y k
        @variables u(..)
        D(u(x, y, t)) ~ 3k * u(x, y, t) + 1
    end

    @test isequal(result, wanteq)
end
