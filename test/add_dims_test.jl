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

@testset "Skip already-spatial variables" begin
    @variables v(..)
    v_spatial = v(t, x, y)

    result = EarthSciMLBase.add_dims([u, v_spatial], [x, y, t])
    @test length(result) == 2
    # u(t) should be expanded to u(x, y, t)
    @test length(Symbolics.arguments(Symbolics.unwrap(result[1]))) == 3
    # v(t, x, y) should be unchanged
    @test isequal(result[2], v_spatial)
end

@testset "Skip already-spatial in expression" begin
    @variables v(..)
    v_spatial = v(t, x, y)

    exp = 2u + 3k * v_spatial + 1

    wantexp = let
        @variables u(..) v(..)
        2u(x, y, t) + 3k * v(t, x, y) + 1
    end

    result = EarthSciMLBase.add_dims(exp, [u, v_spatial], [x, y, t])
    @test isequal(result, wantexp)
end

@testset "Skip already-spatial in equation" begin
    @variables v(..)
    v_spatial = v(t, x, y)

    eq_mixed = D(u) ~ k * v_spatial

    wanteq = let
        @variables u(..) v(..)
        D(u(x, y, t)) ~ k * v(t, x, y)
    end

    result = EarthSciMLBase.add_dims(eq_mixed, [u, v_spatial], [x, y, t])
    @test isequal(result, wanteq)
end
