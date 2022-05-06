using EarthSciMLBase
using ModelingToolkit

@variables t x(t) y(t)
D = Differential(t)

eq1 = D(x) ~ x
eq2 = D(x) ~ 2x + 1
eq3 = D(y) ~ y

@testset "Single equations" begin
    @test isequal(eq1 + eq2, [D(x) ~ 3x + 1])
    @test isequal(eq1 + eq3, [eq1, eq3])
end

@testset "Equation vectors" begin
    @test isequal([eq1, eq2] + [eq2, eq3], [eq1 + eq2..., eq3, eq2])
    @test isequal([eq2, eq3] + [eq1, eq2], [eq2 + eq1..., eq2, eq3])
end

@testset "Single plus vector" begin
    @test isequal([eq1, eq3] + eq2, [eq1, eq3] + [eq2])
    @test isequal(eq2 + [eq1, eq3], [eq2] + [eq1, eq3])
end