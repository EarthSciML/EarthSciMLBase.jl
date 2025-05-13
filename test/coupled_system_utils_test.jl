using EarthSciMLBase: steplength, grid,
                      timesteps, icbc, prune_observed, dtype
using EarthSciMLBase
using Test
using ModelingToolkit, DomainSets
using ModelingToolkit: t_nounits;
t = t_nounits;
using ModelingToolkit: D_nounits;
D = D_nounits;
using DynamicQuantities
using OrdinaryDiffEq: solve
using SciMLBase: ReturnCode

@test steplength([0, 0.1, 0.2]) == 0.1

@parameters x y lon=0.0 lat=0.0 lev=1.0 α=10.0
@constants p = 1.0
@variables(u(t)=1.0, v(t)=1.0, x(t)=1.0, y(t)=1.0)

eqs = [D(u) ~ -α * √abs(v) + lon,
    D(v) ~ -α * √abs(u) + lat,
    x ~ 2α + p + y,
    y ~ 4α - 2p
]

@named sys = ODESystem(eqs, t)
sys = structural_simplify(sys)
p = MTKParameters(sys, [])

xf = ModelingToolkit.build_explicit_observed_function(sys, x)

p = MTKParameters(sys, [α => 2.0])
@test isequal(xf([0.0, 0], p, 0.0), -1.0 + 6 * 2)

t_min = 0.0
lon_min, lon_max = -π, π
lat_min, lat_max = -0.45π, 0.45π
t_max = 11.5

indepdomain = t ∈ Interval(t_min, t_max)

partialdomains = [lon ∈ Interval(lon_min, lon_max),
    lat ∈ Interval(lat_min, lat_max)]

domain = DomainInfo(
    partialderivatives_δxyδlonlat,
    constIC(16.0, indepdomain), constBC(16.0, partialdomains...))

vars = unknowns(sys)

bcs = icbc(domain, vars)

@test dtype(domain) == Float64
@test dtype(DomainInfo(constIC(0, t ∈ Interval(0, 1)),
    constBC(16.0, lon ∈ Interval(0.0, 1.0)), dtype = Float32)) == Float32

@test timesteps(0:0.1:1, 0:0.15:1) ==
      [0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]

@test timesteps(0:0.1:0.3, 0:0.1000000000001:0.3) == [0.0, 0.1, 0.2, 0.3]

@testset "prune observed" begin
    @variables(x(t)=0.25, y(t), z(t), a(t))
    eqs = [D(x) ~ -z
           z ~ x
           y ~ x
           a ~ x]

    @named sys = ODESystem(eqs, t; metadata = :metatest,
        continuous_events = [[x ~ 0] => [x ~ 1], [x ~ 0.5, x ~ 2] => [x ~ 1.0]],
        discrete_events = (t == 1.0) => [x ~ x + 1]
    )

    @testset "single system" begin
        sys_simple = structural_simplify(sys)
        sys2 = prune_observed(sys, sys_simple, [])
        sys2 = structural_simplify(sys2)

        @test occursin("Differential(t)(x(t)) ~ -x(t)", string(equations(sys2))) ||
              occursin("Differential(t)(x(t)) ~ -z(t)", string(equations(sys2)))

        prob = ODEProblem(sys2, [], (0.0, 2.0))
        sol = solve(prob, saveat = 0.05)
        @test sol.retcode == ReturnCode.Success

        @test length(ModelingToolkit.get_continuous_events(sys2)) == 2
        @test length(ModelingToolkit.get_discrete_events(sys2)) == 1
    end

    @testset "nested system" begin
        @named sysx = ODESystem([D(x) ~ x], t)
        sys_nested = compose(sysx, sys)
        sys_nested_simple = structural_simplify(sys_nested)
        sys2 = prune_observed(sys_nested, sys_nested_simple, [])
        sys2 = structural_simplify(sys2)

        @test length(equations(sys2)) == 2

        prob = ODEProblem(sys2, [], (0.0, 2.0))
        sol = solve(prob, saveat = 0.05)
        @test sol.retcode == ReturnCode.Success

        @test length(ModelingToolkit.get_continuous_events(sys2)) == 2
        @test length(ModelingToolkit.get_discrete_events(sys2)) == 1
    end
end
