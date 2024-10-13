using Test
using Main.EarthSciMLBase
using Main.EarthSciMLBase: pvars, grid, time_range, add_partial_derivative_func
using ModelingToolkit, Catalyst
using MethodOfLines, DifferentialEquations, DomainSets
using ModelingToolkit: t_nounits;
t = t_nounits;
using ModelingToolkit: D_nounits;
D = D_nounits;
import SciMLBase
using Dates

@parameters x y α = 10.0
@variables u(t) v(t)

x_min = y_min = t_min = 0.0
x_max = y_max = 1.0
t_max = 11.5

eqs = [D(u) ~ -α * √abs(v),
    D(v) ~ -α * √abs(u),
]

@named sys = ODESystem(eqs, t)

indepdomain = t ∈ Interval(t_min, t_max)

partialdomains = [x ∈ Interval(x_min, x_max),
    y ∈ Interval(y_min, y_max)]

domain = DomainInfo(constIC(16.0, indepdomain), constBC(16.0, partialdomains...))

@testset "dims" begin
    dims_result = EarthSciMLBase.dims(domain)

    dims_want = [t, x, y]

    @test isequal(dims_result, dims_want)
end

@testset "domains" begin
    domains_result = EarthSciMLBase.domains(domain)

    domains_want = [
        t ∈ Interval(t_min, t_max),
        x ∈ Interval(x_min, x_max),
        y ∈ Interval(y_min, y_max),
    ]

    @test isequal(domains_result, domains_want)
end

@testset "pde" begin
    pde_want = let
        @parameters x y α = 10.0
        @variables u(..) v(..)

        x_min = y_min = t_min = 0.0
        x_max = y_max = 1.0
        t_max = 11.5

        eqs = [D(u(t, x, y)) ~ -α * √abs(v(t, x, y)),
            D(v(t, x, y)) ~ -α * √abs(u(t, x, y)),
        ]

        domains = [
            t ∈ Interval(t_min, t_max),
            x ∈ Interval(x_min, x_max),
            y ∈ Interval(y_min, y_max),
        ]

        # Periodic BCs
        bcs = [
            u(t_min, x, y) ~ 16.0,
            v(t_min, x, y) ~ 16.0,
            u(t, x_min, y) ~ 16.0,
            u(t, x_max, y) ~ 16.0,
            u(t, x, y_min) ~ 16.0,
            u(t, x, y_max) ~ 16.0,
            v(t, x_min, y) ~ 16.0,
            v(t, x_max, y) ~ 16.0,
            v(t, x, y_min) ~ 16.0,
            v(t, x, y_max) ~ 16.0,
        ]

        @named pdesys = PDESystem(eqs, bcs, domains, [t, x, y], [u(t, x, y), v(t, x, y)], [α])
    end

    pde_result = sys + domain

    @test isequal(pde_result.eqs, pde_want.eqs)
    @test isequal(pde_result.ivs, pde_want.ivs)
    @test isequal(pde_result.dvs, pde_want.dvs)
    @test isequal(pde_result.bcs, pde_want.bcs)
    @test isequal(pde_result.domain, pde_want.domain)
    @test isequal(pde_result.ps, pde_want.ps)
end

@testset "ReactionSystem" begin
    pde_want = let
        @parameters x y
        @variables m₁(..) m₂(..)
        eqs = [
            D(m₁(t, x, y)) ~ -10.0 * m₁(t, x, y),
            D(m₂(t, x, y)) ~ 10.0 * m₁(t, x, y),
        ]

        bcs = [
            m₁(t_min, x, y) ~ 16.0,
            m₂(t_min, x, y) ~ 16.0,
            m₁(t, x_min, y) ~ 16.0,
            m₁(t, x_max, y) ~ 16.0,
            m₁(t, x, y_min) ~ 16.0,
            m₁(t, x, y_max) ~ 16.0,
            m₂(t, x_min, y) ~ 16.0,
            m₂(t, x_max, y) ~ 16.0,
            m₂(t, x, y_min) ~ 16.0,
            m₂(t, x, y_max) ~ 16.0,
        ]

        dmns = [
            t ∈ Interval(t_min, t_max),
            x ∈ Interval(x_min, x_max),
            y ∈ Interval(y_min, y_max),
        ]

        PDESystem(eqs, bcs, dmns, [t, x, y], [m₁(t, x, y), m₂(t, x, y)], [], name=:sys)
    end

    rn = @reaction_network begin
        10.0, m₁ --> m₂
    end
    pde_result = rn + domain

    @test isequal(pde_result.eqs, pde_want.eqs)
    @test isequal(pde_result.ivs, pde_want.ivs)
    @test isequal(pde_result.dvs, pde_want.dvs)
    @test isequal(pde_result.bcs, pde_want.bcs)
    @test isequal(pde_result.domain, pde_want.domain)
    @test isequal(pde_result.ps, pde_want.ps)
end

@testset "zero-grad and periodic" begin
    domain = DomainInfo(
        constIC(16.0, indepdomain),
        periodicBC(x ∈ Interval(x_min, x_max)),
        zerogradBC(y ∈ Interval(y_min, y_max)),
    )
    pdesys = sys + domain

    want_bcs = let
        @parameters x y t
        @variables u(..) v(..)
        Dy = Differential(y)
        [
            u(t_min, x, y) ~ 16.0,
            v(t_min, x, y) ~ 16.0,
            u(t, x_min, y) ~ u(t, x_max, y),
            v(t, x_min, y) ~ v(t, x_max, y),
            Dy(u(t, x, y_min)) ~ 0.0,
            Dy(u(t, x, y_max)) ~ 0.0,
            Dy(v(t, x, y_min)) ~ 0.0,
            Dy(v(t, x, y_max)) ~ 0.0,
        ]
    end

    @test pdesys.bcs == want_bcs
end

@testset "Solve PDE" begin
    pdesys = sys + domain
    dx = dy = 0.5
    discretization = MOLFiniteDifference([x => dx, y => dy], t, approx_order=2, grid_align=center_align)
    prob = discretize(pdesys, discretization)
    sol = solve(prob, Tsit5(), saveat=0.1)
    @test sol.retcode == SciMLBase.ReturnCode.Success
end

@testset "Simplify" begin
    @parameters x
    domain = DomainInfo(
        constIC(16.0, t ∈ Interval(0, 1)),
        periodicBC(x ∈ Interval(0, 1)),
    )

    function ExSys()
        @variables u(t) v(t)
        D = Differential(t)
        ODESystem([
                v ~ 2u,
                D(v) ~ v
            ], t; name=:sys)
    end

    sys_domain = couple(ExSys(), domain)
    sys_mtk = convert(PDESystem, sys_domain)

    discretization = MOLFiniteDifference([x => 10], t, approx_order=2)
    prob = discretize(sys_mtk, discretization)
    sol = solve(prob, Tsit5())
    @test sol.retcode == SciMLBase.ReturnCode.Success
end

@testset "replacement_params" begin
    globalcoords = @variables lat lon
    localcoords = @variables model₊lat model₊lon model₊p

    toreplace, replacements = EarthSciMLBase.replacement_params(localcoords, globalcoords)

    want_toreplace = [model₊lat, model₊lon]
    want_replacements = [lat, lon]
    @test isequal(toreplace, want_toreplace)
    @test isequal(replacements, want_replacements)
end

@testset "xy" begin
    di = DomainInfo(DateTime(2024, 1, 1), DateTime(2024, 1, 1, 3);
        xrange=0:0.1:1, yrange=0:0.1:2)

    @test Symbol.(pvars(di)) == [:x, :y]
    @test grid(di) == [0.0:0.1:1.0, 0.0:0.1:2.0]
    @test time_range(di) == (1.7040672e9, 1.704078e9)
    @test length(di.partial_derivative_funcs) == 0
end

@testset "xy offset" begin
    di = DomainInfo(DateTime(2024, 1, 1), DateTime(2024, 1, 1, 3); offsettime=DateTime(2024, 1, 1),
        xrange=0:0.1:1, yrange=0:0.1:2)

    @test Symbol.(pvars(di)) == [:x, :y]
    @test grid(di) == [0.0:0.1:1.0, 0.0:0.1:2.0]
    @test time_range(di) == (0.0, 10800.0)
    @test length(di.partial_derivative_funcs) == 0
end

@testset "xy level" begin
    di = DomainInfo(DateTime(2024, 1, 1), DateTime(2024, 1, 1, 3); offsettime=DateTime(2024, 1, 1),
        xrange=0:0.1:1, yrange=0:0.1:2, levelrange=1:15)

    @test Symbol.(pvars(di)) == [:x, :y, :lev]
    @test grid(di) == [0.0:0.1:1.0, 0.0:0.1:2.0, 1:15]
    @test time_range(di) == (0.0, 10800.0)
    @test length(di.partial_derivative_funcs) == 0
end

@testset "xy float32" begin
    di = DomainInfo(DateTime(2024, 1, 1), DateTime(2024, 1, 1, 3); offsettime=DateTime(2024, 1, 1),
        xrange=0:0.1:1, yrange=0:0.1:2, levelrange=1:15, dtype=Float32)

    @test Symbol.(pvars(di)) == [:x, :y, :lev]
    @test grid(di) == [0.0f0:0.1f0:1.0f0, 0.0f0:0.1f0:2.0f0, 1.0f0:15.0f0]
    @test time_range(di) == (0.0f0, 10800.0f0)
    @test length(di.partial_derivative_funcs) == 0
end

@testset "lon lat float32" begin
    di = DomainInfo(DateTime(2024, 1, 1), DateTime(2024, 1, 1, 3); offsettime=DateTime(2024, 1, 1),
        lonrange=-2π:π/10:2π, latrange=0:π/10:π, levelrange=1:0.5:10, dtype=Float32)

    @test Symbol.(pvars(di)) == [:lon, :lat, :lev]
    @test grid(di) ≈ [Float32(-2π):Float32(π / 10):Float32(2π), 0:Float32(π / 10):Float32(π), 1.0f0:0.5f0:10.0f0]
    @test time_range(di) == (0.0f0, 10800.0f0)
    @test length(di.partial_derivative_funcs) == 1
end

@testset "lon lat" begin
    di = DomainInfo(DateTime(2024, 1, 1), DateTime(2024, 1, 1, 3); offsettime=DateTime(2024, 1, 1),
        lonrange=-2π:π/10:2π, latrange=0:π/5:π, levelrange=1:0.5:10)

    @test Symbol.(pvars(di)) == [:lon, :lat, :lev]
    @test grid(di) ≈ [-2π:π/10:2π, 0:π/5:π, 1:0.5:10]
    @test time_range(di) == (0.0, 10800.0)
    @test length(di.partial_derivative_funcs) == 1
end

@testset "add pd func" begin
    di = DomainInfo(DateTime(2024, 1, 1), DateTime(2024, 1, 1, 3); offsettime=DateTime(2024, 1, 1),
        lonrange=-2π:π/10:2π, latrange=0:π/5:π, levelrange=1:0.5:10)

    di = add_partial_derivative_func(di, x -> x^2)

    @test Symbol.(pvars(di)) == [:lon, :lat, :lev]
    @test grid(di) ≈ [-2π:π/10:2π, 0:π/5:π, 1:0.5:10]
    @test time_range(di) == (0.0, 10800.0)
    @test length(di.partial_derivative_funcs) == 2
end

@testset "errors" begin
    @test_throws AssertionError DomainInfo(DateTime(2024, 1, 1), DateTime(2024, 1, 1);
        latrange=0:1, lonrange=0:1)
    @test_throws AssertionError DomainInfo(DateTime(2024, 1, 1), DateTime(2024, 1, 1, 3);
        latrange=0:10, lonrange=0:10)
    @test_throws AssertionError DomainInfo(DateTime(2024, 1, 1), DateTime(2024, 1, 1, 3);
        latrange=0:1, xrange=0:10)
    @test_throws AssertionError DomainInfo(DateTime(2024, 1, 1), DateTime(2024, 1, 1, 3);
        latrange=0:1, xrange=0:10, lonrange=0:1)
    @test_throws AssertionError DomainInfo(DateTime(2024, 1, 1), DateTime(2024, 1, 1, 3))
    @test_throws AssertionError DomainInfo(DateTime(2024, 1, 1), DateTime(2024, 1, 1, 3);
        lonrange=0:1)
end
