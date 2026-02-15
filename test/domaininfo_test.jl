using Test
using EarthSciMLBase
using EarthSciMLBase: pvars, grid, get_tspan, get_tspan_datetime,
                      add_partial_derivative_func
using ModelingToolkit
using OrdinaryDiffEqTsit5, DomainSets
t = ModelingToolkit.t_nounits;
D = ModelingToolkit.D_nounits;
import SciMLBase
using Dates

@parameters x y α=10.0
@variables u(t) v(t)

x_min = y_min = t_min = 0.0
x_max = y_max = 1.0
t_max = 11.5

eqs = [D(u) ~ -α * √abs(v),
    D(v) ~ -α * √abs(u)
]

@named sys = System(eqs, t)

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
        y ∈ Interval(y_min, y_max)
    ]

    @test isequal(domains_result, domains_want)
end

@testset "pde" begin
    pde_want = let
        @parameters x y α=10.0
        @variables u(..) v(..)

        x_min = y_min = t_min = 0.0
        x_max = y_max = 1.0
        t_max = 11.5

        eqs = [D(u(t, x, y)) ~ -α * √abs(v(t, x, y)),
            D(v(t, x, y)) ~ -α * √abs(u(t, x, y))
        ]

        domains = [
            t ∈ Interval(t_min, t_max),
            x ∈ Interval(x_min, x_max),
            y ∈ Interval(y_min, y_max)
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
            v(t, x, y_max) ~ 16.0
        ]

        @named pdesys = PDESystem(
            eqs, bcs, domains, [t, x, y], [u(t, x, y), v(t, x, y)], [α])
    end

    pde_result = sys + domain

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
        zerogradBC(y ∈ Interval(y_min, y_max))
    )
    pdesys = sys + domain

    want_bcs = let
        @parameters x y
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
            Dy(v(t, x, y_max)) ~ 0.0
        ]
    end

    @test pdesys.bcs == want_bcs
end

@testset "Solve PDE" begin
    # MethodOfLines is not yet compatible with Symbolics v7/MTK v11
end

@testset "Simplify" begin
    # MethodOfLines is not yet compatible with Symbolics v7/MTK v11
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
        xrange = 0:0.1:1, yrange = 0:0.1:2)

    @test Symbol.(pvars(di)) == [:x, :y]
    @test grid(di) == [0.0:0.1:1.0, 0.0:0.1:2.0]
    @test grid(di, (true, false)) == [-0.05:0.1:1.05, 0.0:0.1:2.0]
    @test get_tspan(di) == (0.0, 10800.0)
    @test length(di.partial_derivative_funcs) == 0
end

@testset "xy staggered" begin
    di = DomainInfo(DateTime(2024, 1, 1), DateTime(2024, 1, 1, 3);
        xrange = 0:0.1:1, yrange = 0:0.1:2, u_proto = zeros(Float32, 1, 1, 1, 1))

    @test Symbol.(pvars(di)) == [:x, :y]
    @test grid(di) == [0.0:0.1:1.0, 0.0:0.1:2.0]
    @test grid(di, (true, false)) ≈ [-0.05:0.1:1.05, 0.0:0.1:2.0]
    @test get_tspan(di) == (0.0, 10800.0)
    @test length(di.partial_derivative_funcs) == 0
end

@testset "t_ref" begin
    s, e = DateTime(2024, 1, 1), DateTime(2024, 1, 1, 3)
    di = DomainInfo(s, e; xrange = 0:0.1:1, yrange = 0:0.1:2)
    @test get_tspan(di) == (0.0, 10800.0)

    di = DomainInfo(s, e; tref = 0.0, xrange = 0:0.1:1, yrange = 0:0.1:2)
    @test get_tspan(di) == (1.7040672e9, 1.704078e9)

    t_ref = DateTime(2024, 1, 1, 2)
    di = DomainInfo(s, e; tref = t_ref, xrange = 0:0.1:1, yrange = 0:0.1:2)
    @test get_tspan(di) == (1.7040672e9, 1.704078e9) .- datetime2unix(t_ref)

    t_ref = datetime2unix(t_ref)
    di = DomainInfo(s, e; tref = t_ref,
        xrange = 0:0.1:1, yrange = 0:0.1:2)
    @test get_tspan(di) == (1.7040672e9, 1.704078e9) .- t_ref

    @test get_tspan_datetime(di) == (s, e)

    @test get_tref(di) == t_ref
end

@testset "xy level" begin
    di = DomainInfo(
        DateTime(2024, 1, 1), DateTime(2024, 1, 1, 3);
        xrange = 0:0.1:1, yrange = 0:0.1:2, levrange = 1:15)

    @test Symbol.(pvars(di)) == [:x, :y, :lev]
    @test grid(di) == [0.0:0.1:1.0, 0.0:0.1:2.0, 1:15]
    @test get_tspan(di) == (0.0, 10800.0)
    @test length(di.partial_derivative_funcs) == 0
end

@testset "xy float32" begin
    di = DomainInfo(
        DateTime(2024, 1, 1), DateTime(2024, 1, 1, 3);
        xrange = 0:0.1:1, yrange = 0:0.1:2, levrange = 1:15,
        u_proto = zeros(Float32, 1, 1, 1, 1))

    @test Symbol.(pvars(di)) == [:x, :y, :lev]
    @test grid(di) == [0.0:0.1:1.0, 0.0:0.1:2.0, 1.0:15.0]
    @test get_tspan(di) == (0.0, 10800.0)
    @test length(di.partial_derivative_funcs) == 0
end

@testset "lon lat float32" begin
    di = DomainInfo(
        DateTime(2024, 1, 1), DateTime(2024, 1, 1, 3);
        lonrange = (-2π):(π / 10):(2π), latrange = 0:(π / 10):π, levrange = 1:0.5:10,
        u_proto = zeros(Float32, 1, 1, 1, 1))

    @test Symbol.(pvars(di)) == [:lon, :lat, :lev]
    @test grid(di) ≈ [Float32(-2π):Float32(π / 10):Float32(2π),
        0:Float32(π / 10):Float32(π), 1.0f0:0.5f0:10.0f0]
    @test get_tspan(di) == (0.0f0, 10800.0f0)
    @test length(di.partial_derivative_funcs) == 1
end

@testset "lon lat" begin
    di = DomainInfo(
        DateTime(2024, 1, 1), DateTime(2024, 1, 1, 3);
        lonrange = (-2π):(π / 10):(2π), latrange = 0:(π / 5):π, levrange = 1:0.5:10)

    @test Symbol.(pvars(di)) == [:lon, :lat, :lev]
    @test grid(di) ≈ [(-2π):(π / 10):(2π), 0:(π / 5):π, 1:0.5:10]
    @test get_tspan(di) == (0.0, 10800.0)
    @test length(di.partial_derivative_funcs) == 1
end

@testset "add pd func" begin
    di = DomainInfo(
        DateTime(2024, 1, 1), DateTime(2024, 1, 1, 3);
        lonrange = (-2π):(π / 10):(2π), latrange = 0:(π / 5):π, levrange = 1:0.5:10)

    di = add_partial_derivative_func(di, x -> x^2)

    @test Symbol.(pvars(di)) == [:lon, :lat, :lev]
    @test grid(di) ≈ [(-2π):(π / 10):(2π), 0:(π / 5):π, 1:0.5:10]
    @test get_tspan(di) == (0.0, 10800.0)
    @test length(di.partial_derivative_funcs) == 2
end

@testset "errors" begin
    @test_throws AssertionError DomainInfo(DateTime(2024, 1, 1), DateTime(2024, 1, 1);
        latrange = 0:1, lonrange = 0:1)
    @test_throws AssertionError DomainInfo(DateTime(2024, 1, 1), DateTime(2024, 1, 1, 3);
        latrange = 0:10, lonrange = 0:10)
    @test_throws AssertionError DomainInfo(DateTime(2024, 1, 1), DateTime(2024, 1, 1, 3);
        latrange = 0:1, xrange = 0:10)
    @test_throws AssertionError DomainInfo(DateTime(2024, 1, 1), DateTime(2024, 1, 1, 3);
        latrange = 0:1, xrange = 0:10, lonrange = 0:1)
    @test_throws AssertionError DomainInfo(DateTime(2024, 1, 1), DateTime(2024, 1, 1, 3))
    @test_throws AssertionError DomainInfo(DateTime(2024, 1, 1), DateTime(2024, 1, 1, 3);
        lonrange = 0:1)
end
