using Test
using Main.EarthSciMLBase
using ModelingToolkit, Catalyst
using MethodOfLines, DifferentialEquations, DomainSets
using ModelingToolkit: t_nounits; t=t_nounits
using ModelingToolkit: D_nounits; D=D_nounits
import SciMLBase

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