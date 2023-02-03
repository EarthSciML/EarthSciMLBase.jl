using EarthSciMLBase
using ModelingToolkit, Catalyst
using MethodOfLines, DifferentialEquations, DomainSets

@parameters x y t α = 10.0
@variables u(t) v(t)
Dt = Differential(t)

x_min = y_min = t_min = 0.0
x_max = y_max = 1.0
t_max = 11.5

eqs = [Dt(u) ~ -α * √abs(v),
    Dt(v) ~ -α * √abs(u),
]

@named sys = ODESystem(eqs)

indepdomain = t ∈ Interval(t_min, t_max)

partialdomains = [x ∈ Interval(x_min, x_max),
    y ∈ Interval(y_min, y_max)]

domain = DomainInfo(constBC(16.0, partialdomains...), constIC(16.0, indepdomain))

@testset "dims" begin
    dims_result = EarthSciMLBase.dims(domain)

    dims_want = [x, y, t]

    @test isequal(dims_result, dims_want)
end

@testset "domains" begin
    domains_result = EarthSciMLBase.domains(domain)

    domains_want = [
        x ∈ Interval(x_min, x_max),
        y ∈ Interval(y_min, y_max),
        t ∈ Interval(t_min, t_max),
    ]

    @test isequal(domains_result, domains_want)
end

@testset "pde" begin
    pde_want = let
        @parameters x y t α = 10.0
        @variables u(..) v(..)
        Dt = Differential(t)

        x_min = y_min = t_min = 0.0
        x_max = y_max = 1.0
        t_max = 11.5

        eqs = [Dt(u(x, y, t)) ~ -α * √abs(v(x, y, t)),
            Dt(v(x, y, t)) ~ -α * √abs(u(x, y, t)),
        ]

        domains = [x ∈ Interval(x_min, x_max),
            y ∈ Interval(y_min, y_max),
            t ∈ Interval(t_min, t_max)]

        # Periodic BCs
        bcs = [
            u(x_min, y, t) ~ 16.0,
            u(x_max, y, t) ~ 16.0,
            u(x, y_min, t) ~ 16.0,
            u(x, y_max, t) ~ 16.0, v(x_min, y, t) ~ 16.0,
            v(x_max, y, t) ~ 16.0,
            v(x, y_min, t) ~ 16.0,
            v(x, y_max, t) ~ 16.0, u(x, y, t_min) ~ 16.0,
            v(x, y, t_min) ~ 16.0,
        ]

        @named pdesys = PDESystem(eqs, bcs, domains, [x, y, t], [u(x, y, t), v(x, y, t)], [α => 10.0])
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
        @parameters x y t
        @variables m₁(..) m₂(..)
        Dt = Differential(t)
        eqs = [
            Dt(m₁(x, y, t)) ~ -10.0 * m₁(x, y, t),
            Dt(m₂(x, y, t)) ~ 10.0 * m₁(x, y, t),
        ]

        bcs = [
            m₁(x_min, y, t) ~ 16.0,
            m₁(x_max, y, t) ~ 16.0,
            m₁(x, y_min, t) ~ 16.0,
            m₁(x, y_max, t) ~ 16.0, m₂(x_min, y, t) ~ 16.0,
            m₂(x_max, y, t) ~ 16.0,
            m₂(x, y_min, t) ~ 16.0,
            m₂(x, y_max, t) ~ 16.0, m₁(x, y, t_min) ~ 16.0,
            m₂(x, y, t_min) ~ 16.0,
        ]

        dmns = [
            x ∈ Interval(x_min, x_max),
            y ∈ Interval(y_min, y_max),
            t ∈ Interval(t_min, t_max),
        ]

        PDESystem(eqs, bcs, dmns, [x, y, t], [m₁(x, y, t), m₂(x, y, t)], [], name=:sys)
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
        periodicBC(x ∈ Interval(x_min, x_max)),
        zerogradBC(y ∈ Interval(y_min, y_max)),
        constIC(16.0, indepdomain),
    )
    pdesys = sys + domain

    want_bcs = let
        @parameters x y t
        @variables u(..) v(..)
        Dt = Differential(t)
        [
            u(x_min, y, t) ~ u(x_max, y, t),
            v(x_min, y, t) ~ v(x_max, y, t),
            Dt(u(x, y_min, t)) ~ 0.0,
            Dt(u(x, y_max, t)) ~ 0.0,
            Dt(v(x, y_min, t)) ~ 0.0,
            Dt(v(x, y_max, t)) ~ 0.0, u(x, y, t_min) ~ 16.0,
            v(x, y, t_min) ~ 16.0,
        ]
    end

    @test pdesys.bcs == want_bcs
end

@testset "Solve PDE" begin
    pdesys = sys + domain
    dx = dy = 0.5
    discretization = MOLFiniteDifference([x => dx, y => dy], t, approx_order=2, grid_align=center_align)
    prob = discretize(pdesys, discretization)
    sol = solve(prob, TRBDF2(), saveat=0.1)
    @test sol.retcode == :Success
end

@testset "Simplify" begin
    @parameters x, t
    domain = DomainInfo(
        periodicBC(x ∈ Interval(0, 1)),
        constIC(16.0, t ∈ Interval(0, 1)),
    )

    struct ExSys <: EarthSciMLODESystem
        sys
        function ExSys(t)
            @variables u(t) v(t)
            D = Differential(t)
            new(ODESystem([
                    v ~ 2u,
                    D(v) ~ v
                ], t; name=:sys))
        end
    end

    sys_domain = ExSys(t) + domain
    sys_mtk = get_mtk(sys_domain)

    discretization = MOLFiniteDifference([x => 10], t, approx_order=2)
    prob = discretize(sys_mtk, discretization)
    sol = solve(prob, Tsit5())
    @test sol.retcode == :Success
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