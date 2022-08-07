using EarthSciMLBase
using ModelingToolkit, Catalyst
using MethodOfLines, DifferentialEquations

@parameters x y t k=0.1
@variables u(t) q(t)
Dt = Differential(t)

eqs = [
    Dt(u) ~ 2u + 3k * q + 1,
    Dt(q) ~ 3u + k * q + 1
]
@named sys = ODESystem(eqs)

indepdomain = t ∈ (0.0, 1.0)

partialdomains = [x ∈ (1.0, 2.0),
    y ∈ (2.5, 3.0)]

icbc = constICBC(16.0, indepdomain, partialdomains)

@testset "dims" begin
    dims_result = dims(icbc)

    dims_want = [t, x, y]

    @test isequal(dims_result, dims_want)
end

@testset "domains" begin
    domains_result = domains(icbc)

    domains_want = [t ∈ (0.0, 1.0),
        x ∈ (1.0, 2.0),
        y ∈ (2.5, 3.0)]

    @test isequal(domains_result, domains_want)
end

@testset "pde" begin
    pde_want = let
        @parameters x y t k=0.1
        @variables u(..) q(..)
        Dt = Differential(t)

        eqs = [
            Dt(u(t, x, y)) ~ 2u(t, x, y) + 3k * q(t, x, y) + 1,
            Dt(q(t, x, y)) ~ 3u(t, x, y) + k * q(t, x, y) + 1
        ]

        bcs = [u(0.0, x, y) ~ 16.0,
            u(t, 1.0, y) ~ 16.0,
            u(t, 2.0, y) ~ 16.0,
            u(t, x, 2.5) ~ 16.0,
            u(t, x, 3.0) ~ 16.0,
            q(0.0, x, y) ~ 16.0,
            q(t, 1.0, y) ~ 16.0,
            q(t, 2.0, y) ~ 16.0,
            q(t, x, 2.5) ~ 16.0,
            q(t, x, 3.0) ~ 16.0]

        dmns = [t ∈ (0.0, 1.0),
            x ∈ (1.0, 2.0),
            y ∈ (2.5, 3.0)]

        PDESystem(eqs, bcs, dmns, [t, x, y], [u(t, x, y), q(t, x, y)], [k], name=:sys)
    end

    pde_result = sys + icbc

    @test isequal(pde_result.eqs, pde_want.eqs)
    @test isequal(pde_result.ivs, pde_want.ivs)
    @test isequal(pde_result.dvs, pde_want.dvs) # These don't match exactly: u(t, x, y) vs u*
    @test isequal(pde_result.bcs, pde_want.bcs)
    @test isequal(pde_result.domain, pde_want.domain)
    @test isequal(pde_result.ps, pde_want.ps)
end

@testset "ReactionSystem" begin
    pde_want = let
        @parameters β x y t
        @variables m₁(..) m₂(..)
        Dt = Differential(t)
        eqs = [
            Dt(m₁(t, x, y)) ~ -β * m₁(t, x, y),
            Dt(m₂(t, x, y)) ~ β * m₁(t, x, y),
        ]

        bcs = [m₁(0.0, x, y) ~ 16.0,
            m₁(t, 1.0, y) ~ 16.0,
            m₁(t, 2.0, y) ~ 16.0,
            m₁(t, x, 2.5) ~ 16.0,
            m₁(t, x, 3.0) ~ 16.0,
            m₂(0.0, x, y) ~ 16.0,
            m₂(t, 1.0, y) ~ 16.0,
            m₂(t, 2.0, y) ~ 16.0,
            m₂(t, x, 2.5) ~ 16.0,
            m₂(t, x, 3.0) ~ 16.0]

        dmns = [t ∈ (0.0, 1.0),
            x ∈ (1.0, 2.0),
            y ∈ (2.5, 3.0)]

        PDESystem(eqs, bcs, dmns, [t, x, y], [m₁(t, x, y), m₂(t, x, y)], [β], name=:sys)
    end

    rn = @reaction_network begin
        β, m₁ --> m₂
    end β
    pde_result = rn + icbc

    @test isequal(pde_result.eqs, pde_want.eqs)
    @test isequal(pde_result.ivs, pde_want.ivs)
    @test isequal(pde_result.dvs, pde_want.dvs) # These don't match exactly: u(t, x, y) vs u*
    @test isequal(pde_result.bcs, pde_want.bcs)
    @test isequal(pde_result.domain, pde_want.domain)
    @test isequal(pde_result.ps, pde_want.ps)
end

@testset "Solve PDE" begin
    pdesys = sys + icbc
    dx = dy = 0.5
    discretization = MOLFiniteDifference([x=>dx, y=>dy], t, approx_order=2, grid_align=center_align)
    prob = discretize(pdesys,discretization)
    sol = solve(prob, TRBDF2(), saveat=0.1)
end