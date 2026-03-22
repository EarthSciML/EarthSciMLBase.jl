using EarthSciMLBase
using Test
using ModelingToolkit, Symbolics
using ModelingToolkit: t_nounits, D_nounits
using DomainSets
using DynamicQuantities
import SciMLBase

# Use unitless t and D for test PDESystems to avoid unit validation issues.
const t_test = t_nounits
const D_test = D_nounits

@testset "validate_compatible_domains" begin
    @parameters x
    @variables u(..) v(..)

    pde1 = PDESystem(
        [D_test(u(t_test, x)) ~ -u(t_test, x)],
        [u(0, x) ~ 1.0, u(t_test, 0) ~ 0.0, u(t_test, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)],
        [t_test, x], [u(t_test, x)], [];
        name = :pde1
    )
    pde2 = PDESystem(
        [D_test(v(t_test, x)) ~ -v(t_test, x)],
        [v(0, x) ~ 2.0, v(t_test, 0) ~ 0.0, v(t_test, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)],
        [t_test, x], [v(t_test, x)], [];
        name = :pde2
    )

    # Compatible domains should not error
    @test isnothing(EarthSciMLBase.validate_compatible_domains([pde1, pde2]))

    # Incompatible domain range should error
    @parameters y
    @variables w(..)
    pde3 = PDESystem(
        [D_test(w(t_test, y)) ~ -w(t_test, y)],
        [w(0, y) ~ 1.0, w(t_test, 0) ~ 0.0, w(t_test, 2) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 2.0)],
        [t_test, y], [w(t_test, y)], [];
        name = :pde3
    )
    @test_throws ErrorException EarthSciMLBase.validate_compatible_domains([pde1, pde3])
end

@testset "unique_syms" begin
    @parameters a b c
    result = EarthSciMLBase.unique_syms([a, b, a, c, b])
    @test length(result) == 3
end

@testset "unique_eqs" begin
    @parameters x
    @variables u(..)
    eq1 = u(t_test, x) ~ 1.0
    eq2 = u(t_test, x) ~ 2.0
    eq3 = u(t_test, x) ~ 1.0  # duplicate of eq1
    result = EarthSciMLBase.unique_eqs([eq1, eq2, eq3])
    @test length(result) == 2
end

@testset "merge_pdesystems - two PDEs" begin
    @parameters x
    @parameters k
    @variables u(..) v(..)

    Dx = Differential(x)

    pde1 = PDESystem(
        [D_test(u(t_test, x)) ~ Dx(Dx(u(t_test, x))) - k * u(t_test, x)],
        [u(0, x) ~ 1.0, u(t_test, 0) ~ 0.0, u(t_test, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)],
        [t_test, x], [u(t_test, x)], [k];
        name = :pde1
    )
    pde2 = PDESystem(
        [D_test(v(t_test, x)) ~ Dx(Dx(v(t_test, x))) + k * u(t_test, x)],
        [v(0, x) ~ 0.0, v(t_test, 0) ~ 0.0, v(t_test, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)],
        [t_test, x], [v(t_test, x)], [k];
        name = :pde2
    )

    merged = merge_pdesystems([pde1, pde2])

    @test length(equations(merged)) == 2
    @test length(merged.bcs) == 6  # 3 from each
    @test length(merged.dvs) == 2  # u and v
    @test length(merged.ps) == 1   # k (deduplicated)
    @test length(merged.ivs) == 2  # t and x
end

@testset "merge_pdesystems - with coupling equations" begin
    @parameters x
    @variables u(..) v(..)

    Dx = Differential(x)

    pde1 = PDESystem(
        [D_test(u(t_test, x)) ~ Dx(Dx(u(t_test, x)))],
        [u(0, x) ~ 1.0, u(t_test, 0) ~ 0.0, u(t_test, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)],
        [t_test, x], [u(t_test, x)], [];
        name = :pde1
    )
    pde2 = PDESystem(
        [D_test(v(t_test, x)) ~ Dx(Dx(v(t_test, x)))],
        [v(0, x) ~ 0.0, v(t_test, 0) ~ 0.0, v(t_test, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)],
        [t_test, x], [v(t_test, x)], [];
        name = :pde2
    )

    # Coupling: add +v to u's equation and -u to v's equation
    coupling = [
        D_test(u(t_test, x)) ~ v(t_test, x),
        D_test(v(t_test, x)) ~ -u(t_test, x),
    ]

    merged = merge_pdesystems([pde1, pde2], coupling)
    eqs = equations(merged)

    @test length(eqs) == 2
    # Check that coupling terms were added to existing equations
    u_eq = eqs[findfirst(eq -> isequal(eq.lhs, D_test(u(t_test, x))), eqs)]
    v_eq = eqs[findfirst(eq -> isequal(eq.lhs, D_test(v(t_test, x))), eqs)]
    u_rhs_str = string(u_eq.rhs)
    v_rhs_str = string(v_eq.rhs)
    @test occursin("v(t, x)", u_rhs_str)
    @test occursin("u(t, x)", v_rhs_str)
end

@testset "couple() accepts PDESystems" begin
    @parameters x
    @variables u(..)

    pde = PDESystem(
        [D_test(u(t_test, x)) ~ -u(t_test, x)],
        [u(0, x) ~ 1.0, u(t_test, 0) ~ 0.0, u(t_test, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)],
        [t_test, x], [u(t_test, x)], [];
        name = :pde
    )

    cs = couple(pde)
    @test length(cs.pdesystems) == 1
    @test length(cs.systems) == 0
end

@testset "couple() two PDESystems with CoupleType" begin
    @parameters x
    @variables u(..) v(..)
    Dx = Differential(x)

    struct PDE1Coupler
        sys
    end
    struct PDE2Coupler
        sys
    end

    pde1 = PDESystem(
        [D_test(u(t_test, x)) ~ Dx(Dx(u(t_test, x)))],
        [u(0, x) ~ 1.0, u(t_test, 0) ~ 0.0, u(t_test, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)],
        [t_test, x], [u(t_test, x)], [];
        name = :pde1,
        metadata = Dict(CoupleType => PDE1Coupler)
    )
    pde2 = PDESystem(
        [D_test(v(t_test, x)) ~ Dx(Dx(v(t_test, x)))],
        [v(0, x) ~ 0.0, v(t_test, 0) ~ 0.0, v(t_test, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)],
        [t_test, x], [v(t_test, x)], [];
        name = :pde2,
        metadata = Dict(CoupleType => PDE2Coupler)
    )

    # Define coupling: u and v interact
    function EarthSciMLBase.couple2(a::PDE1Coupler, b::PDE2Coupler)
        a_sys, b_sys = a.sys, b.sys
        # Add +v to u's equation and -u to v's equation
        coupling_eqs = [
            D_test(u(t_test, x)) ~ v(t_test, x),
            D_test(v(t_test, x)) ~ -u(t_test, x),
        ]
        ConnectorSystem(coupling_eqs, a_sys, b_sys)
    end

    cs = couple(pde1, pde2)
    merged = convert(PDESystem, cs)

    eqs = equations(merged)
    @test length(eqs) == 2
    @test length(merged.dvs) == 2
    @test length(merged.bcs) == 6

    # Verify coupling terms are present in equations
    u_eq = eqs[findfirst(eq -> isequal(eq.lhs, D_test(u(t_test, x))), eqs)]
    v_eq = eqs[findfirst(eq -> isequal(eq.lhs, D_test(v(t_test, x))), eqs)]
    @test occursin("v(t, x)", string(u_eq.rhs))
    @test occursin("u(t, x)", string(v_eq.rhs))
end

@testset "couple() PDESystem + ODE System + DomainInfo" begin
    @parameters x
    @parameters D_diff
    @variables u(..)

    Dx = Differential(x)

    pde = PDESystem(
        [D_test(u(t_test, x)) ~ D_diff * Dx(Dx(u(t_test, x)))],
        [u(0, x) ~ 1.0, u(t_test, 0) ~ 0.0, u(t_test, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)],
        [t_test, x], [u(t_test, x)], [D_diff];
        name = :diffusion
    )

    @variables y(t_test) = 0.5
    @parameters p = 1.0
    ode = System([D_test(y) ~ p], t_test; name = :source)

    domain = DomainInfo(
        constIC(0.0, t_test ∈ Interval(0.0, 1.0)),
        constBC(0.0, x ∈ Interval(0.0, 1.0))
    )

    cs = couple(pde, ode, domain)
    @test length(cs.pdesystems) == 1
    @test length(cs.systems) == 1
    @test cs.domaininfo !== nothing

    merged = convert(PDESystem, cs)

    # The merged system should have equations from both systems
    @test length(equations(merged)) >= 2
    # Should have both u and y as dependent variables
    dvs_str = string.(merged.dvs)
    @test any(occursin.("u", dvs_str))
    @test any(occursin.("y", dvs_str))
end

@testset "convert(System, ...) errors with PDESystems" begin
    @parameters x
    @variables u(..)

    pde = PDESystem(
        [D_test(u(t_test, x)) ~ -u(t_test, x)],
        [u(0, x) ~ 1.0, u(t_test, 0) ~ 0.0, u(t_test, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)],
        [t_test, x], [u(t_test, x)], [];
        name = :pde
    )

    cs = couple(pde)
    @test_throws ErrorException convert(System, cs)
end

@testset "single PDESystem passthrough" begin
    @parameters x
    @variables u(..)

    pde = PDESystem(
        [D_test(u(t_test, x)) ~ -u(t_test, x)],
        [u(0, x) ~ 1.0, u(t_test, 0) ~ 0.0, u(t_test, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)],
        [t_test, x], [u(t_test, x)], [];
        name = :pde
    )

    cs = couple(pde)
    merged = convert(PDESystem, cs)

    # Single system should pass through unchanged
    @test length(equations(merged)) == 1
    @test length(merged.dvs) == 1
    @test length(merged.bcs) == 3
end
