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

@testset "validate_and_unify_domains" begin
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

    # Compatible same-dimension domains should succeed
    ivs, doms = EarthSciMLBase.validate_and_unify_domains([pde1, pde2])
    @test length(ivs) == 2  # t, x
    @test length(doms) == 2

    # Incompatible domain range should error
    @parameters y
    @variables w(..)
    pde_bad = PDESystem(
        [D_test(w(t_test, x)) ~ -w(t_test, x)],
        [w(0, x) ~ 1.0, w(t_test, 0) ~ 0.0, w(t_test, 2) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 2.0)],
        [t_test, x], [w(t_test, x)], [];
        name = :pde_bad
    )
    @test_throws ErrorException EarthSciMLBase.validate_and_unify_domains([pde1, pde_bad])

    # Mixed dimensions: 1D (t, x) + 2D (t, x, y) should produce union (t, x, y)
    pde_2d = PDESystem(
        [D_test(w(t_test, x, y)) ~ -w(t_test, x, y)],
        [w(0, x, y) ~ 1.0],
        [t_test ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)],
        [t_test, x, y], [w(t_test, x, y)], [];
        name = :pde_2d
    )
    ivs, doms = EarthSciMLBase.validate_and_unify_domains([pde1, pde_2d])
    @test length(ivs) == 3  # t, x, y (union)
    @test length(doms) == 3
    # Verify ordering: system with most IVs comes first, so t, x, y
    @test Symbol.(ivs) == [:t, :x, :y]
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

    # Coupling equations are added as separate equations (not merged additively).
    # This matches the ODE path where couple2 connector equations are composed
    # via MTK's compose without additive merge.
    coupling = [
        D_test(u(t_test, x)) ~ v(t_test, x),
        D_test(v(t_test, x)) ~ -u(t_test, x)
    ]

    merged = merge_pdesystems([pde1, pde2], coupling)
    eqs = equations(merged)

    @test length(eqs) == 4  # 2 original + 2 coupling
    # Coupling equations should exist as separate equations
    @test any(eq -> isequal(eq.lhs, D_test(u(t_test, x))) &&
                    occursin("v(t, x)", string(eq.rhs)), eqs)
    @test any(eq -> isequal(eq.lhs, D_test(v(t_test, x))) &&
                    occursin("u(t, x)", string(eq.rhs)), eqs)
end

@testset "merge_pdesystems - connector equations not merged additively" begin
    @parameters x_conn
    @variables u_conn(..) v_conn(..)
    @parameters k_conn

    # u_conn has a defining algebraic equation: u_conn ~ k_conn * v_conn
    pde1 = PDESystem(
        [u_conn(t_test, x_conn) ~ k_conn * v_conn(t_test, x_conn)],
        Equation[],
        [t_test ∈ Interval(0.0, 1.0), x_conn ∈ Interval(0.0, 1.0)],
        [t_test, x_conn], [u_conn(t_test, x_conn)], [k_conn];
        name = :pde_conn1, checks = false
    )
    pde2 = PDESystem(
        [D_test(v_conn(t_test, x_conn)) ~ -v_conn(t_test, x_conn)],
        [v_conn(0, x_conn) ~ 1.0, v_conn(t_test, 0) ~ 0.0, v_conn(t_test, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x_conn ∈ Interval(0.0, 1.0)],
        [t_test, x_conn], [v_conn(t_test, x_conn)], [];
        name = :pde_conn2, checks = false
    )

    # Connector equation with same LHS as existing algebraic equation.
    # This should NOT be merged additively (would create self-referential eq).
    connector_eq = [u_conn(t_test, x_conn) ~ 2.0 * v_conn(t_test, x_conn)]

    merged = EarthSciMLBase.merge_pdesystems([pde1, pde2], connector_eq; name = :merged_conn)
    eqs = equations(merged)

    # Both algebraic equations should exist (the original and the connector),
    # NOT a merged self-referential equation.
    for eq in eqs
        eq_str = string(eq)
        # Check no equation has u_conn on BOTH sides
        if contains(string(eq.lhs), "u_conn")
            @test !contains(string(eq.rhs), "u_conn")
        end
    end
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
            D_test(v(t_test, x)) ~ -u(t_test, x)
        ]
        ConnectorSystem(coupling_eqs, a_sys, b_sys)
    end

    cs = couple(pde1, pde2)
    merged = convert(PDESystem, cs)

    eqs = equations(merged)
    @test length(eqs) == 4  # 2 original + 2 coupling (separate equations)
    @test length(merged.dvs) == 2
    @test length(merged.bcs) == 6

    # Verify coupling equations are present as separate equations
    @test any(eq -> isequal(eq.lhs, D_test(u(t_test, x))) &&
                    occursin("v(t, x)", string(eq.rhs)), eqs)
    @test any(eq -> isequal(eq.lhs, D_test(v(t_test, x))) &&
                    occursin("u(t, x)", string(eq.rhs)), eqs)
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

@testset "merge_pdesystems - mixed dimensions (2D + 3D)" begin
    @parameters x y z
    @parameters k
    @variables u(..) v(..)

    Dx = Differential(x)
    Dy = Differential(y)
    Dz = Differential(z)

    # 2D system: u(t, x, y)
    pde_2d = PDESystem(
        [D_test(u(t_test, x, y)) ~ Dx(Dx(u(t_test, x, y))) + Dy(Dy(u(t_test, x, y)))],
        [u(0, x, y) ~ 1.0,
            u(t_test, 0, y) ~ 0.0, u(t_test, 1, y) ~ 0.0,
            u(t_test, x, 0) ~ 0.0, u(t_test, x, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)],
        [t_test, x, y], [u(t_test, x, y)], [];
        name = :pde_2d
    )

    # 3D system: v(t, x, y, z)
    pde_3d = PDESystem(
        [D_test(v(t_test, x, y, z)) ~ Dz(Dz(v(t_test, x, y, z))) - k * v(t_test, x, y, z)],
        [v(0, x, y, z) ~ 0.0,
            v(t_test, 0, y, z) ~ 0.0, v(t_test, 1, y, z) ~ 0.0,
            v(t_test, x, 0, z) ~ 0.0, v(t_test, x, 1, z) ~ 0.0,
            v(t_test, x, y, 0) ~ 0.0, v(t_test, x, y, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0),
            y ∈ Interval(0.0, 1.0), z ∈ Interval(0.0, 1.0)],
        [t_test, x, y, z], [v(t_test, x, y, z)], [k];
        name = :pde_3d
    )

    merged = merge_pdesystems([pde_2d, pde_3d])

    @test length(equations(merged)) == 2
    @test length(merged.dvs) == 2  # u and v
    @test length(merged.ivs) == 4  # t, x, y, z (union)
    @test length(merged.domain) == 4
    @test length(merged.ps) == 1   # k

    # Verify IVs are the union: t, x, y, z
    iv_syms = Symbol.(merged.ivs)
    @test :t ∈ iv_syms
    @test :x ∈ iv_syms
    @test :y ∈ iv_syms
    @test :z ∈ iv_syms

    # u should still be 2D (t, x, y) and v should be 3D (t, x, y, z)
    dvs_str = string.(merged.dvs)
    @test any(s -> occursin("u(t, x, y)", s), dvs_str)
    @test any(s -> occursin("v(t, x, y, z)", s), dvs_str)
end

@testset "merge_pdesystems - mixed dims with coupling equations" begin
    @parameters x y z
    @variables u(..) v(..)

    Dx = Differential(x)

    # 2D system: u(t, x, y)
    pde_2d = PDESystem(
        [D_test(u(t_test, x, y)) ~ Dx(Dx(u(t_test, x, y)))],
        [u(0, x, y) ~ 1.0,
            u(t_test, 0, y) ~ 0.0, u(t_test, 1, y) ~ 0.0,
            u(t_test, x, 0) ~ 0.0, u(t_test, x, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)],
        [t_test, x, y], [u(t_test, x, y)], [];
        name = :pde_2d
    )

    # 3D system: v(t, x, y, z)
    pde_3d = PDESystem(
        [D_test(v(t_test, x, y, z)) ~ -v(t_test, x, y, z)],
        [v(0, x, y, z) ~ 0.0,
            v(t_test, 0, y, z) ~ 0.0, v(t_test, 1, y, z) ~ 0.0,
            v(t_test, x, 0, z) ~ 0.0, v(t_test, x, 1, z) ~ 0.0,
            v(t_test, x, y, 0) ~ 0.0, v(t_test, x, y, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0),
            y ∈ Interval(0.0, 1.0), z ∈ Interval(0.0, 1.0)],
        [t_test, x, y, z], [v(t_test, x, y, z)], [];
        name = :pde_3d
    )

    # Coupling: connector equation added as-is (not merged additively)
    coupling = [D_test(u(t_test, x, y)) ~ v(t_test, x, y, 0.0)]
    merged = merge_pdesystems([pde_2d, pde_3d], coupling)

    @test length(equations(merged)) == 3  # 2 original + 1 coupling
    @test length(merged.ivs) == 4  # t, x, y, z
    # Verify coupling equation is present as a separate equation
    @test any(eq -> occursin("v(t, x, y, 0.0)", string(eq.rhs)), equations(merged))
end

@testset "slice_variable" begin
    @parameters x y lev
    @variables U(..)

    new_dv, eq = slice_variable(U(t_test, x, y, lev), lev, 1.0)

    # New DV should have lev removed
    new_args = Symbolics.arguments(Symbolics.unwrap(new_dv))
    @test length(new_args) == 3  # t, x, y
    @test all(Symbol(a) != :lev for a in new_args)

    # Equation RHS should have lev replaced with 1.0
    @test occursin("1.0", string(eq.rhs))
    # The new DV should have a distinct name encoding the sliced dimension
    @test occursin("U_at_lev", string(eq.lhs))
    @test string(eq.rhs) == "U(t, x, y, 1.0)"
end

@testset "couple() two PDESystems with mixed dimensions and CoupleType" begin
    @parameters x y z
    @variables u(..) v(..)
    Dx = Differential(x)

    struct MixedPDE2DCoupler
        sys
    end
    struct MixedPDE3DCoupler
        sys
    end

    pde_2d = PDESystem(
        [D_test(u(t_test, x, y)) ~ Dx(Dx(u(t_test, x, y)))],
        [u(0, x, y) ~ 1.0,
            u(t_test, 0, y) ~ 0.0, u(t_test, 1, y) ~ 0.0,
            u(t_test, x, 0) ~ 0.0, u(t_test, x, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)],
        [t_test, x, y], [u(t_test, x, y)], [];
        name = :pde_2d,
        metadata = Dict(CoupleType => MixedPDE2DCoupler)
    )

    pde_3d = PDESystem(
        [D_test(v(t_test, x, y, z)) ~ -v(t_test, x, y, z)],
        [v(0, x, y, z) ~ 2.0,
            v(t_test, 0, y, z) ~ 0.0, v(t_test, 1, y, z) ~ 0.0,
            v(t_test, x, 0, z) ~ 0.0, v(t_test, x, 1, z) ~ 0.0,
            v(t_test, x, y, 0) ~ 0.0, v(t_test, x, y, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0),
            y ∈ Interval(0.0, 1.0), z ∈ Interval(0.0, 1.0)],
        [t_test, x, y, z], [v(t_test, x, y, z)], [];
        name = :pde_3d,
        metadata = Dict(CoupleType => MixedPDE3DCoupler)
    )

    # Coupling: add ground-level v to u's equation
    function EarthSciMLBase.couple2(a::MixedPDE3DCoupler, b::MixedPDE2DCoupler)
        a_sys, b_sys = a.sys, b.sys
        coupling_eqs = [D_test(u(t_test, x, y)) ~ v(t_test, x, y, 0.0)]
        ConnectorSystem(coupling_eqs, a_sys, b_sys)
    end

    cs = couple(pde_2d, pde_3d)
    merged = convert(PDESystem, cs)

    @test length(equations(merged)) == 3  # 2 original + 1 coupling
    @test length(merged.ivs) == 4  # t, x, y, z
    @test length(merged.dvs) == 2

    # Verify coupling equation is present
    @test any(eq -> occursin("v(t, x, y, 0.0)", string(eq.rhs)), equations(merged))
end

@testset "SysDomainInfo metadata" begin
    @parameters x_sdi y_sdi
    @variables w_sdi(t_test) = 0.5
    @parameters p_sdi = 1.0

    domain_2d = DomainInfo(
        constIC(0.0, t_test ∈ Interval(0.0, 1.0)),
        constBC(0.0, x_sdi ∈ Interval(0.0, 1.0), y_sdi ∈ Interval(0.0, 1.0))
    )

    sys_no_meta = System([D_test(w_sdi) ~ p_sdi], t_test; name = :no_meta)
    @test isnothing(EarthSciMLBase.get_sys_domaininfo(sys_no_meta))

    sys_with_meta = System([D_test(w_sdi) ~ p_sdi], t_test; name = :with_meta,
        metadata = Dict(SysDomainInfo => domain_2d))
    di = EarthSciMLBase.get_sys_domaininfo(sys_with_meta)
    @test di isa DomainInfo
    @test di === domain_2d
end

@testset "metadata preserved through ODE-to-PDE promotion" begin
    @parameters x_mp
    @variables w_mp(t_test) = 0.5
    @parameters p_mp = 1.0

    struct PromotionTestCoupler
        sys
    end

    domain = DomainInfo(
        constIC(0.0, t_test ∈ Interval(0.0, 1.0)),
        constBC(0.0, x_mp ∈ Interval(0.0, 1.0))
    )

    sys = System([D_test(w_mp) ~ p_mp], t_test; name = :test_promote,
        metadata = Dict(CoupleType => PromotionTestCoupler))

    pde = sys + domain
    ct = EarthSciMLBase.get_coupletype(pde)
    @test ct === PromotionTestCoupler
end

@testset "_group_by_domaininfo" begin
    @parameters x_g y_g z_g
    @variables a_g(t_test) = 0.0
    @variables b_g(t_test) = 0.0
    @variables c_g(t_test) = 0.0

    domain_2d = DomainInfo(
        constIC(0.0, t_test ∈ Interval(0.0, 1.0)),
        constBC(0.0, x_g ∈ Interval(0.0, 1.0), y_g ∈ Interval(0.0, 1.0))
    )
    domain_3d = DomainInfo(
        constIC(0.0, t_test ∈ Interval(0.0, 1.0)),
        constBC(0.0, x_g ∈ Interval(0.0, 1.0), y_g ∈ Interval(0.0, 1.0),
            z_g ∈ Interval(0.0, 1.0))
    )

    sys_a = System([D_test(a_g) ~ 1], t_test; name = :sys_a)
    sys_b = System([D_test(b_g) ~ 1], t_test; name = :sys_b,
        metadata = Dict(SysDomainInfo => domain_3d))
    sys_c = System([D_test(c_g) ~ 1], t_test; name = :sys_c)

    groups = EarthSciMLBase._group_by_domaininfo([sys_a, sys_b, sys_c], domain_2d)
    @test length(groups) == 2

    # sys_a and sys_c should be in the 2D group, sys_b in the 3D group
    for (di, indices) in groups
        if di === domain_2d
            @test sort(indices) == [1, 3]
        elseif di === domain_3d
            @test indices == [2]
        else
            error("unexpected DomainInfo group")
        end
    end

    # All same DomainInfo → single group
    groups_same = EarthSciMLBase._group_by_domaininfo([sys_a, sys_c], domain_2d)
    @test length(groups_same) == 1
    @test first(groups_same)[2] == [1, 2]
end

@testset "couple() ODE systems with different DomainInfos + PDESystem" begin
    @parameters x_md y_md z_md
    @variables u_md(..) a_md(t_test)=0.0 b_md(t_test)=0.0
    @parameters p_a_md=1.0 p_b_md=2.0

    Dx = Differential(x_md)

    # 2D PDE system
    pde_2d = PDESystem(
        [D_test(u_md(t_test, x_md, y_md)) ~ Dx(Dx(u_md(t_test, x_md, y_md)))],
        [u_md(0, x_md, y_md) ~ 1.0,
            u_md(t_test, 0, y_md) ~ 0.0, u_md(t_test, 1, y_md) ~ 0.0,
            u_md(t_test, x_md, 0) ~ 0.0, u_md(t_test, x_md, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x_md ∈ Interval(0.0, 1.0), y_md ∈ Interval(0.0, 1.0)],
        [t_test, x_md, y_md], [u_md(t_test, x_md, y_md)], [];
        name = :pde_2d
    )

    # 2D DomainInfo (default)
    domain_2d = DomainInfo(
        constIC(0.0, t_test ∈ Interval(0.0, 1.0)),
        constBC(0.0, x_md ∈ Interval(0.0, 1.0), y_md ∈ Interval(0.0, 1.0))
    )

    # 3D DomainInfo (for the "data source" system)
    domain_3d = DomainInfo(
        constIC(0.0, t_test ∈ Interval(0.0, 1.0)),
        constBC(0.0, x_md ∈ Interval(0.0, 1.0), y_md ∈ Interval(0.0, 1.0),
            z_md ∈ Interval(0.0, 1.0))
    )

    # ODE system using default 2D domain
    ode_2d = System([D_test(a_md) ~ p_a_md], t_test; name = :ode_2d)

    # ODE system carrying its own 3D domain (like ERA5)
    ode_3d = System([D_test(b_md) ~ p_b_md], t_test; name = :ode_3d,
        metadata = Dict(SysDomainInfo => domain_3d))

    cs = couple(pde_2d, ode_2d, ode_3d, domain_2d)
    @test length(cs.pdesystems) == 1
    @test length(cs.systems) == 2
    @test cs.domaininfo === domain_2d

    merged = convert(PDESystem, cs)

    # The unified IVs should be the union: t, x_md, y_md, z_md
    iv_syms = Symbol.(merged.ivs)
    @test :t ∈ iv_syms
    @test :x_md ∈ iv_syms
    @test :y_md ∈ iv_syms
    @test :z_md ∈ iv_syms
    @test length(merged.ivs) == 4

    # Should have u_md (2D), a_md (2D), b_md (3D) as dependent variables
    dvs_str = string.(merged.dvs)
    @test any(s -> occursin("u_md(t, x_md, y_md)", s), dvs_str)
    @test any(s -> occursin("a_md(t, x_md, y_md)", s), dvs_str)
    @test any(s -> occursin("b_md(t, x_md, y_md, z_md)", s), dvs_str)
end

@testset "backward compat: single DomainInfo fast path" begin
    @parameters x_bc
    @variables u_bc(..) a_bc(t_test)=0.0 b_bc(t_test)=0.0
    @parameters p_a_bc=1.0 p_b_bc=2.0

    Dx = Differential(x_bc)

    pde = PDESystem(
        [D_test(u_bc(t_test, x_bc)) ~ Dx(Dx(u_bc(t_test, x_bc)))],
        [u_bc(0, x_bc) ~ 1.0, u_bc(t_test, 0) ~ 0.0, u_bc(t_test, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x_bc ∈ Interval(0.0, 1.0)],
        [t_test, x_bc], [u_bc(t_test, x_bc)], [];
        name = :pde
    )

    domain = DomainInfo(
        constIC(0.0, t_test ∈ Interval(0.0, 1.0)),
        constBC(0.0, x_bc ∈ Interval(0.0, 1.0))
    )

    ode_a = System([D_test(a_bc) ~ p_a_bc], t_test; name = :ode_a)
    ode_b = System([D_test(b_bc) ~ p_b_bc], t_test; name = :ode_b)

    # Two ODE systems, no SysDomainInfo → single group fast path
    cs = couple(pde, ode_a, ode_b, domain)
    merged = convert(PDESystem, cs)

    @test length(merged.ivs) == 2  # t, x_bc only
    dvs_str = string.(merged.dvs)
    @test any(s -> occursin("u_bc(t, x_bc)", s), dvs_str)
    @test any(s -> occursin("a_bc(t, x_bc)", s), dvs_str)
    @test any(s -> occursin("b_bc(t, x_bc)", s), dvs_str)
end

@testset "cross-group PDE-level coupling via couple2" begin
    @parameters x_cg y_cg z_cg
    @variables u_cg(..) v_cg(t_test) = 0.0
    @parameters p_v_cg = 1.0

    Dx = Differential(x_cg)

    struct CrossGroup2DCoupler
        sys
    end
    struct CrossGroup3DCoupler
        sys
    end

    # 2D PDE system
    pde_2d = PDESystem(
        [D_test(u_cg(t_test, x_cg, y_cg)) ~ Dx(Dx(u_cg(t_test, x_cg, y_cg)))],
        [u_cg(0, x_cg, y_cg) ~ 1.0,
            u_cg(t_test, 0, y_cg) ~ 0.0, u_cg(t_test, 1, y_cg) ~ 0.0,
            u_cg(t_test, x_cg, 0) ~ 0.0, u_cg(t_test, x_cg, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x_cg ∈ Interval(0.0, 1.0), y_cg ∈ Interval(0.0, 1.0)],
        [t_test, x_cg, y_cg], [u_cg(t_test, x_cg, y_cg)], [];
        name = :pde_2d,
        metadata = Dict(CoupleType => CrossGroup2DCoupler)
    )

    # 3D DomainInfo
    domain_3d = DomainInfo(
        constIC(0.0, t_test ∈ Interval(0.0, 1.0)),
        constBC(0.0, x_cg ∈ Interval(0.0, 1.0), y_cg ∈ Interval(0.0, 1.0),
            z_cg ∈ Interval(0.0, 1.0))
    )

    # ODE system with its own 3D DomainInfo, simulating a data source
    ode_3d = System([D_test(v_cg) ~ p_v_cg], t_test; name = :ode_3d,
        metadata = Dict(
            SysDomainInfo => domain_3d,
            CoupleType => CrossGroup3DCoupler
        ))

    # 2D default domain
    domain_2d = DomainInfo(
        constIC(0.0, t_test ∈ Interval(0.0, 1.0)),
        constBC(0.0, x_cg ∈ Interval(0.0, 1.0), y_cg ∈ Interval(0.0, 1.0))
    )

    # Define PDE-level cross-group coupling: add ground-level v_cg to u_cg's equation.
    # The couple2 method receives the promoted PDESystems, so we extract the 3D
    # variable from the DVs and use slice_variable to fix z_cg at ground level.
    function EarthSciMLBase.couple2(a::CrossGroup2DCoupler, b::CrossGroup3DCoupler)
        a_sys, b_sys = a.sys, b.sys
        # Find the 3D dependent variable from the promoted data source system.
        b_v = first(filter(dv -> occursin("v_cg", string(dv)), b_sys.dvs))
        # Slice it at z_cg = 0 to get a 2D version and a defining equation.
        sliced_v, slice_eq = slice_variable(b_v, z_cg, 0.0)
        # Add the sliced variable as a forcing term to u_cg's equation.
        coupling_eqs = [
            D_test(u_cg(t_test, x_cg, y_cg)) ~ sliced_v,
            slice_eq
        ]
        ConnectorSystem(coupling_eqs, a_sys, b_sys)
    end

    cs = couple(pde_2d, ode_3d, domain_2d)
    merged = convert(PDESystem, cs)

    # Should have 4 IVs: t, x_cg, y_cg, z_cg
    @test length(merged.ivs) == 4
    iv_syms = Symbol.(merged.ivs)
    @test :z_cg ∈ iv_syms

    # Verify coupling: u_cg's equation should reference the sliced v_cg variable,
    # and there should be a slice equation defining v_cg(t, x_cg, y_cg) ~ v_cg(t, x_cg, y_cg, 0.0).
    eqs = equations(merged)
    # There should be a coupling equation with D(u_cg) on LHS and v_cg on RHS
    @test any(eq -> occursin("u_cg", string(eq.lhs)) &&
                    occursin("Differential(t", string(eq.lhs)) &&
                    occursin("v_cg", string(eq.rhs)), eqs)
    # There should be a slice equation with 0.0 substituted
    slice_eq = findfirst(
        eq -> occursin("0.0", string(eq.rhs)) &&
              occursin("v_cg", string(eq.rhs)), eqs)
    @test !isnothing(slice_eq)
end

@testset "validate_and_unify_domains - unit mismatch (#187)" begin
    # Create two parameters with the same base name (:x) but different units.
    # We need separate scopes so that the @parameters macro creates distinct objects.
    x_m = let
        @parameters x [unit = u"m"]
        x
    end
    x_km = let
        @parameters x [unit = u"km"]
        x
    end
    @variables u_um(..) v_um(..)

    pde1 = PDESystem(
        [D_test(u_um(t_test, x_m)) ~ -u_um(t_test, x_m)],
        [u_um(0, x_m) ~ 1.0, u_um(t_test, 0) ~ 0.0, u_um(t_test, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x_m ∈ Interval(0.0, 1.0)],
        [t_test, x_m], [u_um(t_test, x_m)], [];
        name = :pde1_um
    )
    pde2 = PDESystem(
        [D_test(v_um(t_test, x_km)) ~ -v_um(t_test, x_km)],
        [v_um(0, x_km) ~ 1.0, v_um(t_test, 0) ~ 0.0, v_um(t_test, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x_km ∈ Interval(0.0, 1.0)],
        [t_test, x_km], [v_um(t_test, x_km)], [];
        name = :pde2_um
    )

    @test_throws ErrorException EarthSciMLBase.validate_and_unify_domains([pde1, pde2])

    # Verify the error message mentions "Unit mismatch"
    try
        EarthSciMLBase.validate_and_unify_domains([pde1, pde2])
        @test false  # Should not reach here
    catch e
        @test occursin("Unit mismatch", e.msg)
        @test occursin("x", e.msg)
    end
end

@testset "unique_syms - different dimensions same name (#188)" begin
    @parameters x_us y_us
    @variables u_us(..)
    # u_us(t, x_us) and u_us(t, x_us, y_us) have the same base name but different dimensions
    syms = [u_us(t_test, x_us), u_us(t_test, x_us, y_us)]
    result = EarthSciMLBase.unique_syms(syms)
    @test length(result) == 2  # Both should be kept
    result_strs = string.(result)
    @test any(s -> occursin("u_us(t, x_us)", s) && !occursin("y_us", s), result_strs)
    @test any(s -> occursin("u_us(t, x_us, y_us)", s), result_strs)

    # True duplicates should still be removed
    syms_dup = [u_us(t_test, x_us), u_us(t_test, x_us)]
    result_dup = EarthSciMLBase.unique_syms(syms_dup)
    @test length(result_dup) == 1
end

@testset "slice_variable dvs in merged PDESystem (#183)" begin
    @parameters x_sv y_sv z_sv
    @variables u_sv(..) v_sv(..)

    Dx_sv = Differential(x_sv)

    # 2D system: u_sv(t, x_sv, y_sv)
    pde_2d = PDESystem(
        [D_test(u_sv(t_test, x_sv, y_sv)) ~ Dx_sv(Dx_sv(u_sv(t_test, x_sv, y_sv)))],
        [u_sv(0, x_sv, y_sv) ~ 1.0,
            u_sv(t_test, 0, y_sv) ~ 0.0, u_sv(t_test, 1, y_sv) ~ 0.0,
            u_sv(t_test, x_sv, 0) ~ 0.0, u_sv(t_test, x_sv, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x_sv ∈ Interval(0.0, 1.0), y_sv ∈ Interval(0.0, 1.0)],
        [t_test, x_sv, y_sv], [u_sv(t_test, x_sv, y_sv)], [];
        name = :pde_2d_sv
    )

    # 3D system: v_sv(t, x_sv, y_sv, z_sv)
    pde_3d = PDESystem(
        [D_test(v_sv(t_test, x_sv, y_sv, z_sv)) ~ -v_sv(t_test, x_sv, y_sv, z_sv)],
        [v_sv(0, x_sv, y_sv, z_sv) ~ 0.0,
            v_sv(t_test, 0, y_sv, z_sv) ~ 0.0, v_sv(t_test, 1, y_sv, z_sv) ~ 0.0,
            v_sv(t_test, x_sv, 0, z_sv) ~ 0.0, v_sv(t_test, x_sv, 1, z_sv) ~ 0.0,
            v_sv(t_test, x_sv, y_sv, 0) ~ 0.0, v_sv(t_test, x_sv, y_sv, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x_sv ∈ Interval(0.0, 1.0),
            y_sv ∈ Interval(0.0, 1.0), z_sv ∈ Interval(0.0, 1.0)],
        [t_test, x_sv, y_sv, z_sv], [v_sv(t_test, x_sv, y_sv, z_sv)], [];
        name = :pde_3d_sv
    )

    # Use slice_variable to create a 2D view of the 3D variable
    sliced_v, slice_eq = slice_variable(v_sv(t_test, x_sv, y_sv, z_sv), z_sv, 0.0)

    # Coupling: add sliced v_sv to u_sv's equation, plus the slice equation
    coupling = [
        D_test(u_sv(t_test, x_sv, y_sv)) ~ sliced_v,
        slice_eq
    ]

    merged = merge_pdesystems([pde_2d, pde_3d], coupling)

    dvs_str = string.(merged.dvs)
    # Original dvs should be present
    @test any(s -> occursin("u_sv(t, x_sv, y_sv)", s), dvs_str)
    @test any(s -> occursin("v_sv(t, x_sv, y_sv, z_sv)", s), dvs_str)

    # The sliced variable has a distinct name (v_sv_at_z_sv_0ₓ0) and should
    # be added to dvs since it doesn't collide with the original v_sv.
    @test any(s -> occursin("v_sv_at_z_sv", s), dvs_str)
    @test length(merged.dvs) == 3  # u_sv, v_sv, and the sliced variant

    # The slice equation should be present in the merged system
    eqs = equations(merged)
    slice_eq_found = findfirst(
        eq -> occursin("v_sv(t, x_sv, y_sv, 0.0)", string(eq.rhs)) &&
              occursin("v_sv_at_z_sv", string(eq.lhs)),
        eqs)
    @test !isnothing(slice_eq_found)

    # There should be a coupling equation with D(u_sv) on LHS and v_sv on RHS
    @test any(eq -> occursin("u_sv", string(eq.lhs)) &&
                    occursin("Differential(t", string(eq.lhs)) &&
                    occursin("v_sv", string(eq.rhs)), eqs)
end

@testset "new DV from coupling equation added to dvs (#183)" begin
    # When a coupling equation introduces a variable with a new base name
    # (not sharing a name with any existing DV), it should be added to dvs.
    @parameters x_n183 k_n183
    @variables u_n183(..) w_n183(..)

    Dx_n183 = Differential(x_n183)

    pde = PDESystem(
        [D_test(u_n183(t_test, x_n183)) ~ Dx_n183(Dx_n183(u_n183(t_test, x_n183)))],
        [u_n183(0, x_n183) ~ 1.0, u_n183(t_test, 0) ~ 0.0, u_n183(t_test, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x_n183 ∈ Interval(0.0, 1.0)],
        [t_test, x_n183], [u_n183(t_test, x_n183)], [];
        name = :pde_n183
    )

    # Coupling introduces a new variable w_n183 not in any PDE's dvs
    coupling = [
        w_n183(t_test, x_n183) ~ k_n183 * u_n183(t_test, x_n183),
    ]
    merged = merge_pdesystems([pde], coupling)

    dvs_str = string.(merged.dvs)
    @test any(s -> occursin("u_n183", s), dvs_str)
    @test any(s -> occursin("w_n183", s), dvs_str)
    @test length(merged.dvs) == 2  # u_n183 and w_n183
end

@testset "cross-type ODE→PDE pre-coupling (WildlandFire pattern)" begin
    # This test mimics WildlandFire.jl's Rothermel→LevelSet coupling pattern:
    # - An ODE system computes a rate R (like Rothermel fire spread rate)
    # - A PDESystem has a parameter S that drives propagation (like level-set speed)
    # - couple2 uses param_to_var to convert S to a variable, then links S ~ R
    #   using sys.varname dot access on the individual ODE system.

    @parameters x_ct y_ct
    @variables ψ_ct(..) [description = "Level-set function"]
    @parameters S_ct = 1.0 [description = "Spread rate"]

    Dx_ct = Differential(x_ct)
    Dy_ct = Differential(y_ct)

    # PDE system (like LevelSetFireSpread): ∂ψ/∂t + S‖∇ψ‖ = 0
    struct LevelSetTestCoupler
        sys
    end

    pde = PDESystem(
        [D_test(ψ_ct(t_test, x_ct, y_ct)) ~
         -S_ct * sqrt(
            Dx_ct(ψ_ct(t_test, x_ct, y_ct))^2 + Dy_ct(ψ_ct(t_test, x_ct, y_ct))^2)],
        [ψ_ct(0, x_ct, y_ct) ~ sqrt((x_ct - 0.5)^2 + (y_ct - 0.5)^2) - 0.1],
        [t_test ∈ Interval(0.0, 1.0),
            x_ct ∈ Interval(0.0, 1.0), y_ct ∈ Interval(0.0, 1.0)],
        [t_test, x_ct, y_ct], [ψ_ct(t_test, x_ct, y_ct)], [S_ct];
        name = :level_set,
        metadata = Dict(CoupleType => LevelSetTestCoupler)
    )

    # ODE system (like RothermelFireSpread): computes rate R from inputs
    struct RothermelTestCoupler
        sys
    end

    @variables R_ct(t_test) = 0.5 [description = "Rate of spread"]
    @parameters k_ct = 2.0 [description = "Rate constant"]

    ode = System([D_test(R_ct) ~ k_ct * (1.0 - R_ct)], t_test; name = :rothermel,
        metadata = Dict(CoupleType => RothermelTestCoupler))

    # Cross-type couple2: follows the EXACT WildlandFire.jl pattern
    # - Uses param_to_var on the PDE system
    # - Accesses variables via sys.varname (dot notation) on the ODE system
    # - Extracts the converted variable from PDE equations
    function EarthSciMLBase.couple2(r::RothermelTestCoupler, ls::LevelSetTestCoupler)
        r_sys, ls_sys = r.sys, ls.sys
        ls_sys = param_to_var(ls_sys, :S_ct)
        eq_vars = collect(Symbolics.get_variables(equations(ls_sys)[1]))
        S_sym = only(filter(v -> Symbolics.tosymbol(v, escape = false) == :S_ct, eq_vars))
        return ConnectorSystem([S_sym ~ r_sys.R_ct], ls_sys, r_sys)
    end

    domain = DomainInfo(
        constIC(0.0, t_test ∈ Interval(0.0, 1.0)),
        constBC(0.0, x_ct ∈ Interval(0.0, 1.0), y_ct ∈ Interval(0.0, 1.0))
    )

    cs = couple(pde, ode, domain)
    merged = convert(PDESystem, cs)

    eqs = equations(merged)
    dvs_str = string.(merged.dvs)

    # Both ψ_ct and R_ct should be dependent variables
    @test any(s -> occursin("ψ_ct", s), dvs_str)
    @test any(s -> occursin("R_ct", s), dvs_str)

    # The PDE equation for ψ_ct should now contain S_ct as a variable, not parameter
    ψ_eq = eqs[findfirst(
        eq -> occursin("ψ_ct", string(eq.lhs)) &&
              occursin("Differential(t", string(eq.lhs)), eqs)]
    @test occursin("S_ct", string(ψ_eq.rhs))

    # S_ct should NOT be a parameter (it was converted by param_to_var)
    ps_str = string.(merged.ps)
    @test !any(s -> occursin("S_ct", s), ps_str)

    # There should be a coupling equation linking S_ct to the promoted R_ct
    coupling_eq = findfirst(eqs) do eq
        lhs_str = string(eq.lhs)
        rhs_str = string(eq.rhs)
        occursin("S_ct", lhs_str) && occursin("R_ct", rhs_str)
    end
    @test !isnothing(coupling_eq)
    # The coupling equation's R_ct should have spatial dims from promotion (check RHS specifically)
    coupling_eq_rhs = string(eqs[coupling_eq].rhs)
    @test occursin("x_ct", coupling_eq_rhs)

    # Verify no temporal-only duplicate DVs (issue #181)
    dvs_names = [string(Symbolics.tosymbol(dv, escape = false)) for dv in merged.dvs]
    # Count occurrences of each bare variable name (ignoring namespace prefix)
    bare_names = [last(split(n, "₊")) for n in dvs_names]
    for bn in unique(bare_names)
        count = sum(x -> x == bn, bare_names)
        @test count == 1 || @warn "Duplicate DV with bare name $bn: found $count times"
    end
end

@testset "Issue #182: metadata merge preserves multiple CoupleTypes" begin
    # Two ODE systems with DIFFERENT CoupleTypes in the same DomainInfo group,
    # coupled with a PDE. Verify that couple2 methods for BOTH types fire.

    @parameters x_182 y_182
    @variables ψ_182(..) [description = "PDE variable"]
    @parameters S_182 = 1.0 [description = "PDE parameter 1"]
    @parameters Q_182 = 1.0 [description = "PDE parameter 2"]

    Dx_182 = Differential(x_182)
    Dy_182 = Differential(y_182)

    struct PDECoupler182
        sys
    end

    pde_182 = PDESystem(
        [D_test(ψ_182(t_test, x_182, y_182)) ~
         -S_182 * ψ_182(t_test, x_182, y_182) - Q_182],
        [ψ_182(0, x_182, y_182) ~ 1.0],
        [t_test ∈ Interval(0.0, 1.0),
            x_182 ∈ Interval(0.0, 1.0), y_182 ∈ Interval(0.0, 1.0)],
        [t_test, x_182, y_182], [ψ_182(t_test, x_182, y_182)], [S_182, Q_182];
        name = :pde_182,
        metadata = Dict(CoupleType => PDECoupler182)
    )

    # ODE system A with CoupleType A
    struct ODE_A_Coupler182
        sys
    end
    @variables R_182(t_test) = 0.5 [description = "Rate A"]
    @parameters k_182 = 2.0

    ode_a_182 = System([D_test(R_182) ~ k_182 * (1.0 - R_182)], t_test;
        name = :ode_a_182,
        metadata = Dict(CoupleType => ODE_A_Coupler182))

    # ODE system B with CoupleType B
    struct ODE_B_Coupler182
        sys
    end
    @variables W_182(t_test) = 0.3 [description = "Rate B"]
    @parameters m_182 = 3.0

    ode_b_182 = System([D_test(W_182) ~ m_182 * (1.0 - W_182)], t_test;
        name = :ode_b_182,
        metadata = Dict(CoupleType => ODE_B_Coupler182))

    # couple2 for ODE_A × PDE: link S_182 ~ R_182
    function EarthSciMLBase.couple2(a::ODE_A_Coupler182, b::PDECoupler182)
        a_sys, b_sys = a.sys, b.sys
        b_sys = param_to_var(b_sys, :S_182)
        eq_vars = collect(Symbolics.get_variables(equations(b_sys)[1]))
        S_sym = only(filter(v -> Symbolics.tosymbol(v, escape = false) == :S_182, eq_vars))
        return ConnectorSystem([S_sym ~ a_sys.R_182], b_sys, a_sys)
    end

    # couple2 for ODE_B × PDE: link Q_182 ~ W_182
    function EarthSciMLBase.couple2(b::ODE_B_Coupler182, p::PDECoupler182)
        b_sys, p_sys = b.sys, p.sys
        p_sys = param_to_var(p_sys, :Q_182)
        eq_vars = collect(Symbolics.get_variables(equations(p_sys)[1]))
        Q_sym = only(filter(v -> Symbolics.tosymbol(v, escape = false) == :Q_182, eq_vars))
        return ConnectorSystem([Q_sym ~ b_sys.W_182], p_sys, b_sys)
    end

    domain_182 = DomainInfo(
        constIC(0.0, t_test ∈ Interval(0.0, 1.0)),
        constBC(0.0, x_182 ∈ Interval(0.0, 1.0), y_182 ∈ Interval(0.0, 1.0))
    )

    cs = couple(pde_182, ode_a_182, ode_b_182, domain_182)
    merged = convert(PDESystem, cs)

    eqs = equations(merged)
    dvs_str = string.(merged.dvs)

    # Both R_182 and W_182 should be dependent variables (both ODE systems promoted)
    @test any(s -> occursin("R_182", s), dvs_str)
    @test any(s -> occursin("W_182", s), dvs_str)

    # The PDE equation should reference both S_182 and Q_182 as variables (not parameters)
    ps_str = string.(merged.ps)
    @test !any(s -> occursin("S_182", s), ps_str)
    @test !any(s -> occursin("Q_182", s), ps_str)

    # There should be coupling equations for BOTH: S_182 ~ R_182 and Q_182 ~ W_182
    coupling_S = findfirst(eqs) do eq
        occursin("S_182", string(eq.lhs)) && occursin("R_182", string(eq.rhs))
    end
    coupling_Q = findfirst(eqs) do eq
        occursin("Q_182", string(eq.lhs)) && occursin("W_182", string(eq.rhs))
    end
    @test !isnothing(coupling_S)  # couple2 for ODE_A (S_182~R_182) should have fired
    @test !isnothing(coupling_Q)  # couple2 for ODE_B (Q_182~W_182) should have fired
end

@testset "Issue #184: namespaced variables in cross-coupling equations" begin
    # Test that coupling equations using dot-notation (sys.varname) work correctly
    # when the variable names are namespaced (e.g. rothermel₊R_ct).
    # This is a more specific version of the WildlandFire pattern test above,
    # verifying that the RHS of coupling equations contains promoted spatial vars.

    @parameters x_184 y_184
    @variables ψ_184(..) [description = "PDE variable"]
    @parameters S_184 = 1.0 [description = "Speed"]

    Dx_184 = Differential(x_184)
    Dy_184 = Differential(y_184)

    struct PDECoupler184
        sys
    end

    pde_184 = PDESystem(
        [D_test(ψ_184(t_test, x_184, y_184)) ~ -S_184 * ψ_184(t_test, x_184, y_184)],
        [ψ_184(0, x_184, y_184) ~ 1.0],
        [t_test ∈ Interval(0.0, 1.0),
            x_184 ∈ Interval(0.0, 1.0), y_184 ∈ Interval(0.0, 1.0)],
        [t_test, x_184, y_184], [ψ_184(t_test, x_184, y_184)], [S_184];
        name = :pde_184,
        metadata = Dict(CoupleType => PDECoupler184)
    )

    struct ODECoupler184
        sys
    end
    @variables R_184(t_test) = 0.5 [description = "Rate"]
    @parameters k_184 = 2.0

    ode_184 = System([D_test(R_184) ~ k_184 * (1.0 - R_184)], t_test;
        name = :ode_184,
        metadata = Dict(CoupleType => ODECoupler184))

    # couple2 uses dot-notation: r_sys.R_184 produces a namespaced variable
    function EarthSciMLBase.couple2(r::ODECoupler184, ls::PDECoupler184)
        r_sys, ls_sys = r.sys, ls.sys
        ls_sys = param_to_var(ls_sys, :S_184)
        eq_vars = collect(Symbolics.get_variables(equations(ls_sys)[1]))
        S_sym = only(filter(v -> Symbolics.tosymbol(v, escape = false) == :S_184, eq_vars))
        # r_sys.R_184 creates a namespaced variable (ode_184₊R_184)
        return ConnectorSystem([S_sym ~ r_sys.R_184], ls_sys, r_sys)
    end

    domain_184 = DomainInfo(
        constIC(0.0, t_test ∈ Interval(0.0, 1.0)),
        constBC(0.0, x_184 ∈ Interval(0.0, 1.0), y_184 ∈ Interval(0.0, 1.0))
    )

    cs = couple(pde_184, ode_184, domain_184)
    merged = convert(PDESystem, cs)

    eqs = equations(merged)
    dvs_str = string.(merged.dvs)

    # R_184 should be a DV (promoted to PDE)
    @test any(s -> occursin("R_184", s), dvs_str)

    # Find the coupling equation S_184 ~ (promoted R_184)
    coupling_eq = findfirst(eqs) do eq
        occursin("S_184", string(eq.lhs)) && occursin("R_184", string(eq.rhs))
    end
    @test !isnothing(coupling_eq)  # Coupling equation S_184 ~ R_184 should exist

    # The RHS should contain spatial dimensions from promotion (x_184, y_184)
    coupling_rhs = string(eqs[coupling_eq].rhs)
    @test occursin("x_184", coupling_rhs)
    @test occursin("y_184", coupling_rhs)

    # Verify no temporal-only duplicate DVs
    dvs_names = [string(Symbolics.tosymbol(dv, escape = false)) for dv in merged.dvs]
    bare_names = [last(split(n, "₊")) for n in dvs_names]
    for bn in unique(bare_names)
        count = sum(x -> x == bn, bare_names)
        @test count == 1
    end
end

@testset "Issue #190: param_to_var connector resolved by merge" begin
    # When param_to_var promotes a parameter to a DV with the same name as a DV
    # in another system, the connector equation linking them should be resolved
    # by merge_pdesystems (substituted to trivial, then removed).
    # This is the pattern that WildlandFire.jl uses for FireSpreadDirection→LevelSet.

    @parameters x_190 y_190
    @variables u_190(..) [description = "PDE state"]
    @parameters R_190 = 1.0 [description = "Rate parameter"]
    Dx_190 = Differential(x_190)

    # PDE system with parameter R_190
    pde_190 = PDESystem(
        [D_test(u_190(t_test, x_190, y_190)) ~ -R_190 * u_190(t_test, x_190, y_190)],
        [u_190(0, x_190, y_190) ~ 1.0],
        [t_test ∈ Interval(0.0, 1.0),
            x_190 ∈ Interval(0.0, 1.0), y_190 ∈ Interval(0.0, 1.0)],
        [t_test, x_190, y_190], [u_190(t_test, x_190, y_190)], [R_190];
        name = :pde_190
    )

    # Another PDE system that defines R_190 as a DV (simulates promoted ODE group)
    @variables R_190_src(..) [description = "Rate from ODE"]
    @parameters k_190 = 2.0
    pde_src = PDESystem(
        [R_190_src(t_test, x_190, y_190) ~ k_190],
        [R_190_src(0, x_190, y_190) ~ 2.0],
        [t_test ∈ Interval(0.0, 1.0),
            x_190 ∈ Interval(0.0, 1.0), y_190 ∈ Interval(0.0, 1.0)],
        [t_test, x_190, y_190], [R_190_src(t_test, x_190, y_190)], [k_190];
        name = :src_190
    )

    # Use param_to_var to convert R_190 to a variable, then create a connector
    pde_190_mod = param_to_var(pde_190, :R_190)
    @test length(pde_190_mod.dvs) == 2  # u_190 + R_190

    # The connector links R_190 (now a DV in pde_190) to R_190_src
    eq_vars = Symbolics.get_variables(equations(pde_190_mod)[1])
    R_sym = only(filter(v -> Symbolics.tosymbol(v, escape = false) == :R_190, eq_vars))
    connector = [R_sym ~ R_190_src(t_test, x_190, y_190)]

    merged = merge_pdesystems([pde_src, pde_190_mod], connector)

    # Key assertion: equation count must equal DV count
    @test length(equations(merged)) == length(merged.dvs)
end

@testset "Issue #190: namespaced DV dedup in merge" begin
    # When param_to_var creates R(t,x,y) in one system and the other system
    # has prefix₊R(t,x,y) (from ODE→PDE promotion with namespace), they share
    # the same base symbol and dimensionality but differ in string representation.
    # merge_pdesystems must unify them.

    @parameters x_190b y_190b
    @parameters k_190b = 2.0

    # System A: has a namespaced DV "R_190b" via a subsystem (simulates promoted ODE group)
    @variables R_190b(..) [description = "Rate"]
    pde_a = PDESystem(
        [R_190b(t_test, x_190b, y_190b) ~ k_190b],
        [R_190b(0, x_190b, y_190b) ~ 2.0],
        [t_test ∈ Interval(0.0, 1.0),
            x_190b ∈ Interval(0.0, 1.0), y_190b ∈ Interval(0.0, 1.0)],
        [t_test, x_190b, y_190b], [R_190b(t_test, x_190b, y_190b)], [k_190b];
        name = :subsys_a
    )

    # System B: has a parameter R_190b that gets converted to DV via param_to_var
    @variables u_190b(..) [description = "PDE state"]
    @parameters R_190b_param = 1.0 [description = "Rate"]
    # Manually create a DV with the same base name R_190b but a different symbolic object
    # (this is what param_to_var does — it creates a new @variables R_190b(t,x,y))
    @variables R_190b_new(..) [description = "Rate promoted"]
    pde_b = PDESystem(
        [D_test(u_190b(t_test, x_190b, y_190b)) ~
         -R_190b_new(t_test, x_190b, y_190b) *
         u_190b(t_test, x_190b, y_190b)],
        [u_190b(0, x_190b, y_190b) ~ 1.0],
        [t_test ∈ Interval(0.0, 1.0),
            x_190b ∈ Interval(0.0, 1.0), y_190b ∈ Interval(0.0, 1.0)],
        [t_test, x_190b, y_190b],
        [u_190b(t_test, x_190b, y_190b), R_190b(t_test, x_190b, y_190b)],
        [];
        name = :pde_b
    )

    # Connector: links pde_b's R_190b to pde_a's R_190b (same base name, same dims)
    connector = [R_190b(t_test, x_190b, y_190b) ~ R_190b(t_test, x_190b, y_190b)]

    merged = merge_pdesystems([pde_a, pde_b], connector)

    # The connector should be resolved as trivial and removed.
    # Equation count must equal DV count.
    @test length(equations(merged)) == length(merged.dvs)
end

@testset "cross-group ODE-ODE coupling with same spatial dimensions (#194)" begin
    # Two ODE systems in different DomainInfo groups (same 1D spatial dimension)
    # with a couple2 method that uses param_to_var.
    @parameters x_194
    @variables a_194(t_test)=0.0 b_194(t_test)=0.0
    @parameters p_a_194=1.0 p_b_194=2.0

    struct CrossGroupA194
        sys
    end
    struct CrossGroupB194
        sys
    end

    # Two separate 1D DomainInfos (same extent, but distinct objects → different groups).
    domain_A = DomainInfo(
        constIC(0.0, t_test ∈ Interval(0.0, 1.0)),
        constBC(0.0, x_194 ∈ Interval(0.0, 1.0))
    )
    domain_B = DomainInfo(
        constIC(0.0, t_test ∈ Interval(0.0, 1.0)),
        constBC(0.0, x_194 ∈ Interval(0.0, 1.0))
    )

    ode_a = System([D_test(a_194) ~ -p_a_194 * a_194], t_test; name = :ode_a_194,
        metadata = Dict(
            SysDomainInfo => domain_A,
            CoupleType => CrossGroupA194
        ))
    ode_b = System([D_test(b_194) ~ -p_b_194], t_test; name = :ode_b_194,
        metadata = Dict(
            SysDomainInfo => domain_B,
            CoupleType => CrossGroupB194
        ))

    # couple2: convert p_a_194 in ode_a to a variable and link it to b_194.
    function EarthSciMLBase.couple2(a::CrossGroupA194, b::CrossGroupB194)
        a_sys = param_to_var(a.sys, :p_a_194)
        ConnectorSystem([a_sys.p_a_194 ~ b.sys.b_194], a_sys, b.sys)
    end

    # Need a PDESystem or DomainInfo to trigger the PDE conversion path.
    # Use a simple PDE so the PDESystem path is exercised.
    Dx194 = Differential(x_194)
    @variables u_194(..)
    pde_194 = PDESystem(
        [D_test(u_194(t_test, x_194)) ~ Dx194(Dx194(u_194(t_test, x_194)))],
        [u_194(0, x_194) ~ 1.0, u_194(t_test, 0) ~ 0.0, u_194(t_test, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x_194 ∈ Interval(0.0, 1.0)],
        [t_test, x_194], [u_194(t_test, x_194)], [];
        name = :pde_194
    )

    cs = couple(pde_194, ode_a, ode_b, domain_A)
    merged = convert(PDESystem, cs)

    # Both ODE systems should be promoted and present in DVs.
    dvs_str = string.(merged.dvs)
    @test any(s -> occursin("a_194", s), dvs_str)
    @test any(s -> occursin("b_194", s), dvs_str)

    # The coupling should produce an algebraic equation linking p_a_194 to b_194.
    # param_to_var converts p_a_194 from parameter to variable, and the connector
    # equation p_a_194 ~ b_194 becomes a separate algebraic equation in the merged PDE.
    eqs = equations(merged)

    # The a_194 equation should reference p_a_194 (the promoted parameter) in its RHS.
    a_eq_idx = findfirst(
        eq -> occursin("a_194", string(eq.lhs)) &&
              occursin("Differential(t", string(eq.lhs)),
        eqs)
    @test !isnothing(a_eq_idx)
    a_eq = eqs[a_eq_idx]
    @test occursin("p_a_194", string(a_eq.rhs))

    # There should be an algebraic equation linking p_a_194 to b_194.
    coupling_eq_idx = findfirst(
        eq -> occursin("p_a_194", string(eq.lhs)) &&
              occursin("b_194", string(eq.rhs)),
        eqs)
    @test !isnothing(coupling_eq_idx)
end

@testset "cross-group ODE-ODE coupling with different dimensions errors (#194)" begin
    # Two ODE systems in groups with different spatial dimensions (1D vs 2D).
    # The couple2 method creates a connector at the ODE level, but after
    # promotion the variables have different numbers of spatial dimensions.
    # This should produce a helpful error.
    @parameters x_194d y_194d
    @variables c_194d(t_test)=0.0 d_194d(t_test)=0.0
    @parameters p_c_194d=1.0 p_d_194d=2.0

    struct CrossGroupC194d
        sys
    end
    struct CrossGroupD194d
        sys
    end

    # 1D domain
    domain_1d = DomainInfo(
        constIC(0.0, t_test ∈ Interval(0.0, 1.0)),
        constBC(0.0, x_194d ∈ Interval(0.0, 1.0))
    )
    # 2D domain
    domain_2d = DomainInfo(
        constIC(0.0, t_test ∈ Interval(0.0, 1.0)),
        constBC(0.0, x_194d ∈ Interval(0.0, 1.0), y_194d ∈ Interval(0.0, 1.0))
    )

    ode_c = System([D_test(c_194d) ~ -p_c_194d * c_194d], t_test; name = :ode_c_194d,
        metadata = Dict(
            SysDomainInfo => domain_1d,
            CoupleType => CrossGroupC194d
        ))
    ode_d = System([D_test(d_194d) ~ -p_d_194d], t_test; name = :ode_d_194d,
        metadata = Dict(
            SysDomainInfo => domain_2d,
            CoupleType => CrossGroupD194d
        ))

    function EarthSciMLBase.couple2(a::CrossGroupC194d, b::CrossGroupD194d)
        a_sys = param_to_var(a.sys, :p_c_194d)
        ConnectorSystem([a_sys.p_c_194d ~ b.sys.d_194d], a_sys, b.sys)
    end

    # Simple PDE to trigger the PDE path.
    Dx194d = Differential(x_194d)
    @variables w_194d(..)
    pde_194d = PDESystem(
        [D_test(w_194d(t_test, x_194d)) ~ Dx194d(Dx194d(w_194d(t_test, x_194d)))],
        [w_194d(0, x_194d) ~ 1.0, w_194d(t_test, 0) ~ 0.0, w_194d(t_test, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x_194d ∈ Interval(0.0, 1.0)],
        [t_test, x_194d], [w_194d(t_test, x_194d)], [];
        name = :pde_194d
    )

    cs = couple(pde_194d, ode_c, ode_d, domain_1d)
    @test_throws ErrorException convert(PDESystem, cs)
    # Verify the error message is helpful.
    try
        convert(PDESystem, cs)
        @test false  # Should not reach here
    catch e
        @test occursin("mismatched", string(e.msg))
        @test occursin("slice_variable", string(e.msg))
    end
end

@testset "cross-group couple2 deferred when method expects PDESystem" begin
    # A couple2 method that accesses .dvs (which only exists on PDESystem)
    # should be silently skipped during Phase 1.25 (ODE-ODE cross-group)
    # and deferred to PDE-phase dispatch.
    @parameters x_defer y_defer
    @variables e_defer(t_test)=0.0 f_defer(t_test)=0.0

    struct DeferTestE
        sys
    end
    struct DeferTestF
        sys
    end

    domain_e = DomainInfo(
        constIC(0.0, t_test ∈ Interval(0.0, 1.0)),
        constBC(0.0, x_defer ∈ Interval(0.0, 1.0))
    )
    domain_f = DomainInfo(
        constIC(0.0, t_test ∈ Interval(0.0, 1.0)),
        constBC(0.0, x_defer ∈ Interval(0.0, 1.0), y_defer ∈ Interval(0.0, 1.0))
    )

    ode_e = System([D_test(e_defer) ~ -e_defer], t_test; name = :ode_e_defer,
        metadata = Dict(SysDomainInfo => domain_e, CoupleType => DeferTestE))
    ode_f = System([D_test(f_defer) ~ -f_defer], t_test; name = :ode_f_defer,
        metadata = Dict(SysDomainInfo => domain_f, CoupleType => DeferTestF))

    # This couple2 accesses .dvs which only exists on PDESystem.
    # It should NOT crash during Phase 1.25; it should be deferred.
    function EarthSciMLBase.couple2(a::DeferTestE, b::DeferTestF)
        dvs = b.sys.dvs  # This will error if sys is an ODE System
        ConnectorSystem(Equation[], a.sys, b.sys)
    end

    @variables w_defer(..)
    Dx_defer = Differential(x_defer)
    pde_defer = PDESystem(
        [D_test(w_defer(t_test, x_defer)) ~ Dx_defer(w_defer(t_test, x_defer))],
        [w_defer(0, x_defer) ~ 1.0, w_defer(t_test, 0) ~ 0.0, w_defer(t_test, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x_defer ∈ Interval(0.0, 1.0)],
        [t_test, x_defer], [w_defer(t_test, x_defer)], [];
        name = :pde_defer
    )

    # This should not throw — Phase 1.25 should catch the error and defer.
    cs = couple(pde_defer, ode_e, ode_f, domain_e)
    @test cs isa CoupledSystem
end

@testset "Issue #198: @constants in connector equations preserved" begin
    @parameters x198
    @variables u198(..)

    pde = PDESystem(
        [D_test(u198(t_test, x198)) ~ -u198(t_test, x198)],
        [u198(0, x198) ~ 1.0, u198(t_test, 0) ~ 0.0, u198(t_test, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x198 ∈ Interval(0.0, 1.0)],
        [t_test, x198], [u198(t_test, x198)], [];
        name = :pde198, checks = false
    )

    @constants c198 = 2.5
    coupling_eqs = [D_test(u198(t_test, x198)) ~ c198]

    merged = EarthSciMLBase.merge_pdesystems([pde], coupling_eqs; name = :merged198)

    # c198 should be in the merged parameters
    ps_names = [string(Symbolics.tosymbol(p, escape = false)) for p in merged.ps]
    @test "c198" in ps_names

    # c198 default should be in initial_conditions
    ics = merged.initial_conditions
    @test !isempty(ics)
    c198_key = first(k for (k, v) in ics if string(Symbolics.tosymbol(Symbolics.wrap(k), escape = false)) == "c198")
    @test Symbolics.value(ics[c198_key]) == 2.5

    # Derivative artifacts should NOT be collected as parameters
    @test !any(n -> occursin("ˍ", n), ps_names)
end

@testset "Issue #199: Namespaced spatial coordinates replaced" begin
    @parameters x199
    @variables u199a(..) v199a(..)

    pde1 = PDESystem(
        [D_test(u199a(t_test, x199)) ~ -u199a(t_test, x199)],
        [u199a(0, x199) ~ 1.0, u199a(t_test, 0) ~ 0.0, u199a(t_test, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x199 ∈ Interval(0.0, 1.0)],
        [t_test, x199], [u199a(t_test, x199)], [];
        name = :pde199a, checks = false
    )

    @parameters subsys199₊x199
    pde2 = PDESystem(
        [D_test(v199a(t_test, x199)) ~ -v199a(t_test, subsys199₊x199)],
        [v199a(0, x199) ~ 1.0, v199a(t_test, 0) ~ 0.0, v199a(t_test, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x199 ∈ Interval(0.0, 1.0)],
        [t_test, x199], [v199a(t_test, x199)], [subsys199₊x199];
        name = :pde199b, checks = false
    )

    merged = EarthSciMLBase.merge_pdesystems([pde1, pde2]; name = :merged199)

    # Namespaced coordinate should be removed from parameters
    ps_names = [string(Symbolics.tosymbol(p, escape = false)) for p in merged.ps]
    @test !("subsys199₊x199" in ps_names)

    # Equations should not contain the namespaced coordinate
    for eq in equations(merged)
        @test !contains(string(eq), "subsys199₊x199")
    end
end

@testset "Issue #201: Parameter defaults through PDE pipeline" begin
    # Test 1: Defaults preserved when promoting ODE to PDE
    @parameters x201
    @parameters k201a = 3.0
    @variables v201a(t_test)
    @named ode201 = System([D_test(v201a) ~ -k201a * v201a], t_test)

    di201 = DomainInfo(
        constIC(0.0, t_test ∈ Interval(0.0, 1.0)),
        constBC(0.0, x201 ∈ Interval(0.0, 1.0))
    )
    promoted = ode201 + di201
    @test !isempty(promoted.initial_conditions)

    # Test 2: Defaults preserved through merge
    @parameters k201b = 5.0
    @variables u201b(..)
    pde201 = PDESystem(
        [D_test(u201b(t_test, x201)) ~ -k201b * u201b(t_test, x201)],
        [u201b(0, x201) ~ 1.0, u201b(t_test, 0) ~ 0.0, u201b(t_test, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x201 ∈ Interval(0.0, 1.0)],
        [t_test, x201], [u201b(t_test, x201)], [k201b];
        name = :pde201, checks = false
    )

    merged = EarthSciMLBase.merge_pdesystems([pde201, promoted]; name = :merged201)
    ics = merged.initial_conditions
    @test !isempty(ics)

    # Both parameter defaults should be present
    ics_strs = Dict(string(Symbolics.tosymbol(Symbolics.wrap(k), escape = false)) => v
                    for (k, v) in ics)
    @test haskey(ics_strs, "k201b")
    @test Symbolics.value(ics_strs["k201b"]) == 5.0
end

@testset "Issue #200: merge_pdesystems adds missing ICs for DVs" begin
    @parameters x200
    @variables u200(..) v200(..)

    pde1 = PDESystem(
        [D_test(u200(t_test, x200)) ~ -u200(t_test, x200)],
        [u200(0, x200) ~ 1.0, u200(t_test, 0) ~ 0.0, u200(t_test, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x200 ∈ Interval(0.0, 1.0)],
        [t_test, x200], [u200(t_test, x200)], [];
        name = :pde200a, checks = false
    )

    # v200 has NO t=0 IC (simulates a param_to_var promoted variable)
    pde2 = PDESystem(
        [D_test(v200(t_test, x200)) ~ -v200(t_test, x200)],
        [v200(t_test, 0) ~ 0.0, v200(t_test, 1) ~ 0.0],  # spatial BCs only, no IC
        [t_test ∈ Interval(0.0, 1.0), x200 ∈ Interval(0.0, 1.0)],
        [t_test, x200], [v200(t_test, x200)], [];
        name = :pde200b, checks = false
    )

    merged = EarthSciMLBase.merge_pdesystems([pde1, pde2]; name = :merged200)

    # v200 should now have a t=0 IC added by merge_pdesystems
    v200_ics = filter(bc -> contains(string(bc.lhs), "v200") &&
                            !contains(string(bc.lhs), "t"),  # IC has numeric t, not symbolic t
                      merged.bcs)
    @test length(v200_ics) >= 1

    # Total BCs: 3 from pde1 + 2 spatial from pde2 + 1 generated IC for v200 = 6
    @test length(merged.bcs) == 6
end

@testset "Issue #200: algebraic DVs don't get spurious ICs" begin
    @parameters x200b
    @variables u200b(..) w200b(..)

    # u200b has a differential equation and an IC
    pde1 = PDESystem(
        [D_test(u200b(t_test, x200b)) ~ -u200b(t_test, x200b)],
        [u200b(0, x200b) ~ 1.0, u200b(t_test, 0) ~ 0.0, u200b(t_test, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x200b ∈ Interval(0.0, 1.0)],
        [t_test, x200b], [u200b(t_test, x200b)], [];
        name = :pde200b1, checks = false
    )

    # w200b is an algebraic variable (no D(w200b), no IC)
    pde2 = PDESystem(
        [w200b(t_test, x200b) ~ 2.0 * u200b(t_test, x200b)],
        Equation[],
        [t_test ∈ Interval(0.0, 1.0), x200b ∈ Interval(0.0, 1.0)],
        [t_test, x200b], [w200b(t_test, x200b)], [];
        name = :pde200b2, checks = false
    )

    merged = EarthSciMLBase.merge_pdesystems([pde1, pde2]; name = :merged200b)

    # w200b should NOT get an IC because it's algebraic (no D(w200b) in equations)
    w200b_ics = filter(merged.bcs) do bc
        s = string(bc.lhs)
        contains(s, "w200b") && !contains(s, "t")
    end
    @test length(w200b_ics) == 0

    # Total BCs should be 3 (from pde1 only, no spurious IC for w200b)
    @test length(merged.bcs) == 3
end
