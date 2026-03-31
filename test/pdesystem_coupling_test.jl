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

    # Coupling: add v at z=0 to u's equation
    coupling = [D_test(u(t_test, x, y)) ~ v(t_test, x, y, 0.0)]
    merged = merge_pdesystems([pde_2d, pde_3d], coupling)

    @test length(equations(merged)) == 2
    @test length(merged.ivs) == 4  # t, x, y, z
    # Verify coupling term was added to u's equation
    u_eq = equations(merged)[findfirst(eq -> occursin("u(t, x, y)", string(eq.lhs)), equations(merged))]
    @test occursin("v(t, x, y, 0.0)", string(u_eq.rhs))
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
    # The equation should have the correct form
    @test string(eq.lhs) == "U(t, x, y)"
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

    @test length(equations(merged)) == 2
    @test length(merged.ivs) == 4  # t, x, y, z
    @test length(merged.dvs) == 2

    # Verify coupling term in u's equation
    u_eq = equations(merged)[findfirst(eq -> occursin("u(t, x, y)", string(eq.lhs)), equations(merged))]
    @test occursin("v(t, x, y, 0.0)", string(u_eq.rhs))
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
    @variables u_md(..) a_md(t_test) = 0.0 b_md(t_test) = 0.0
    @parameters p_a_md = 1.0 p_b_md = 2.0

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
    @variables u_bc(..) a_bc(t_test) = 0.0 b_bc(t_test) = 0.0
    @parameters p_a_bc = 1.0 p_b_bc = 2.0

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
            CoupleType => CrossGroup3DCoupler,
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
            slice_eq,
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
    u_eq = eqs[findfirst(eq -> occursin("u_cg(t, x_cg, y_cg)", string(eq.lhs)) &&
                                occursin("Differential(t", string(eq.lhs)), eqs)]
    # The u_cg equation should have a coupling term (the sliced v_cg)
    @test occursin("v_cg", string(u_eq.rhs))
    # There should be a slice equation with 0.0 substituted
    slice_eq = findfirst(eq -> occursin("0.0", string(eq.rhs)) &&
                               occursin("v_cg", string(eq.rhs)), eqs)
    @test !isnothing(slice_eq)
end

@testset "couple() ODE + PDE with cross-type CoupleType" begin
    @parameters x
    @variables u(..) w(..)

    Dx = Differential(x)

    # ODE system with CoupleType
    struct ODESourceCoupler
        sys
    end

    @variables y(t_test) = 0.5
    @parameters p_ode = 1.0
    ode = System([D_test(y) ~ p_ode], t_test; name = :source,
        metadata = Dict(CoupleType => ODESourceCoupler))

    # PDE system with CoupleType
    struct PDESinkCoupler
        sys
    end

    pde = PDESystem(
        [D_test(u(t_test, x)) ~ Dx(Dx(u(t_test, x)))],
        [u(0, x) ~ 1.0, u(t_test, 0) ~ 0.0, u(t_test, 1) ~ 0.0],
        [t_test ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)],
        [t_test, x], [u(t_test, x)], [];
        name = :sink,
        metadata = Dict(CoupleType => PDESinkCoupler)
    )

    # Cross-type couple2: ODE source drives PDE sink
    function EarthSciMLBase.couple2(a::ODESourceCoupler, b::PDESinkCoupler)
        coupling_eqs = [
            D_test(u(t_test, x)) ~ w(t_test, x),
        ]
        ConnectorSystem(coupling_eqs, a.sys, b.sys)
    end

    domain = DomainInfo(
        constIC(0.0, t_test ∈ Interval(0.0, 1.0)),
        constBC(0.0, x ∈ Interval(0.0, 1.0))
    )

    cs = couple(pde, ode, domain)
    merged = convert(PDESystem, cs)

    eqs = equations(merged)
    # The merged system should have equations from both PDE and promoted ODE
    @test length(eqs) >= 2
    # Both u and y should be dependent variables
    dvs_str = string.(merged.dvs)
    @test any(occursin.("u", dvs_str))
    @test any(occursin.("y", dvs_str))
    # Verify that the cross-type coupling equation was applied
    u_eq = eqs[findfirst(eq -> occursin("u(t, x)", string(eq.lhs)), eqs)]
    @test occursin("w(t, x)", string(u_eq.rhs))
end
