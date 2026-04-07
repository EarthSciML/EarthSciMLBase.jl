using Test
using EarthSciMLBase
using ModelingToolkit, DynamicQuantities, Symbolics
using ModelingToolkit: t, D

@parameters α=1 [unit = u"kg", description = "α description"]
@parameters β=2 [unit = u"kg*s", description = "β description"]
@variables x(t) [unit = u"m", description = "x description"]
@constants onex [unit = u"m", description = "unit x"]
@constants zerox [unit = u"m"]
eq = D(x) ~ α * x / β
@named sys = System([eq], t; metadata = Dict(CoupleType => :metatest))

ii(x, y) = findfirst(isequal(x), collect(y))
isin(x, y) = any(isequal(x), y)
@variables β(t) [unit = u"kg*s", description = "β description"]

@testset "Single substitution" begin
    sys2 = param_to_var(sys, :β)
    @test isin(β, unknowns(sys2)) == true
    @test isin(β, parameters(sys2)) == false
    @test isin(β, Symbolics.get_variables(equations(sys2)[1])) == true
    var = unknowns(sys2)[ii(β, unknowns(sys2))]
    @test Symbolics.getmetadata(var, ModelingToolkit.VariableUnit) == u"kg*s"
    @test Symbolics.getmetadata(var, ModelingToolkit.VariableDescription) == "β description"
    @test getmetadata(sys2, CoupleType, nothing) == :metatest
end

@variables α(t) [unit = u"kg*s", description = "α description"]

@testset "Multiple substitutions" begin
    sys3 = param_to_var(sys, :β, :α)
    @test isin(β, unknowns(sys3)) == true
    @test isin(β, parameters(sys3)) == false
    @test isin(β, Symbolics.get_variables(equations(sys3)[1])) == true
    @test isin(α, unknowns(sys3)) == true
    @test isin(α, parameters(sys3)) == false
    @test isin(α, Symbolics.get_variables(equations(sys3)[1])) == true
end

@testset "events" begin
    @named sys = System([eq], t;
        continuous_events = [x ~ zerox],
        discrete_events = (t == 1.0) => [x ~ Pre(x) + onex]
    )

    sys3 = param_to_var(sys, :β, :α)
    @test length(ModelingToolkit.get_continuous_events(sys3)) == 1
    @test length(ModelingToolkit.get_discrete_events(sys3)) == 1
end

@testset "repeated substitution" begin
    sys2 = EarthSciMLBase.param_to_var(sys, :β)
    sys3 = EarthSciMLBase.param_to_var(sys2, :β)
    @test isequal(sys2, sys3)
end

@testset "PDESystem param_to_var" begin
    using DomainSets

    @parameters px [unit = u"m"]
    @parameters py [unit = u"m"]
    @parameters S = 1.0 [description = "S description", unit = u"m/s"]
    @variables ψ(..) [description = "Level-set function", unit = u"m"]
    Dpx = Differential(px)
    Dpy = Differential(py)

    pde_eq = [D(ψ(t, px, py)) ~ -S * sqrt(Dpx(ψ(t, px, py))^2 + Dpy(ψ(t, px, py))^2)]
    pde_bcs = [ψ(0.0, px, py) ~ sqrt((px - 50.0)^2 + (py - 50.0)^2) - 10.0]
    pde_domains = [t ∈ Interval(0.0, 10.0), px ∈ Interval(0.0, 100.0), py ∈ Interval(0.0, 100.0)]
    pdesys = PDESystem(pde_eq, pde_bcs, pde_domains, [t, px, py], [ψ(t, px, py)], [S];
        name = :pdetest, metadata = Dict(CoupleType => :pdemetatest))

    pdesys2 = param_to_var(pdesys, :S)

    # S should no longer be a parameter
    @test length(pdesys2.ps) == 0

    # The substituted equation should contain S(t, px, py) instead of S
    # (param_to_var creates a variable with all IVs for PDESystem)
    eq_vars = Symbolics.get_variables(equations(pdesys2)[1])
    S_var = only(filter(v -> Symbolics.tosymbol(v, escape = false) == :S, eq_vars))
    @test length(Symbolics.arguments(Symbolics.unwrap(S_var))) == 3  # t, px, py

    # Metadata should be preserved
    @test pdesys2.metadata[CoupleType] == :pdemetatest

    # System structure should be preserved
    @test length(equations(pdesys2)) == 1
    @test length(pdesys2.bcs) == 1  # original IC only; promoted var ICs added by merge_pdesystems
    @test length(pdesys2.ivs) == 3
    # S was promoted from parameter to DV
    @test length(pdesys2.dvs) == 2
    dv_names = [Symbolics.tosymbol(dv, escape = false) for dv in pdesys2.dvs]
    @test :S ∈ dv_names
    @test :ψ ∈ dv_names
end

@testset "PDESystem param_to_var - skip existing variable" begin
    using DomainSets

    @parameters px2 [unit = u"m"]
    @parameters S2 = 1.0 [description = "S2 description", unit = u"m/s"]
    @variables ψ2(..) [description = "ψ2", unit = u"m"]
    @variables S2_var(..) [description = "S2 var", unit = u"m/s"]

    pde_eq = [D(ψ2(t, px2)) ~ -S2 * Differential(px2)(ψ2(t, px2))]
    pde_bcs = [ψ2(0.0, px2) ~ px2]
    pde_domains = [t ∈ Interval(0.0, 10.0), px2 ∈ Interval(0.0, 100.0)]

    # S2_var is already a DV — passing its symbol should be a no-op
    pdesys = PDESystem(pde_eq, pde_bcs, pde_domains, [t, px2],
        [ψ2(t, px2), S2_var(t, px2)], [S2]; name = :pdetest2)

    pdesys2 = param_to_var(pdesys, :S2_var)  # already a DV, should skip
    @test length(pdesys2.ps) == 1  # S2 still a parameter
end

@testset "PDESystem param_to_var - ICs not added (deferred to merge_pdesystems)" begin
    using DomainSets

    @parameters px200 [unit = u"m"]
    @parameters S200 = 1.0 [description = "Speed", unit = u"m/s"]
    @variables ψ200(..) [description = "Level-set", unit = u"m"]

    pde_eq = [D(ψ200(t, px200)) ~ -S200 * Differential(px200)(ψ200(t, px200))]
    pde_bcs = [ψ200(0.0, px200) ~ px200]
    pde_domains = [t ∈ Interval(0.0, 10.0), px200 ∈ Interval(0.0, 100.0)]
    pdesys = PDESystem(pde_eq, pde_bcs, pde_domains, [t, px200], [ψ200(t, px200)], [S200];
        name = :pdetest200)

    pdesys2 = param_to_var(pdesys, :S200)

    # param_to_var should NOT add ICs (they're added by merge_pdesystems instead,
    # because DV names may change during merge/dedup).
    @test length(pdesys2.bcs) == 1  # only original IC
end

@testset "PDESystem param_to_var - initial_conditions forwarded" begin
    using DomainSets
    using ModelingToolkit: t_nounits, D_nounits

    @parameters px200c
    @parameters S200c = 1.0
    @parameters k200c = 2.0
    @variables ψ200c(..)

    pde_eq = [D_nounits(ψ200c(t_nounits, px200c)) ~ -S200c * k200c]
    pde_bcs = [ψ200c(0.0, px200c) ~ px200c]
    pde_domains = [t_nounits ∈ Interval(0.0, 10.0), px200c ∈ Interval(0.0, 100.0)]
    pdesys = PDESystem(pde_eq, pde_bcs, pde_domains, [t_nounits, px200c],
        [ψ200c(t_nounits, px200c)], [S200c, k200c];
        name = :pdetest200c, checks = false,
        initial_conditions = Dict(Symbolics.unwrap(k200c) => 2.0))

    # Promote S200c, keep k200c
    pdesys2 = param_to_var(pdesys, :S200c)

    # k200c should still be in initial_conditions
    @test !isempty(pdesys2.initial_conditions)
end
