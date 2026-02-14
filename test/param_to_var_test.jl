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
