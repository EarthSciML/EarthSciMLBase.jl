using Test
using Main.EarthSciMLBase, ModelingToolkit, DynamicQuantities, Symbolics
using ModelingToolkit: t, D

@parameters α = 1 [unit = u"kg", description = "α description"]
@parameters β = 2 [unit = u"kg*s", description = "β description"]
@variables x(t) [unit = u"m", description = "x description"]
eq = D(x) ~ α * x / β
@named sys = ODESystem([eq], t; metadata=:metatest)

ii(x, y) = findfirst(isequal(x), y)
isin(x, y) = ii(x, y) !== nothing
@variables β(t) [unit = u"kg*s", description = "β description"]


@testset "Single substitution" begin
    sys2 = param_to_var(sys, :β)
    @test isin(β, unknowns(sys2)) == true
    @test isin(β, parameters(sys2)) == false
    @test isin(β, Symbolics.get_variables(equations(sys2)[1])) == true
    var = unknowns(sys2)[ii(β, unknowns(sys2))]
    @test Symbolics.getmetadata(var, ModelingToolkit.VariableUnit) == u"kg*s"
    @test Symbolics.getmetadata(var, ModelingToolkit.VariableDescription) == "β description"
    @test ModelingToolkit.get_metadata(sys2) == :metatest
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
    @named sys = ODESystem([eq], t; metadata=:metatest,
        continuous_events=[x ~ 0],
        discrete_events=(t == 1.0) => [x ~ x + 1],
    )

    sys3 = param_to_var(sys, :β, :α)
    @test length(ModelingToolkit.get_continuous_events(sys3)) == 1
    @test length(ModelingToolkit.get_discrete_events(sys3)) == 1
end
