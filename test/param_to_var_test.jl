using EarthSciMLBase, ModelingToolkit, Unitful, Symbolics

@parameters α=1 [unit = u"kg", description="α description"] 
@parameters β=2 [unit = u"kg*s", description="β description"]
@variables t [unit=u"s", description="time"]
@variables x(t) [unit=u"m", description="x description"]
eq = Differential(t)(x) ~ α * x / β
@named sys = ODESystem([eq])

ii(x, y) = findfirst(isequal(x), y)
isin(x, y) = ii(x, y) !== nothing
@variables β(t) [unit = u"kg*s", description="β description"]


@testset "Single substitution" begin
    sys2 = param_to_var(sys, :β)
    @test isin(β, states(sys2)) == true
    @test isin(β, parameters(sys2)) == false
    @test isin(β, Symbolics.get_variables(equations(sys2)[1])) == true
    var = states(sys2)[ii(β, states(sys2))]
    @test Symbolics.getmetadata(var, ModelingToolkit.VariableUnit) == u"kg*s"
    @test Symbolics.getmetadata(var, ModelingToolkit.VariableDescription) == "β description"
end

@variables α(t) [unit = u"kg*s", description="α description"]

@testset "Multiple substitutions" begin
    sys3 = param_to_var(sys, :β, :α)
    @test isin(β, states(sys3)) == true
    @test isin(β, parameters(sys3)) == false
    @test isin(β, Symbolics.get_variables(equations(sys3)[1])) == true
    @test isin(α, states(sys3)) == true
    @test isin(α, parameters(sys3)) == false
    @test isin(α, Symbolics.get_variables(equations(sys3)[1])) == true
end