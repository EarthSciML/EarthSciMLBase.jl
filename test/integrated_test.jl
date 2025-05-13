using Test
using ModelingToolkit, Catalyst, EarthSciMLBase
using ModelingToolkit: t_nounits, D_nounits
using OrdinaryDiffEq: ODEProblem, solve
using SciMLBase: ReturnCode
t = t_nounits
D = D_nounits

struct PhotolysisCoupler
    sys::Any
end
function Photolysis(; name = :Photolysis)
    @variables j_NO2(t)
    eqs = [
        j_NO2 ~ max(sin(t / 86400), 0)
    ]
    ODESystem(eqs, t, [j_NO2], [], name = name,
        metadata = Dict(:coupletype => PhotolysisCoupler))
end

struct ChemistryCoupler
    sys::Any
end
function Chemistry(; name = :Chemistry)
    @parameters jNO2 = 1
    @species NO2(t) = 2
    rxs = [
        Reaction(jNO2, [NO2], [], [1], [])
    ]
    rsys = ReactionSystem(rxs, t, [NO2], [jNO2];
        combinatoric_ratelaws = false, name = name)
    convert(ODESystem, complete(rsys), metadata = Dict(:coupletype => ChemistryCoupler))
end

struct EmissionsCoupler
    sys::Any
end
function Emissions(; name = :Emissions)
    @parameters emis = 1
    @variables NO2(t) = 3
    eqs = [D(NO2) ~ emis]
    ODESystem(eqs, t; name = name,
        metadata = Dict(:coupletype => EmissionsCoupler))
end

function EarthSciMLBase.couple2(c::ChemistryCoupler, p::PhotolysisCoupler)
    c, p = c.sys, p.sys
    c = param_to_var(convert(ODESystem, c), :jNO2)
    ConnectorSystem([c.jNO2 ~ p.j_NO2], c, p)
end

function EarthSciMLBase.couple2(c::ChemistryCoupler, emis::EmissionsCoupler)
    c, emis = c.sys, emis.sys
    operator_compose(convert(ODESystem, c), emis, Dict(
        c.NO2 => emis.NO2,
    ))
end

p = Photolysis()
@testset "Photolysis single" begin
    prob = ODEProblem(structural_simplify(p), [], (0.0, 1.0))
    sol = solve(prob)
    @test sol.retcode == ReturnCode.Success
end

c = Chemistry()
@testset "Chemistry single" begin
    prob = ODEProblem(structural_simplify(c), [], (0.0, 1.0))
    sol = solve(prob)
    @test sol.retcode == ReturnCode.Success
end

e = Emissions()
@testset "Emissions single" begin
    prob = ODEProblem(structural_simplify(e), [], (0.0, 1.0))
    sol = solve(prob)
    @test sol.retcode == ReturnCode.Success
end

@testset "Coupled model" begin
    model = couple(c, p, e)
    sys = convert(ODESystem, model)

    prob = ODEProblem(sys, [], (0.0, 1.0))
    sol = solve(prob, u0 = [1.0])
    @test sol.retcode == ReturnCode.Success
end
