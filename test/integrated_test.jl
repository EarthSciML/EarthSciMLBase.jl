using Test
using ModelingToolkit, EarthSciMLBase
using ModelingToolkit: t_nounits, D_nounits
using OrdinaryDiffEqTsit5
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
    System(eqs, t, [j_NO2], [], name = name,
        metadata = Dict(CoupleType => PhotolysisCoupler))
end

struct ChemistryCoupler
    sys::Any
end
function Chemistry(; name = :Chemistry)
    @parameters jNO2 = 1
    @variables NO2(t) = 2
    eqs = [
        D(NO2) ~ -jNO2 * NO2,
    ]
    System(eqs, t, [NO2], [jNO2]; name = name,
        metadata = Dict(CoupleType => ChemistryCoupler))
end

struct EmissionsCoupler
    sys::Any
end
function Emissions(; name = :Emissions)
    @parameters emis = 1
    @variables NO2(t)
    eqs = [D(NO2) ~ emis]
    System(eqs, t; name = name,
        metadata = Dict(CoupleType => EmissionsCoupler))
end

function EarthSciMLBase.couple2(c::ChemistryCoupler, p::PhotolysisCoupler)
    c, p = c.sys, p.sys
    c = param_to_var(convert(System, c), :jNO2)
    ConnectorSystem([c.jNO2 ~ p.j_NO2], c, p)
end

function EarthSciMLBase.couple2(c::ChemistryCoupler, emis::EmissionsCoupler)
    c, emis = c.sys, emis.sys
    operator_compose(convert(System, c), emis, Dict(
        c.NO2 => emis.NO2,
    ))
end

p = Photolysis()
@testset "Photolysis single" begin
    prob = ODEProblem(mtkcompile(p), [], (0.0, 1.0))
    sol = solve(prob, Tsit5())
    @test sol.retcode == ReturnCode.Success
end

c = Chemistry()
@testset "Chemistry single" begin
    prob = ODEProblem(mtkcompile(c), [], (0.0, 1.0))
    sol = solve(prob, Tsit5())
    @test sol.retcode == ReturnCode.Success
end

e = Emissions()
@testset "Emissions single" begin
    ee = mtkcompile(e)
    prob = ODEProblem(ee, [ee.NO2=>0], (0.0, 1.0))
    sol = solve(prob, Tsit5())
    @test sol.retcode == ReturnCode.Success
end

@testset "Coupled model" begin
    model = couple(c, p, e)
    sys = convert(System, model)

    prob = ODEProblem(sys, [], (0.0, 1.0))
    sol = solve(prob, Tsit5(), u0 = [1.0])
    @test sol.retcode == ReturnCode.Success
end
