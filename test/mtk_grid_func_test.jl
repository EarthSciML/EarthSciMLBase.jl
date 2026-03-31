using EarthSciMLBase
using ModelingToolkit
using DomainSets
using Test
using SymbolicIndexingInterface
using OrdinaryDiffEqTsit5
import Reactant
using JLArrays
using Symbolics: value as sym_value
t = ModelingToolkit.t_nounits
D = ModelingToolkit.D_nounits

@parameters y lon=0.0 lat=0.0 lev=1.0 α=10.0 β=1.0
@constants p = 1.0
@variables(u(t)=1.0, v(t)=1.0, x(t)=1.0, y(t)=1.0, windspeed(t))

t_min = 0.0
lon_min, lon_max = -π, π
lat_min, lat_max = -0.45π, 0.45π
t_max = 11.5

indepdomain = t ∈ Interval(t_min, t_max)

partialdomains = [lon ∈ Interval(lon_min, lon_max),
    lat ∈ Interval(lat_min, lat_max),
    lev ∈ Interval(1, 3)]

domain = DomainInfo(
    partialderivatives_δxyδlonlat,
    constIC(16.0, indepdomain), constBC(16.0, partialdomains...); grid_spacing = [
        1.0, 1.0, 1.0])

eqs = [D(u) ~ -α * √abs(v) + u + lon + β,
    D(v) ~ -α * √abs(u) + lat + lev * 1.0f-14,
    windspeed ~ lon + 2lat + 3lev
]
sys = System(eqs, t, name = :sys)

sys_simplified = mtkcompile(sys)
prob = ODEProblem(sys_simplified, [], (0.0, 1); jac = true, tgrad = true)
solve(prob, Tsit5())

sys_coord, coord_args = EarthSciMLBase._prepare_coord_sys(sys_simplified, domain)
@test occursin(
    "EarthSciMLBase._CoordTmpF(lon, 1)(t)", string(ModelingToolkit.observed(sys_coord)))
f = EarthSciMLBase.build_coord_ode_function(
    sys_coord, coord_args; eval_module = ModelingToolkit)
p = MTKParameters(sys_coord, ModelingToolkit.initial_conditions(sys_coord))
duop = f(prob.u0, p, 0.0, 2.0, 1.0, 1e13)
u_perm = [findfirst(isequal(u), unknowns(sys_simplified)) for u in unknowns(sys_coord)]
new_p = remake_buffer(sys_simplified, prob.p, [lon, lat, lev], [2.0, 1.0, 1e13])
@test duop ≈ prob.f(prob.u0, new_p, 0.0)[u_perm]
@test prob.f(prob.u0, prob.p, 0.0)[u_perm] ≈ f(prob.u0, p, 0.0, 0, 0, 0)

duip = similar(prob.u0)
f(duip, prob.u0, p, 0.0, 2.0, 1.0, 1e13)
@test duip ≈ duop

jac = EarthSciMLBase.build_coord_jac_function(sys_coord, coord_args; sparse = true)
Jop = jac(prob.u0, p, 0.0, 0.0, 0.0, 0.0)
@test Jop ≈ prob.f.jac(prob.u0, prob.p, 0.0)[u_perm, u_perm]

jac = EarthSciMLBase.build_coord_jac_function(sys_coord, coord_args; sparse = false)
Jop = jac(prob.u0, p, 0.0, 0.0, 0.0, 0.0)
@test Jop ≈ prob.f.jac(prob.u0, prob.p, 0.0)[u_perm, u_perm]

Jip = zeros(length(prob.u0), length(prob.u0))
jac(Jip, prob.u0, p, 0.0, 0.0, 0.0, 0.0)
@test Jip ≈ Jop

tgrad = EarthSciMLBase.build_coord_tgrad_function(sys_coord, coord_args)
# Convert symbolic Num values to concrete numbers for comparison (MTK v11 tgrad may return Num types)
@test isapprox(Float64.(sym_value.(tgrad(prob.u0, p, 0.0, 0.0, 0.0, 0.0))),
    Float64.(sym_value.(prob.f.tgrad(prob.u0, prob.p, 0.0))); atol = 1e-10)

u0 = EarthSciMLBase.init_u(sys_coord, domain)

@testset "grid solve" begin
    f, _, _ = EarthSciMLBase.mtk_grid_func(sys, domain, u0)

    @testset "in place" begin
        prob = ODEProblem(f, u0, (0.0, 1.0), p)
        sol1 = solve(prob, Tsit5())
        @test sum(sol1.u[end]) ≈ -3029.442918648946
    end

    @testset "out of place" begin
        prob = ODEProblem{false}(f, u0, (0.0, 1.0), p)
        sol2 = solve(prob, Tsit5())
        @test sum(sol2.u[end]) ≈ -3029.442918648946
    end
end

@testset "observed" begin
    obs_f = EarthSciMLBase.build_coord_observed_function(
        sys_coord, coord_args, [sys_coord.windspeed])
    @test [14] ≈ obs_f(u0, p, 0.0, 1, 2, 3)
end

if Sys.isapple()
    @testset "GPU jacobian" begin
        using Metal
        domain = DomainInfo(
            constIC(16.0, indepdomain), constBC(16.0, partialdomains...); grid_spacing = [
                1.0, 1.0, 1.0],
            u_proto = MtlArray(zeros(Float32, 1, 1, 1, 1)))
        csys = couple(sys, domain)
        prob = ODEProblem(csys, SolverIMEX(MapKernel(), stiff_sparse = false))

        Jo = similar(prob.f.jac_prototype)
        prob.f.jac(Jo, prob.u0, prob.p, prob.tspan[1])
        @test Array(EarthSciMLBase.block(Jo, 1)) ≈ Jop
    end
end

@testset "Reactant simple" begin
    domain = DomainInfo(
        partialderivatives_δxyδlonlat,
        constIC(16.0, indepdomain), constBC(16.0, partialdomains...);
        grid_spacing = [1.0, 1.0, 1.0],
        u_proto = Reactant.to_rarray(zeros(Float32, 0)))
    u0 = EarthSciMLBase.init_u(sys_coord, domain)

    f, _, _ = EarthSciMLBase.mtk_grid_func(sys, domain, u0, MapReactant())
    du = similar(u0)
    f(du, u0, p, 0.0f0)
    @test du[1:2] ≈ [-11.413716694115397, -11.141592653589793]
end
