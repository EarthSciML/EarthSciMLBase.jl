using EarthSciMLBase
using ModelingToolkit
using DomainSets
using Test
using SymbolicIndexingInterface
using OrdinaryDiffEqTsit5
using JLArrays
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
    D(v) ~ -α * √abs(u) + lat + lev * 1e-14,
    windspeed ~ lon + 2lat + 3lev
]
sys = System(eqs, t, name = :sys)

sys_simplified = structural_simplify(sys)
prob = ODEProblem(sys_simplified, [], (0.0, 1); jac = true, tgrad = true)
solve(prob, Tsit5())

sys_coord, coord_args = EarthSciMLBase._prepare_coord_sys(sys_simplified, domain)
@test occursin("EarthSciMLBase._coord1_tmp(t)", string(ModelingToolkit.observed(sys_coord)))
f = EarthSciMLBase.build_coord_ode_function(
    sys_coord, coord_args; eval_module = ModelingToolkit)
p = MTKParameters(sys_coord, defaults(sys_coord))
duop = f(prob.u0, p, 0.0, 2.0, 1.0, 1e13)
u_perm = [findfirst(isequal(u), unknowns(sys_simplified)) for u in unknowns(sys_coord)]
new_p = remake_buffer(sys_simplified, prob.p, Dict(lon => 2.0, lat => 1.0, lev => 1e13))
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
@test tgrad(prob.u0, p, 0.0, 0.0, 0.0, 0.0) ≈ prob.f.tgrad(prob.u0, prob.p, 0.0)

u0 = EarthSciMLBase.init_u(sys_coord, domain)

@testset "grid solve" begin
    f, _, _ = EarthSciMLBase.mtk_grid_func(sys, domain, u0)

    @testset "in place" begin
        prob = ODEProblem(f, reshape(u0, :), (0.0, 1.0), p)
        sol1 = solve(prob, Tsit5())
        @test sum(sol1[end]) ≈ -3029.442918648946
    end

    @testset "out of place" begin
        prob = ODEProblem{false}(f, reshape(u0, :), (0.0, 1.0), p)
        sol2 = solve(prob, Tsit5())
        @test sum(sol2[end]) ≈ -3029.442918648946
    end

    @testset "In place GPU" begin
        prob_gpu = ODEProblem(f, jl(reshape(u0, :)), (0.0, 1.0), p)
        @test_broken sol = solve(prob_gpu, Tsit5())
    end

    @testset "Out of place GPU" begin
        prob_gpu = ODEProblem{false}(f, jl(reshape(u0, :)), (0.0, 1.0), p)
        @test_broken sol = solve(prob_gpu, Tsit5())
    end
end

@testset "observed" begin
    obs_f = EarthSciMLBase.build_coord_observed_function(
        sys_coord, coord_args, [sys_coord.windspeed])
    @test [14] ≈ obs_f(reshape(u0, :), p, 0.0, 1, 2, 3)
end
