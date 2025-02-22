using EarthSciMLBase
using ModelingToolkit
using DomainSets
using Test
using SymbolicIndexingInterface
using OrdinaryDiffEq

@parameters y lon = 0.0 lat = 0.0 lev = 1.0 t α = 10.0 β = 1.0
@constants p = 1.0
@variables(
    u(t) = 1.0, v(t) = 1.0, x(t) = 1.0, y(t) = 1.0, windspeed(t)
)
Dt = Differential(t)

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
    constIC(16.0, indepdomain), constBC(16.0, partialdomains...); grid_spacing=[1.0, 1.0, 1.0])

eqs = [Dt(u) ~ -α * √abs(v) + u + lon + β,
    Dt(v) ~ -α * √abs(u) + lat + lev * 1e-14,
    windspeed ~ lon + 2lat + 3lev,
]
sys = ODESystem(eqs, t, name=:sys)

sys_simplified = structural_simplify(sys)
prob = ODEProblem(sys_simplified, jac=true, tgrad=true)

sys_coord, coord_args = EarthSciMLBase._prepare_coord_sys(sys_simplified, domain)
@test occursin("lat_arg", string(ModelingToolkit.observed(sys_coord)))
fop = EarthSciMLBase._build_mtk_coord_arg_function(sys_coord, coord_args, ModelingToolkit.generate_function, false)
p = MTKParameters(sys_coord, defaults(sys_coord))
duop = fop(prob.u0, p, 0.0, 2.0, 1.0, 1e13)
@test duop ≈ prob.f(prob.u0, remake_buffer(sys_simplified, prob.p, Dict(lon => 2.0, lat => 1.0, lev => 1e13)), 0.0)
@test prob.f(prob.u0, prob.p, 0.0) ≈ fop(prob.u0, p, 0.0, 0, 0, 0)

fip = EarthSciMLBase._build_mtk_coord_arg_function(sys_coord, coord_args, ModelingToolkit.generate_function, true)
duip = similar(prob.u0)
fip(duip, prob.u0, p, 0.0, 2.0, 1.0, 1e13)
@test duip ≈ duop

fop = EarthSciMLBase._build_mtk_coord_arg_function(sys_coord, coord_args, ModelingToolkit.generate_function, false)
duop2 = fop(prob.u0, p, 0.0, 2.0, 1.0, 1e13)
@test duop2 ≈ duop

jacop = EarthSciMLBase._build_mtk_coord_arg_function(sys_coord, coord_args, ModelingToolkit.generate_jacobian, false)
Jop = jacop(prob.u0, p, 0.0, 0.0, 0.0, 0.0)
@test Jop ≈ prob.f.jac(prob.u0, prob.p, 0.0)

jacip = EarthSciMLBase._build_mtk_coord_arg_function(sys_coord, coord_args, ModelingToolkit.generate_jacobian, true)
Jip = zeros(length(prob.u0), length(prob.u0))
jacip(Jip, prob.u0, p, 0.0, 0.0, 0.0, 0.0)
@test Jip ≈ Jop

top = EarthSciMLBase._build_mtk_coord_arg_function(sys_coord, coord_args, ModelingToolkit.generate_tgrad, false)
@test top(prob.u0, p, 0.0, 0.0, 0.0, 0.0) ≈ prob.f.tgrad(prob.u0, prob.p, 0.0)

u0 = EarthSciMLBase.init_u(sys_coord, domain)

@testset "in place" begin
    f, _ = EarthSciMLBase.mtk_grid_func(sys, domain, u0, true)
    prob = ODEProblem(f, reshape(u0, :), (0.0, 1.0), p)
    sol1 = solve(prob)
    @test sum(sol1[end]) ≈ -3029.442918648946
end

@testset "out of place" begin
    f, _ = EarthSciMLBase.mtk_grid_func(sys, domain, u0, false)
    prob = ODEProblem{false}(f, reshape(u0, :), (0.0, 1.0), p)
    sol2 = solve(prob)
    @test sum(sol2[end]) ≈ -3029.442918648946
end

@testset "observed" begin
    obs_f = EarthSciMLBase.build_coord_observed_function(sys_coord, coord_args, [sys_coord.windspeed], false)
    @test [14] ≈ obs_f(reshape(u0, :), p, 0.0, 1, 2, 3)
end
