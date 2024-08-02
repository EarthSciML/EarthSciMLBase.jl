using EarthSciMLBase
using Test
using ModelingToolkit, DomainSets, OrdinaryDiffEq

mutable struct ExampleOp <: Operator
    α::Num # Multiplier from ODESystem
    initialized::Bool
    finalized::Bool
end

function EarthSciMLBase.run!(op::ExampleOp, s::Simulator, t, step_length)
    f = s.obs_fs[s.obs_fs_idx[op.α]]
    for ix ∈ 1:size(s.u, 1)
        for (i, c1) ∈ enumerate(s.grid[1])
            for (j, c2) ∈ enumerate(s.grid[2])
                for (k, c3) ∈ enumerate(s.grid[3])
                    # Demonstrate coordinate transforms
                    t1 = s.tf_fs[1](t, c1, c2, c3)
                    t2 = s.tf_fs[2](t, c1, c2, c3)
                    t3 = s.tf_fs[3](t, c1, c2, c3)
                    # Demonstrate calculating observed value.
                    fv = f(t, c1, c2, c3)
                    # Set derivative value.
                    s.du[ix, i, j, k] = (t1 + t2 + t3) * fv
                end
            end
        end
    end
    @. s.u += s.du * step_length
    nothing
end

EarthSciMLBase.timestep(op::ExampleOp) = 1.0
EarthSciMLBase.initialize!(op::ExampleOp, s::Simulator) = op.initialized = true
EarthSciMLBase.finalize!(op::ExampleOp, s::Simulator) = op.finalized = true

t_min = 0.0
lon_min, lon_max = -π, π
lat_min, lat_max = -0.45π, 0.45π
t_max = 11.5

@parameters y lon = 0.0 lat = 0.0 lev = 1.0 t α = 10.0
@constants p = 1.0
@variables(
    u(t) = 1.0, v(t) = 1.0, x(t) = 1.0, y(t) = 1.0, windspeed(t) = 1.0
)
Dt = Differential(t)

indepdomain = t ∈ Interval(t_min, t_max)

partialdomains = [lon ∈ Interval(lon_min, lon_max),
    lat ∈ Interval(lat_min, lat_max),
    lev ∈ Interval(1, 3)]

domain = DomainInfo(
    partialderivatives_δxyδlonlat,
    constIC(16.0, indepdomain), constBC(16.0, partialdomains...))

eqs = [Dt(u) ~ -α * √abs(v) + lon,
    Dt(v) ~ -α * √abs(u) + lat + lev * 1e-14,
    windspeed ~ lat + lon + lev,
]
@named sys = ODESystem(eqs, t)

op = ExampleOp(sys.windspeed, false, false)

csys = couple(sys, op, domain)

sim = Simulator(csys, [0.1, 0.1, 1], Tsit5(); abstol=1e-12, reltol=1e-12)
st = SimulatorStrangThreads(Tsit5(), Euler(), 1.0)

@test 1 / (sim.tf_fs[1](0.0, 0.0, 0.0, 0.0) * 180 / π) ≈ 111319.44444444445
@test 1 / (sim.tf_fs[2](0.0, 0.0, 0.0, 0.0) * 180 / π) ≈ 111320.00000000001
@test sim.tf_fs[3](0.0, 0.0, 0.0, 0.0) == 1.0

@test sim.obs_fs[sim.obs_fs_idx[sys.windspeed]](0.0, 1.0, 3.0, 2.0) == 6.0
@test sim.obs_fs[sim.obs_fs_idx[op.α]](0.0, 1.0, 3.0, 2.0) == 6.0

EarthSciMLBase.run!(op, sim, 0.0, 1.0)

@test sum(abs.(sim.du)) ≈ 26094.203039436292

EarthSciMLBase.operator_step!(sim, 0.0, 1.0)

@test sum(abs.(sim.du)) ≈ 26094.203039436292

prob = ODEProblem(structural_simplify(sys), [], (0.0, 1.0), [
    lon => sim.grid[1][1], lat => sim.grid[2][1], lev => sim.grid[3][1]
])
sol1 = solve(prob, Tsit5(), abstol=1e-12, reltol=1e-12)
@test sol1.u[end] ≈ [-27.15156429366082, -26.264264199779465]

EarthSciMLBase.init_u!(sim)

IIchunks, integrators = let
    II = CartesianIndices(size(sim.u)[2:4])
    IIchunks = collect(Iterators.partition(II, length(II) ÷ st.threads))
    start, finish = EarthSciMLBase.time_range(sim.domaininfo)
    prob = ODEProblem(sim.sys_mtk, [], (start, finish), []; sim.kwargs...)
    integrators = [init(remake(prob, u0=similar(sim.u_init), p=deepcopy(sim.p)), st.stiffalg, save_on=false,
        save_start=false, save_end=false, initialize_save=false; sim.kwargs...)
                   for _ in 1:length(IIchunks)]
    (IIchunks, integrators)
end

EarthSciMLBase.threaded_ode_step!(sim, IIchunks, integrators, 0.0, 1.0)

@test sim.u[1,1,1,1] ≈ sol1.u[end][1]
@test sim.u[2,1,1,1] ≈ sol1.u[end][2]

@test sum(abs.(sim.u)) ≈ 212733.04492722102

run!(sim, st)

@test sum(abs.(sim.u)) ≈ 3.77224671877136e7

@test op.initialized == true
@test op.finalized == true

@testset "Float32" begin
    domain = DomainInfo(
        partialderivatives_δxyδlonlat,
        constIC(16.0, indepdomain), constBC(16.0, partialdomains...);
        dtype=Float32)

    csys = couple(sys, op, domain)

    sim = Simulator(csys, [0.1, 0.1, 1], Tsit5())
        
    run!(sim, st)

    @test sum(abs.(sim.u)) ≈ 3.77224671877136e7
end

@testset "No operator" begin
    domain = DomainInfo(
        partialderivatives_δxyδlonlat,
        constIC(16.0, indepdomain), constBC(16.0, partialdomains...);
        dtype=Float32)

    csys = couple(sys, domain)

    sim = Simulator(csys, [0.1, 0.1, 1], Tsit5())
        
    run!(sim, st)

    @test sum(abs.(sim.u)) ≈ 1.4343245f8
end