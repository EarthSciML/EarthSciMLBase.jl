using EarthSciMLBase
using Test
using ModelingToolkit, DomainSets, OrdinaryDiffEq
using SciMLOperators
using SciMLBase: DiscreteCallback, ReturnCode
using LinearSolve

struct ExampleOp <: Operator
end

function EarthSciMLBase.get_scimlop(op::ExampleOp, csys::CoupledSystem, mtk_sys, coord_args, domain::DomainInfo, u0, p)
    α, trans1, trans2, trans3 = EarthSciMLBase.get_needed_vars(op, csys, mtk_sys, domain)

    obs_f! = EarthSciMLBase.build_coord_observed_function(mtk_sys, coord_args,
        [α, trans1, trans2, trans3], true)

    II = CartesianIndices(tuple(size(domain)...))
    c1, c2, c3 = EarthSciMLBase.grid(domain)
    obscache = zeros(EarthSciMLBase.dtype(domain), 4)
    sz = length.(EarthSciMLBase.grid(domain))

    function run(du, u, p, t) # In-place
        u = reshape(u, :, sz...)
        du = reshape(du, :, sz...)
        II = CartesianIndices(sz)
        for ix ∈ 1:size(u, 1)
            for I in II
                # Demonstrate coordinate transforms and observed values
                obs_f!(obscache, view(u, :, I), p, t, c1[I[1]], c2[I[2]], c3[I[3]])
                t1, t2, t3, fv = obscache
                # Set derivative value.
                du[ix, I] = (t1 + t2 + t3) * fv
            end
        end
        nothing
    end
    function run(u, p, t) # Out-of-place
        u = reshape(u, :, sz...)
        II = CartesianIndices(size(u)[2:end])
        du = vcat([
            begin
                obs_f!(obscache, view(u, :, I), p, t, c1[I[1]], c2[I[2]], c3[I[3]])
                t1, t2, t3, fv = obscache
                (t1 + t2 + t3) * fv
            end for ix ∈ 1:size(u, 1), I in II
        ]...)
        reshape(du, :)
    end
    FunctionOperator(run, reshape(u0, :), p=p)
end

function EarthSciMLBase.get_needed_vars(::ExampleOp, csys, mtk_sys, domain::DomainInfo)
    return [mtk_sys.sys₊windspeed, mtk_sys.sys₊x, mtk_sys.sys₊y, mtk_sys.sys₊z]
end

t_min = 0.0
lon_min, lon_max = -π, π
lat_min, lat_max = -0.45π, 0.45π
t_max = 11.5

@parameters y lon = 0.0 lat = 0.0 lev = 1.0 t α = 10.0
@constants p = 1.0
@variables(
    u(t) = 1.0, v(t) = 1.0, x(t), y(t), z(t), windspeed(t)
)
Dt = Differential(t)

indepdomain = t ∈ Interval(t_min, t_max)

partialdomains = [lon ∈ Interval(lon_min, lon_max),
    lat ∈ Interval(lat_min, lat_max),
    lev ∈ Interval(1, 3)]

domain = DomainInfo(
    partialderivatives_δxyδlonlat,
    constIC(16.0, indepdomain), constBC(16.0, partialdomains...); grid_spacing=[0.1, 0.1, 1.0])

eqs = [Dt(u) ~ -α * √abs(v) + lon,
    Dt(v) ~ -α * √abs(u) + lat + lev * 1e-14,
    windspeed ~ lat + lon + lev,
    x ~ 1.0 / EarthSciMLBase.lon2meters(lat),
    y ~ 1.0 / EarthSciMLBase.lat2meters,
    z ~ 1.0 / lev,
]
sys = ODESystem(eqs, t, name=:sys)

op = ExampleOp()

csys = EarthSciMLBase.couple(sys, op, domain)

sys_mtk = convert(ODESystem, csys)

@test Symbol.(EarthSciMLBase.partialderivative_transform_vars(sys_mtk, domain)) == [
    Symbol("δsys₊lon_transform(t)"),
    Symbol("δsys₊lat_transform(t)"),
    Symbol("δsys₊lev_transform(t)")]

sys_coords, coord_args = EarthSciMLBase._prepare_coord_sys(sys_mtk, domain)
obs_f = EarthSciMLBase.build_coord_observed_function(sys_coords, coord_args,
    EarthSciMLBase.get_needed_vars(op, csys, sys_mtk, domain), false)

p = EarthSciMLBase.default_params(sys_coords)

obs_vals = obs_f([0.0, 0], p, 0.0, 0.0, 0.0, 1.0)
@test obs_vals[1] ≈ 1.0
@test 1 / (obs_vals[2] * 180 / π) ≈ 111319.44444444445
@test 1 / (obs_vals[3] * 180 / π) ≈ 111320.00000000001
@test obs_vals[4] == 1.0

obs_vals = obs_f([0.0, 0], p, 0.0, 1.0, 3.0, 2.0)
@test obs_vals[1] == 6.0

u = EarthSciMLBase.init_u(sys_coords, domain)
scimlop = EarthSciMLBase.nonstiff_ops(csys, sys_coords, coord_args, domain, reshape(u, :), p)
du = similar(u)
du .= 0
@views scimlop(reshape(du, :), reshape(u, :), p, 0.0)

@test sum(abs.(du)) ≈ 26094.193871228505

du2 = scimlop(reshape(u, :), p, 0.0)
@test du2 ≈ reshape(du, :)

setp! = EarthSciMLBase.coord_setter(sys_mtk, domain)

grid = EarthSciMLBase.grid(domain)
prob = ODEProblem(structural_simplify(sys), [], (0.0, 1.0), [
    lon => grid[1][1], lat => grid[2][1], lev => grid[3][1]
])
sol1 = solve(prob, Tsit5(); abstol=1e-12, reltol=1e-12)
@test sol1.retcode == ReturnCode.Success
@test sol1.u[end] ≈ [-27.15156429366082, -26.264264199779465]

st = SolverStrangThreads(Tsit5(), 1.0)
p = EarthSciMLBase.default_params(sys_mtk)

IIchunks, integrators = let
    II = CartesianIndices(size(u)[2:4])
    IIchunks = collect(Iterators.partition(II, length(II) ÷ st.threads))
    start, finish = get_tspan(domain)
    prob = ODEProblem(sys_mtk, [], (start, finish), [])
    integrators = [init(remake(prob, u0=zeros(length(unknowns(sys_mtk))), p=deepcopy(p)), st.stiffalg, save_on=false,
        save_start=false, save_end=false, initialize_save=false; abstol=1e-12, reltol=1e-12)
                   for _ in 1:length(IIchunks)]
    (IIchunks, integrators)
end

EarthSciMLBase.threaded_ode_step!(setp!, u, IIchunks, integrators, 0.0, 1.0)

@test u[1, 1, 1, 1] ≈ sol1.u[end][1]
@test u[2, 1, 1, 1] ≈ sol1.u[end][2]

@test sum(abs.(u)) ≈ 212733.04492722102

@testset "mtk_func" begin
    ucopy = copy(u)
    fiip, sys_coords, coord_args = EarthSciMLBase.mtk_grid_func(sys_mtk, domain, ucopy, true; sparse=true, tgrad=true)
    p = EarthSciMLBase.default_params(sys_coords)
    uu = EarthSciMLBase.init_u(sys_coords, domain)
    prob = ODEProblem(fiip, uu[:], (0.0, 1.0), p)
    sol = solve(prob, Tsit5())
    uu = reshape(sol.u[end], size(ucopy)...)
    @test uu[:] ≈ u[:] rtol = 0.01

    @testset "In-place vs. out of place" begin
        du1 = reshape(similar(ucopy), :)
        prob.f(du1, reshape(ucopy, :), p, 0.0)
        foop, sys_coords, coord_args = EarthSciMLBase.mtk_grid_func(sys_mtk, domain, ucopy, false; sparse=true, tgrad=true)
        p = EarthSciMLBase.default_params(sys_coords)
        du2 = foop(reshape(ucopy, :), p, 0.0)
        @test du1 ≈ du2
    end

    f, _, _ = EarthSciMLBase.mtk_grid_func(sys_mtk, domain, ucopy, true; sparse=false, tgrad=true)
    du = EarthSciMLBase.BlockDiagonal([zeros(size(ucopy, 1), size(ucopy, 1)) for i in 1:reduce(*, size(ucopy)[2:end])])
    f.jac(du, ucopy[:], p, 0.0)
    @test sum(sum.(du.blocks)) ≈ 12617.772209024473

    f, _, _ = EarthSciMLBase.mtk_grid_func(sys_mtk, domain, ucopy, false; sparse=false, tgrad=true)
    du2 = f.jac(reshape(ucopy, :), p, 0.0)
    @test all(du.blocks .≈ du2.blocks)


    @testset "tgrad" begin
        f, _, _ = EarthSciMLBase.mtk_grid_func(sys_mtk, domain, ucopy, true; sparse=false, tgrad=true)
        for i in 1:length(du.blocks)
            du.blocks[i] .= 0
        end
        f.tgrad(du, ucopy[:], p, 0.0)
        @test sum(sum.(du.blocks)) ≈ 0.0
    end
end

prob = ODEProblem(csys, st)
sol = solve(prob, Euler(); dt=1.0, abstol=1e-12, reltol=1e-12)

@test sum(abs.(sol.u[end])) ≈ 3.77224671877136e7 rtol = 1e-3

@testset "Float32" begin
    domain = DomainInfo(
        partialderivatives_δxyδlonlat,
        constIC(16.0, indepdomain), constBC(16.0, partialdomains...);
        dtype=Float32, grid_spacing=[0.1, 0.1, 1])

    csys = couple(sys, op, domain)

    prob = ODEProblem(csys, st)
    sol = solve(prob, Euler(); dt=1.0)

    @test sum(abs.(sol.u[end])) ≈ 3.77224671877136e7
end

@testset "No operator" begin
    domain = DomainInfo(
        partialderivatives_δxyδlonlat,
        constIC(16.0, indepdomain), constBC(16.0, partialdomains...);
        dtype=Float32, grid_spacing=[0.1, 0.1, 1])

    csys = couple(sys, domain)

    prob = ODEProblem(csys, st)
    sol = solve(prob, Euler(); dt=1.0, abstol=1e-6, reltol=1e-6)

    @test sum(abs.(sol.u[end])) ≈ 3.8660308f7
end

@testset "SimulatorStrategies" begin
    @testset "Strang Threads" begin
        st = SolverStrangThreads(Tsit5(), 1.0)
        prob = ODEProblem(csys, st)
        sol = solve(prob, Euler(); dt=1.0, abstol=1e-12, reltol=1e-12)
        @test sum(abs.(sol.u[end])) ≈ 3.77224671877136e7 rtol = 1e-3
    end

    @testset "Strang Serial" begin
        st = SolverStrangSerial(Tsit5(), 1.0)
        prob = ODEProblem(csys, st)
        sol = solve(prob, Euler(); dt=1.0, abstol=1e-12, reltol=1e-12)
        @test sum(abs.(sol.u[end])) ≈ 3.77224671877136e7 rtol = 1e-3
    end

    @testset "IMEX" begin
        st = SolverIMEX()
        prob = ODEProblem(csys, st)
        sol = solve(prob, Tsit5())
        @test sum(abs.(sol.u[end])) ≈ 3.3333500929324217e7 rtol = 1e-3 # No Splitting error in this one.

        for iip in [true, false]
            for sparse in [true, false]
                @testset "iip=$iip; sparse=$sparse" begin
                    st = SolverIMEX(stiff_sparse=sparse)
                    prob = ODEProblem{iip}(csys, st)
                    sol = solve(prob, KenCarp47(linsolve=LUFactorization()), abstol=1e-4, reltol=1e-4)
                    @test sum(abs.(sol.u[end])) ≈ 3.3333500929324217e7 rtol = 1e-3
                end
            end
        end
    end
end

mutable struct cbt
    runcount::Int
end
function EarthSciMLBase.init_callback(c::cbt, sys, sys_mtk, dom)
    DiscreteCallback((u, t, integrator) -> true,
        (_) -> c.runcount += 1,
    )
end

@testset "callback" begin
    runcount = 0
    af(_) = runcount += 1
    cb = DiscreteCallback(
        (u, t, integrator) -> true,
        af,
    )
    @testset "strang" begin
        cc = cbt(0)
        csys2 = couple(csys, cb, cc)
        prob = ODEProblem(csys2, st)
        solve(prob, Euler(); dt=1.0)
        @test runcount > 0
        @test cc.runcount > 0
    end
    runcount = 0
    @testset "imex" begin
        cc = cbt(0)
        csys2 = couple(csys, cb, cc)
        prob = ODEProblem(csys2, SolverIMEX())
        solve(prob, Tsit5(); dt=1.0)
        @test runcount > 0
        @test cc.runcount > 0
    end
end
