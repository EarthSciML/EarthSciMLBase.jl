using EarthSciMLBase
using Test
using ModelingToolkit, DomainSets
using OrdinaryDiffEqTsit5, OrdinaryDiffEqSDIRK, OrdinaryDiffEqLowOrderRK
using DynamicQuantities
using SciMLBase: DiscreteCallback, ContinuousCallback, CallbackSet, ReturnCode,
                 u_modified!
import DiffEqCallbacks
using LinearSolve
import SciMLStructures
import SymbolicIndexingInterface
t = ModelingToolkit.t_nounits
D = ModelingToolkit.D_nounits

struct ExampleOp <: Operator
end

function EarthSciMLBase.get_odefunction(
        op::ExampleOp, csys::CoupledSystem, mtk_sys, coord_args,
        domain::DomainInfo{ET, AT}, u0, p, alg::MapAlgorithm) where {ET, AT}
    α, trans1, trans2, trans3 = EarthSciMLBase.get_needed_vars(op, csys, mtk_sys, domain)

    obs_f = EarthSciMLBase.build_coord_observed_function(mtk_sys, coord_args,
        [α, trans1, trans2, trans3])

    c1, c2, c3 = EarthSciMLBase.concrete_grid(domain)
    obscache = similar(domain.u_proto, 4)

    nrows = length(unknowns(mtk_sys))
    sz = tuple(size(domain)...)
    II = CartesianIndices(sz)

    function run(du, u, p, t) # In-place
        u = reshape(u, nrows, sz...)
        du = reshape(du, nrows, sz...)
        function f(j, du, u, p, t, c1, c2, c3)
            # Demonstrate coordinate transforms and observed values
            obs_f(obscache, view(u, :, II[j]), p, t, c1[j], c2[j], c3[j])
            t1, t2, t3, fv = obscache
            # Set derivative value.
            view(du, :, II[j]) .= (t1 + t2 + t3) * fv
            return nothing
        end
        EarthSciMLBase.map_closure_to_range(f, 1:((*)(sz...)), alg, du, u, p, t, c1, c2, c3)
        return reshape(du, :)
    end
    function run(u, p, t) # Out-of-place
        u = reshape(u, :, sz...)
        function f(j, u, p, t, c1, c2, c3)
            t1, t2, t3, fv = obs_f(view(u, :, II[j]), p, t, c1[j], c2[j], c3[j])
            [(t1 + t2 + t3) * fv for _ in 1:size(u, 1)]
        end
        du = EarthSciMLBase.map_closure_to_range(f, 1:((*)(sz...)),
            alg, u, p, t, c1, c2, c3)
        reshape(hcat(du...), :)
    end
    return run
end

function EarthSciMLBase.get_needed_vars(::ExampleOp, csys, mtk_sys, domain::DomainInfo)
    return [mtk_sys.sys₊windspeed, mtk_sys.sys₊x, mtk_sys.sys₊y, mtk_sys.sys₊z]
end

# An operator that ASSIGNS a tendency into only the first state row and leaves the
# other rows untouched — the write pattern of equilibrium partitioning operators
# (e.g. GasChem's IsorropiaOp).
struct PartialWriterOp <: Operator
end

function EarthSciMLBase.get_odefunction(
        op::PartialWriterOp, csys::CoupledSystem, mtk_sys, coord_args,
        domain::DomainInfo, u0, p, alg::MapAlgorithm)
    nrows = length(unknowns(mtk_sys))
    sz = tuple(size(domain)...)
    function run(du, u, p, t) # In-place
        du = reshape(du, nrows, sz...)
        u = reshape(u, nrows, sz...)
        @views du[1, :, :, :] .= 0.1 .* u[1, :, :, :]
        return reshape(du, :)
    end
    function run(u, p, t) # Out-of-place
        u = reshape(u, nrows, sz...)
        du = zeros(size(u))
        @views du[1, :, :, :] .= 0.1 .* u[1, :, :, :]
        return reshape(du, :)
    end
    return run
end

function EarthSciMLBase.get_needed_vars(::PartialWriterOp, csys, mtk_sys, domain::DomainInfo)
    return Num[]
end

t_min = 0.0
lon_min, lon_max = -π, π
lat_min, lat_max = -0.45π, 0.45π
t_max = 11.5

@parameters y lon=0.0 lat=0.0 lev=1.0 α=10.0
@constants p = 1.0
@variables(u(t)=1.0, v(t)=1.0, x(t), [unit=u"1/m"], y(t), [unit=u"1/m"], z(t),
    windspeed(t))

indepdomain = t ∈ Interval(t_min, t_max)

partialdomains = [lon ∈ Interval(lon_min, lon_max),
    lat ∈ Interval(lat_min, lat_max),
    lev ∈ Interval(1, 3)]

domain = DomainInfo(
    partialderivatives_δxyδlonlat,
    constIC(16.0, indepdomain), constBC(16.0, partialdomains...); grid_spacing = [
        0.1, 0.1, 1.0])

eqs = [D(u) ~ -α * √abs(v) + lon,
    D(v) ~ -α * √abs(u) + lat + lev * 1.0f-14,
    windspeed ~ lat + lon + lev,
    x ~ 1 / EarthSciMLBase.lon2meters(lat),
    y ~ 1 / EarthSciMLBase.lat2meters,
    z ~ 1 / lev
]
sys = System(eqs, t, name = :sys)

op = ExampleOp()

csys = EarthSciMLBase.couple(sys, op, domain)

sys_mtk = convert(System, csys)

@test Symbol.(EarthSciMLBase.partialderivative_transform_vars(sys_mtk, domain)) == [
    Symbol("δsys₊lon_transform(t)"),
    Symbol("δsys₊lat_transform(t)"),
    Symbol("δsys₊lev_transform(t)")]

sys_coords, coord_args = EarthSciMLBase._prepare_coord_sys(sys_mtk, domain)
obs_f = EarthSciMLBase.build_coord_observed_function(sys_coords, coord_args,
    EarthSciMLBase.get_needed_vars(op, csys, sys_mtk, domain))

p = EarthSciMLBase.default_params(sys_coords)

obs_vals = obs_f([0.0, 0], p, 0.0, 0.0, 0.0, 1.0)
@test obs_vals[1] ≈ 1.0
@test 1 / (obs_vals[2] * 180 / π) ≈ 111319.44444444445
@test 1 / (obs_vals[3] * 180 / π) ≈ 111320.00000000001
@test obs_vals[4] == 1.0

obs_vals = obs_f([0.0, 0], p, 0.0, 1.0, 3.0, 2.0)
@test obs_vals[1] == 6.0

u = reshape(EarthSciMLBase.init_u(sys_coords, domain), :, size(domain)...)
scimlop = EarthSciMLBase.nonstiff_ops(csys, sys_coords, coord_args, domain, reshape(u, :),
    p, MapBroadcast())
du = similar(u)
du .= 0
scimlop(reshape(du, :), reshape(u, :), p, 0.0)
@test sum(abs.(du)) ≈ 14542.756845747295
du = scimlop(reshape(u, :), p, 0.0)
@test sum(abs.(du)) ≈ 14542.756845747295

du2 = scimlop(reshape(u, :), p, 0.0)
@test du2 ≈ reshape(du, :)

@testset "multi-op combiner sums operator tendencies" begin
    csys2 = EarthSciMLBase.couple(sys, ExampleOp(), PartialWriterOp(), domain)
    sys_mtk2 = convert(System, csys2)
    sys_coords2, coord_args2 = EarthSciMLBase._prepare_coord_sys(sys_mtk2, domain)
    p2 = EarthSciMLBase.default_params(sys_coords2)
    u2 = EarthSciMLBase.init_u(sys_coords2, domain)

    f_full = EarthSciMLBase.get_odefunction(ExampleOp(), csys2, sys_coords2, coord_args2,
        domain, reshape(u2, :), p2, MapBroadcast())
    f_part = EarthSciMLBase.get_odefunction(PartialWriterOp(), csys2, sys_coords2,
        coord_args2, domain, reshape(u2, :), p2, MapBroadcast())
    du_full = zeros(length(u2))
    f_full(du_full, reshape(u2, :), p2, 0.0)
    du_part = zeros(length(u2))
    f_part(du_part, reshape(u2, :), p2, 0.0)
    @test sum(abs.(du_full)) > 0
    @test sum(abs.(du_part)) > 0

    combined = EarthSciMLBase.nonstiff_ops(csys2, sys_coords2, coord_args2, domain,
        reshape(u2, :), p2, MapBroadcast())
    # Stale incoming du contents must not leak into the result.
    du_c = fill(123.0, length(u2))
    combined(du_c, reshape(u2, :), p2, 0.0)
    # True-sum contract: rows written by both operators carry both contributions
    # (a sequential same-buffer combiner would let PartialWriterOp overwrite
    # ExampleOp's tendency in row 1), and rows only ExampleOp writes are kept.
    @test du_c ≈ du_full .+ du_part
    # Both contributions must be visible individually (kills the overwrite bug
    # under either operator ordering).
    @test !(du_c ≈ du_full)
    @test !(du_c ≈ du_part)
    # In-place and out-of-place forms agree.
    @test combined(reshape(u2, :), p2, 0.0) ≈ du_c
end

grid = EarthSciMLBase.grid(domain)
sys1 = mtkcompile(sys)
op_pt = [lon => grid[1][1], lat => grid[2][1], lev => grid[3][1]]
prob = ODEProblem(sys1, op_pt, (0.0, 1.0))
sol1 = solve(prob, Tsit5(); abstol = 1e-12, reltol = 1e-12)
@test sol1.retcode == ReturnCode.Success
@test sol1.u[end] ≈ [-27.15156429366082, -26.264264199779465] ||
      sol1.u[end] ≈ [-26.264264199779465, -27.15156429366082]

st = SolverStrangThreads(Tsit5(), 1.0; abstol = 1e-12, reltol = 1e-12)
p = EarthSciMLBase.default_params(sys_mtk)

f_ode, u0_single,
p = EarthSciMLBase._strang_ode_func(sys_coords, coord_args,
    get_tspan(domain), grid; sparse = false)
IIchunks,
integrators = EarthSciMLBase._strang_integrators(st, domain, f_ode, u0_single,
    get_tspan(domain)[1], p, nothing)

EarthSciMLBase.threaded_ode_step!(u, IIchunks, integrators, 0.0, 1.0)

let
    varsyms = EarthSciMLBase.var2symbol.(unknowns(sys_coords))
    @test u[1, 1, 1, 1] ≈ sol1[varsyms[1]][end]
    @test u[2, 1, 1, 1] ≈ sol1[varsyms[2]][end]
end

@test sum(abs.(u)) ≈ 212733.04492722102

@testset "mtk_func" begin
    ucopy = copy(u)
    f, sys_coords,
    coord_args = EarthSciMLBase.mtk_grid_func(sys_mtk, domain, ucopy;
        sparse = false, tgrad = true)
    fthreads, = EarthSciMLBase.mtk_grid_func(sys_mtk, domain, ucopy,
        MapThreads(); sparse = false, tgrad = false)
    p = EarthSciMLBase.default_params(sys_coords)
    uu = EarthSciMLBase.init_u(sys_coords, domain)
    prob = ODEProblem(f, uu, (0.0, 1.0), p)
    sol = solve(prob, Tsit5())
    varsyms = [Symbol("sys₊", v) for v in EarthSciMLBase.var2symbol.(unknowns(sys1))]
    u_perm = [findfirst(isequal(u), varsyms)
              for u in EarthSciMLBase.var2symbol.(unknowns(sys_mtk))]
    uu = reshape(sol.u[end], size(ucopy)...)[u_perm, :, :, :]
    @test uu[:]≈u[:] rtol=0.01

    @testset "In-place vs. out of place" begin
        du1 = reshape(similar(ucopy), :)
        prob.f(du1, reshape(ucopy, :), p, 0.0)
        du2 = f(reshape(ucopy, :), p, 0.0)
        du3 = fthreads(reshape(ucopy, :), p, 0.0)
        @test du1 ≈ du2 ≈ du3
    end

    @testset "jac dense" begin
        du = similar(f.jac_prototype)
        f.jac(du, ucopy[:], p, 0.0)
        @test sum(du.data) ≈ 12617.772209024473

        du2 = f.jac(reshape(ucopy, :), p, 0.0)
        @test all(du.data .≈ du2.data)
    end

    @testset "jac MapThreads" begin
        du = similar(fthreads.jac_prototype)
        fthreads.jac(du, ucopy[:], p, 0.0)
        @test sum(du.data) ≈ 12617.772209024473

        du2 = fthreads.jac(reshape(ucopy, :), p, 0.0)
        @test all(du.data .≈ du2.data)
    end

    @testset "tgrad" begin
        du = similar(f.jac_prototype)
        for i in 1:size(du.data, 3)
            EarthSciMLBase.block(du, i) .= 0
        end
        f.tgrad(du, ucopy[:], p, 0.0)
        @test sum(du.data) ≈ 0.0
    end
end

prob = ODEProblem(csys, st)
sol = solve(prob, Euler(); dt = 1.0, abstol = 1e-12, reltol = 1e-12)

@test sum(abs.(sol.u[end]))≈3.820642384890682e7 rtol=1e-3

st = SolverStrangThreads(Tsit5(), 1.0)

@testset "Float32" begin
    domain = DomainInfo(
        constIC(16.0, indepdomain), constBC(16.0, partialdomains...);
        u_proto = zeros(Float32, 1, 1, 1, 1), grid_spacing = [0.1, 0.1, 1])

    csys = couple(sys, op, domain)

    prob = ODEProblem(csys, st)
    @test_broken eltype(prob.f(prob.u0[:], prob.p, prob.tspan[1])) == Float32 # MTK v11 code generation does not preserve Float32
    sol = solve(prob, Euler(); dt = 1.0)

    # Strang-with-Tsit5 sums fall slightly off the pre-fix values because
    # `single_ode_step!` sets `integrator.p.ii` for each cell before `reinit!`, so
    # the initial-dt RHS evaluations and the FSAL first stage (`fsalfirst`) use the
    # correct cell's coordinates instead of the previous cell's — see
    # `single_ode_step!` and EarthSciData issue #207. The sum no longer depends on
    # the number of threads.
    @test sum(abs.(sol.u[end])) ≈ 3.8206604e7 rtol = 1e-6

    @testset "Split problem" begin
        prob = ODEProblem(csys, SolverIMEX())
        @test_broken eltype(prob.f(prob.u0[:], prob.p, prob.tspan[1])) == Float32 # MTK v11 code generation does not preserve Float32
        @test eltype(prob.f.f1(prob.u0[:], prob.p, prob.tspan[1])) == Float32
        @test_broken eltype(prob.f.f2(prob.u0[:], prob.p, prob.tspan[1])) == Float32 # MTK v11 code generation does not preserve Float32
    end
end

@testset "No operator" begin
    domain = DomainInfo(
        partialderivatives_δxyδlonlat,
        constIC(16.0, indepdomain), constBC(16.0, partialdomains...);
        u_proto = zeros(Float32, 1, 1, 1, 1), grid_spacing = [0.1, 0.1, 1])

    csys = couple(sys, domain)

    prob = ODEProblem(csys, st)
    sol = solve(prob, Euler(); dt = 1.0, abstol = 1e-6, reltol = 1e-6)

    @test sum(abs.(sol.u[end])) ≈ 3.8660308f7
end

@testset "MapKernel" begin
    ucopy = Float32.(u)
    domain = DomainInfo(
        constIC(16.0, indepdomain), constBC(16.0, partialdomains...);
        u_proto = ucopy, grid_spacing = [0.1, 0.1, 1])
    csys = couple(sys, domain)
    prob = ODEProblem(csys, SolverIMEX(MapKernel()))
    du = similar(prob.u0)
    prob.f(du, prob.u0, prob.p, prob.tspan[1])
    @test du[1] ≈ -13.141593f0
end

if Sys.isapple() # TODO: Why aren't the results of these tests deterministic?
    @testset "Metal GPU" begin
        using Metal
        ucopy = MtlArray(Float32.(u))
        domain = DomainInfo(
            constIC(16.0, indepdomain), constBC(16.0, partialdomains...);
            u_proto = ucopy, grid_spacing = [0.1, 0.1, 1])

        csys = couple(sys, op, domain)

        prob = ODEProblem(
            csys, SolverIMEX(MapKernel(), BlockDiagonalJacobian(),
                stiff_sparse = false))

        du = similar(prob.u0)
        prob.f.f1(du, prob.u0, prob.p, prob.tspan[1])
        @test Array(du)[1] ≈ -13.141593f0
        du .= 0.0f0
        prob.f.f2(du, prob.u0, prob.p, prob.tspan[1])
        @test Array(du)[1] ≈ -3.5553088f0
        du .= 0.0f0
        prob.f(du, prob.u0, prob.p, prob.tspan[1])
        @test Array(du)[1] ≈ -3.5553088f0 + -13.141593f0

        @testset "generic lu" begin
            sol = solve(
                prob, KenCarp47(linsolve = GenericLUFactorization()), abstol = 1.0f-7,
                reltol = 1.0f-7)
            @test sum(Float64.(Array(abs.(sol.u[end]))))≈4.0054012260009766e7 rtol=2e-2
        end

        @testset "krylov" begin
            prob = ODEProblem(csys,
                SolverIMEX(MapKernel(), BlockDiagonalOperatorJacobian(),
                    stiff_sparse = false))
            sol = solve(prob, KenCarp47(linsolve = KrylovJL()), abstol = 1.0f-7,
                reltol = 1.0f-7)
            @test sum(Float64.(Array(abs.(sol.u[end]))))≈3.477207642895508e7 rtol=2e-2 # TODO: Why is this different from above?
        end
    end
end

@testset "SimulatorStrategies" begin
    @testset "Strang Threads" begin
        st = SolverStrangThreads(Tsit5(), 1.0)
        prob = ODEProblem(csys, st)
        sol = solve(prob, Euler(); dt = 1.0, abstol = 1e-12, reltol = 1e-12)
        @test sum(abs.(sol.u[end]))≈3.820642384890682e7 rtol=1e-3
    end

    @testset "Strang Serial" begin
        st = SolverStrangSerial(Tsit5(), 1.0)
        prob = ODEProblem(csys, st)
        sol = solve(prob, Euler(); dt = 1.0, abstol = 1e-12, reltol = 1e-12)
        @test sum(abs.(sol.u[end]))≈3.820642384890682e7 rtol=1e-3
    end

    @testset "issue #219 stiff sub-integrator event_cb" begin
        # Regression test: the discrete event callback returned by
        # `process_events(sys_mtk)` (e.g. EarthSciData's data-load
        # `SymbolicDiscreteCallback`) must reach the inner stiff
        # sub-integrators, otherwise their `init` runs `auto_dt_reset!`
        # against still-uninitialized parameter buffers.
        fired = Ref(0)
        cb_event = DiscreteCallback(
            (u, t, integrator) -> true,
            integrator -> (fired[] += 1))

        st_local = SolverStrangSerial(Tsit5(), 1.0)
        tstart = EarthSciMLBase.get_tspan(domain)[1]

        # With event_cb passed: inner integrator must carry the callback…
        _, integrators_with = EarthSciMLBase._strang_integrators(
            st_local, domain, f_ode, u0_single, tstart, p, cb_event)
        @test length(integrators_with[1].opts.callback.discrete_callbacks) > 0

        # …and it must actually fire during stiff sub-integration.
        ucopy = copy(u)
        fired[] = 0
        EarthSciMLBase.single_ode_step!(
            ucopy, CartesianIndices(tuple(size(domain)...))[1:1],
            integrators_with[1], 0.0, 1.0)
        @test fired[] > 0

        # With event_cb=nothing: no extra discrete callbacks injected.
        _, integrators_none = EarthSciMLBase._strang_integrators(
            st_local, domain, f_ode, u0_single, tstart, p, nothing)
        @test length(integrators_none[1].opts.callback.discrete_callbacks) == 0

        # IIP must be transparent for parameter access so MTK-generated
        # affect code (which expects `integrator.p::MTKParameters`) works
        # against the inner stiff sub-integrators (PR #220 review).
        ii0 = CartesianIndices(tuple(size(domain)...))[1]
        iip = EarthSciMLBase.IIP{typeof(ii0), typeof(p)}(ii0, p)
        @test iip[1] === p[1]
        @test length(iip) == length(p)
        @test SymbolicIndexingInterface.parameter_values(iip) === p
        @test SciMLStructures.canonicalize(SciMLStructures.Tunable(), iip)[1] ==
              SciMLStructures.canonicalize(SciMLStructures.Tunable(), p)[1]
        @test SciMLStructures.ismutablescimlstructure(iip) ==
              SciMLStructures.ismutablescimlstructure(p)

        # MTK's `GeneratedFunctionWrapper` must unwrap IIP at the call
        # boundary, otherwise its `@generated _generated_call` falls into
        # the "single buffer" branch and packs IIP as `(iip, nothing)` —
        # which breaks `___mtkparameters___[i]` indexing inside the affect's
        # generated body (PR #220 follow-up review).
        gfw_oop = ModelingToolkit.GeneratedFunctionWrapper{(2, 3, true)}(
            (u, p_, t) -> p_,                # OOP returns the (unwrapped) p
            (out, u, p_, t) -> (out[1] = p_[1][1]; nothing))
        # OOP: if IIP isn't unwrapped, MTK would pass `(iip, nothing)` and
        # the OOP function would return that 2-tuple, not `iip.p`.
        @test gfw_oop(zeros(1), iip, 0.0) === p
        # IIP variant
        out = zeros(1)
        gfw_oop(out, zeros(1), iip, 0.0)
        # `p[1][1]` (first element of the 1st parameter portion) must be
        # finite — would be undefined if IIP weren't unwrapped.
        @test isfinite(out[1])
    end

    @testset "per-cell callback initialize guard" begin
        # `single_ode_step!` re-runs the callback-`initialize` pass only for
        # the first cell of a chunk when `_reinit_cb_skippable` proves the
        # repeats redundant: preset-time conditions with no preset time
        # strictly inside the inner span and not both t0 and tf preset, or
        # the default `initialize` (e.g. `PositiveDomain`). A callback with a
        # stateful `initialize` (e.g. `PeriodicCallback` re-anchoring its
        # epoch and scheduling its first tstop, which the per-cell `reinit!`
        # tstop wipe would destroy) must keep per-cell initialization.
        st_local = SolverStrangSerial(Tsit5(), 1.0)
        tstart = EarthSciMLBase.get_tspan(domain)[1]
        ncells_probe = 3
        cells = CartesianIndices(tuple(size(domain)...))[1:ncells_probe]

        inits = Ref(0)
        count_init(c, u, t, integrator) = (inits[] += 1)

        # Stateful, non-preset condition: initialization must stay per-cell.
        cb_stateful = DiscreteCallback(
            (u, t, integrator) -> false, integrator -> nothing;
            initialize = count_init)
        _, integs = EarthSciMLBase._strang_integrators(
            st_local, domain, f_ode, u0_single, tstart, p, cb_stateful)
        ncbs = length(integs[1].opts.callback.discrete_callbacks)
        inits[] = 0
        EarthSciMLBase.single_ode_step!(copy(u), cells, integs[1], 0.0, 1.0)
        @test inits[] == ncells_probe * ncbs

        # Preset-time condition with a preset time at t0 only: one
        # initialization for the whole chunk.
        cb_preset = DiffEqCallbacks.PresetTimeCallback(
            [0.0, 5.0], integrator -> nothing; initialize = count_init)
        _, integs_p = EarthSciMLBase._strang_integrators(
            st_local, domain, f_ode, u0_single, tstart, p, cb_preset)
        ncbs_p = length(integs_p[1].opts.callback.discrete_callbacks)
        inits[] = 0
        EarthSciMLBase.single_ode_step!(copy(u), cells, integs_p[1], 0.0, 1.0)
        @test inits[] == ncbs_p

        # Predicate unit checks.
        skippable(cb, t0, tf) = EarthSciMLBase._reinit_cb_skippable(cb, t0, tf)
        cb_pt = DiffEqCallbacks.PresetTimeCallback([0.0, 1.0, 5.0],
            integrator -> nothing)
        cb_pd = DiffEqCallbacks.PositiveDomain(save = false)
        cb_cont = ContinuousCallback((u, t, integrator) -> u[1], integrator -> nothing)
        @test skippable(nothing, 0.0, 1.0)
        # Preset times at t0 only or tf only are skippable; strictly inside the
        # span or at both ends (the fire at tf changes the parameters that
        # only the next cell's t0 fire would restore) are not.
        @test skippable(CallbackSet(cb_pt), 5.0, 6.0)
        @test skippable(CallbackSet(cb_pt), 4.0, 5.0)
        @test !skippable(CallbackSet(cb_pt), 0.5, 1.5)
        @test !skippable(CallbackSet(cb_pt), 0.0, 1.0)
        # A stiff-kwarg callback with the default `initialize`, such as
        # `PositiveDomain`, does not disable the skip: `INITIALIZE_DEFAULT`
        # only clears `u_modified`, which `reinit!` already does.
        @test skippable(CallbackSet(cb_pd), 0.5, 1.5)
        @test skippable(CallbackSet(cb_pt, cb_pd), 5.0, 6.0)
        @test !skippable(CallbackSet(cb_pt, cb_pd), 0.5, 1.5)
        # A custom `initialize` or any continuous callback does.
        @test !skippable(CallbackSet(cb_stateful), 0.0, 1.0)
        @test !skippable(CallbackSet(cb_stateful, cb_pd), 5.0, 6.0)
        @test !skippable(CallbackSet(cb_cont), 5.0, 6.0)
        @test !skippable(CallbackSet(cb_cont, cb_pt, cb_pd), 5.0, 6.0)

        # `abstol = Inf`: this fixture's states go negative by construction,
        # so the default tolerance would shrink dt until maxiters. The
        # `initialize` under test does not depend on it.
        pd = DiffEqCallbacks.PositiveDomain(save = false, abstol = Inf)
        st_pd = SolverStrangSerial(Tsit5(), 1.0; callback = pd)
        is_pd(c) = c.affect! isa DiffEqCallbacks.PositiveDomainAffect
        is_preset(c) = c.condition isa DiffEqCallbacks.PresetTimeFunction

        # Preset-time callback plus PositiveDomain: one initialization of the
        # preset callback for the whole chunk.
        _, integs_ppd = EarthSciMLBase._strang_integrators(
            st_pd, domain, f_ode, u0_single, tstart, p, cb_preset)
        dcbs_ppd = integs_ppd[1].opts.callback.discrete_callbacks
        @test any(is_pd, dcbs_ppd)
        inits[] = 0
        EarthSciMLBase.single_ode_step!(copy(u), cells, integs_ppd[1], 0.0, 1.0)
        @test inits[] == count(is_preset, dcbs_ppd)

        # Stateful callback plus PositiveDomain: still initialized per cell.
        _, integs_spd = EarthSciMLBase._strang_integrators(
            st_pd, domain, f_ode, u0_single, tstart, p, cb_stateful)
        dcbs_spd = integs_spd[1].opts.callback.discrete_callbacks
        @test any(is_pd, dcbs_spd)
        inits[] = 0
        EarthSciMLBase.single_ode_step!(copy(u), cells, integs_spd[1], 0.0, 1.0)
        @test inits[] == ncells_probe * count(!is_pd, dcbs_spd)

        # Threaded steps with PositiveDomain plus a data-like preset-time
        # callback whose affect writes a parameter the RHS reads, as a data
        # load writes the shared parameter object. Skipping the repeated
        # initializations must reproduce the per-cell result exactly. The
        # per-cell reference swaps PositiveDomain for a copy whose custom but
        # equivalent `initialize` disables the skip. Preset times sit only at
        # step starts: a fire at tf would race against other chunks' solves.
        αsym = only(filter(x -> endswith(string(x), "α"), parameters(sys_coords)))
        getα = SymbolicIndexingInterface.getp(sys_coords, αsym)
        setα! = SymbolicIndexingInterface.setp(sys_coords, αsym)
        α0 = getα(p)
        pd_percell = DiscreteCallback(pd.condition, pd.affect!;
            initialize = (c, u, t, integrator) -> u_modified!(integrator, false),
            save_positions = pd.save_positions)
        # Cells with positive lon and lat, where the solution depends on α.
        II = CartesianIndices(tuple(size(domain)...))
        cells_t = vec(II[(end - 3):end, (end - 2):end, 1])
        chunks = collect(Iterators.partition(cells_t, 4))
        u_t0 = reshape(EarthSciMLBase.init_u(sys_coords, domain), :, size(domain)...)
        ninit = Threads.Atomic{Int}(0)
        function run_threaded(stiff_cb, αfun)
            cb_data = DiffEqCallbacks.PresetTimeCallback([0.0, 2.0, 10.0],
                integrator -> setα!(integrator.p.p, αfun(integrator.t));
                initialize = (c, u, t, integrator) -> Threads.atomic_add!(ninit, 1))
            st_t = SolverStrangThreads(length(chunks), Tsit5(), 1.0; callback = stiff_cb)
            _, integs_t = EarthSciMLBase._strang_integrators(
                st_t, domain, f_ode, u0_single, tstart, p, cb_data)
            integs_t = integs_t[1:length(chunks)]
            ut = copy(u_t0)
            ninit[] = 0
            for t0 in (0.0, 2.0)
                EarthSciMLBase.threaded_ode_step!(ut, chunks, integs_t, t0, 1.0)
            end
            dcbs = integs_t[1].opts.callback.discrete_callbacks
            return (u = ut[:, cells_t], inits_per_cb = ninit[] ÷ count(is_preset, dcbs),
                skip = skippable(integs_t[1].opts.callback, 0.0, 1.0))
        end
        r_skip = run_threaded(pd, t -> 10.0 + t)
        r_ref = run_threaded(pd_percell, t -> 10.0 + t)
        r_other = run_threaded(pd, t -> 20.0 + t)
        setα!(p, α0)
        @test r_skip.skip && !r_ref.skip
        @test r_skip.u == r_ref.u
        @test r_skip.u != r_other.u  # the data callback reaches these cells
        @test r_skip.inits_per_cb == 2 * length(chunks)  # once per chunk per step
        @test r_ref.inits_per_cb == 2 * length(cells_t)  # once per cell per step
    end

    @testset "IMEX" begin
        st = SolverIMEX()
        prob = ODEProblem(csys, st)
        sol = solve(prob, Tsit5())
        @test sum(abs.(sol.u[end]))≈3.444627331604664e7 rtol=1e-3 # No Splitting error in this one.

        for alg in [MapBroadcast(), MapThreads()]
            for iip in [true] # [true, false] oop not currently implemented for BlockDiagonalJacobian.
                for sparse in [false] # [true, false] # BlockDiagonalJacobian not currently implemented for sparse.
                    @testset "alg=$alg; iip=$iip; sparse=$sparse" begin
                        st = SolverIMEX(alg; stiff_sparse = sparse)
                        prob = ODEProblem{iip}(csys, st)
                        sol = solve(prob, KenCarp47(linsolve = LUFactorization()),
                            abstol = 1e-4, reltol = 1e-4)
                        @test sum(abs.(sol.u[end]))≈3.444627331604664e7 rtol=1e-3
                    end
                end
            end
        end
    end
end

mutable struct cbt
    runcount::Int
end
function EarthSciMLBase.init_callback(c::cbt, sys, sys_mtk, coord_args, dom, alg)
    DiscreteCallback((u, t, integrator) -> true,
        (_) -> c.runcount += 1
    )
end

@testset "callback" begin
    runcount = 0
    af(_) = runcount += 1
    cb = DiscreteCallback(
        (u, t, integrator) -> true,
        af
    )
    @testset "strang" begin
        cc = cbt(0)
        csys2 = couple(csys, cb, cc)
        prob = ODEProblem(csys2, st)
        solve(prob, Euler(); dt = 1.0)
        @test runcount > 0
        @test cc.runcount > 0
    end
    runcount = 0
    @testset "imex" begin
        cc = cbt(0)
        csys2 = couple(csys, cb, cc)
        prob = ODEProblem(csys2, SolverIMEX())
        solve(prob, Tsit5(); dt = 1.0)
        @test runcount > 0
        @test cc.runcount > 0
    end
end

@testset "Strang stiff_kwargs strip build-only kwargs" begin
    # Regression test for build-only kwargs: `sparse` is consumed separately
    # for the Jacobian build in `ODEProblem(::CoupledSystem, ::SolverStrang)`
    # and `optimize` is not a solver/init keyword at all. Forwarding either to
    # the inner stiff sub-integrators makes OrdinaryDiffEq's strict keyword
    # check throw ("Unrecognized keyword arguments"), so `_strang_integrators`
    # must strip them while still honoring genuine solver kwargs like `reltol`.
    st_kw = SolverStrangSerial(Tsit5(), 1.0; sparse = true, optimize = false,
        reltol = 1e-2)
    tstart = EarthSciMLBase.get_tspan(domain)[1]

    # Building the sub-integrators calls `init`, which errors if the
    # build-only kwargs are forwarded (i.e. if the strip is reverted).
    _, integrators_kw = EarthSciMLBase._strang_integrators(
        st_kw, domain, f_ode, u0_single, tstart, p, nothing)
    @test length(integrators_kw) == 1
    # Genuine solver kwargs must still reach the inner integrators.
    @test integrators_kw[1].opts.reltol == 1e-2

    # Public API: coupled-problem construction reaches `_strang_integrators`
    # (and consumes `sparse = true` for the Jacobian build), so it must
    # succeed with build-only kwargs present.
    prob_kw = ODEProblem(csys, st_kw)
    @test prob_kw isa ODEProblem
end

# A data-loading event shaped like EarthSciData's: at each preset time it refreshes a
# window of three records of a time series (one before the current record, as
# EarthSciData keeps) and the window's start time, and the RHS interpolates the window
# linearly in time. The record times are integers and the record period is a power of
# two, so every window that holds a time interpolates exactly the same value there.
const DATA_PERIOD = 4.0
data_record(τ) = Float64(1 + mod(round(Int, τ / DATA_PERIOD), 3))
function data_window_interp(t, ts, d1, d2, d3)
    idx = (t - ts) / DATA_PERIOD
    k = clamp(floor(idx), 0.0, 1.0)
    f = idx - k
    return k == 0 ? d1 + f * (d2 - d1) : d2 + f * (d3 - d2)
end
@register_symbolic data_window_interp(t, ts, d1, d2, d3)

# A gridded system forced by that event, with records at `phase + k * DATA_PERIOD`
# scaled by `scale`. The window parameters start as NaN, as data buffers start
# unloaded, and are time-independent `@parameters` like EarthSciData's.
function data_event_csys(phase; scale = 1.0, tmax = 12.0)
    @parameters lon=0.0 lat=0.0 lev=1.0
    @parameters ts=NaN d1=NaN d2=NaN d3=NaN
    @variables c(t) = 1.0
    function load_window(m, o, ctx, integ)
        T = floor((integ.t - phase) / DATA_PERIOD) * DATA_PERIOD + phase
        return (; ts = T - DATA_PERIOD, d1 = scale * data_record(T - DATA_PERIOD),
            d2 = scale * data_record(T), d3 = scale * data_record(T + DATA_PERIOD))
    end
    aff = ModelingToolkit.ImperativeAffect(load_window, modified = (; ts, d1, d2, d3))
    ev = ModelingToolkit.SymbolicDiscreteCallback(
        collect(phase:DATA_PERIOD:(tmax + DATA_PERIOD)), aff;
        initialize = aff, initialize_save_discretes = false)
    @named data = System(
        [D(c) ~ data_window_interp(t, ts, d1, d2, d3) * (1 + lat^2) * lev - (1 + lon^2) * c],
        t; discrete_events = [ev])
    dom = DomainInfo(constIC(0.0, t ∈ Interval(0.0, tmax)),
        constBC(0.0, lon ∈ Interval(-1.0, 1.0), lat ∈ Interval(-0.5, 0.5),
            lev ∈ Interval(1.0, 2.0)); grid_spacing = [0.5, 0.5, 1.0])
    return couple(data, dom), dom
end

@testset "Strang inner events" begin
    # The stiff integrators of a problem: the stiff-step callback, last in the
    # problem's callbacks, holds them.
    stiff_integrators(prob) = prob.kwargs[:callback].discrete_callbacks[end].affect!.affect!.integrators
    n_events(integ) = count(c -> c.condition isa DiffEqCallbacks.PresetTimeFunction,
        integ.opts.callback.discrete_callbacks)
    # The data window parameters, by name.
    function window(prob)
        sys = prob.f.sys
        val(name) = only(SymbolicIndexingInterface.getp(sys, p)(prob.p)
        for p in parameters(sys) if endswith(string(p), name))
        return [val("₊ts"), val("₊d1"), val("₊d2"), val("₊d3")]
    end
    # The window loaded at t = 0 when the latest record before it is at T.
    window0(T) = [T - DATA_PERIOD, data_record(T - DATA_PERIOD), data_record(T),
        data_record(T + DATA_PERIOD)]
    solve_strang(prob) = solve(prob, Euler(); dt = 1.0)

    csys_b, _ = data_event_csys(0.0)

    @testset "data times on step boundaries" begin
        # The stiff integrators do without the event: the outer problem fires it
        # before each stiff step. Its initialize ran once before the integrators were
        # built, so their `init` saw loaded data (issue #219).
        prob = ODEProblem(csys_b, SolverStrangSerial(Tsit5(), 1.0))
        integs = stiff_integrators(prob)
        @test all(i -> n_events(i) == 0, integs)
        @test window(prob) == window0(0.0)
        @test all(i -> isfinite(i.dt) && all(isfinite, i.fsalfirst), integs)
        @test all(i -> i.p.p === prob.p, integs)

        # `inner_events = true` keeps one copy of the event per stiff integrator (a
        # callback given to both the problem and `init` would run twice).
        prob_in = ODEProblem(csys_b, SolverStrangSerial(Tsit5(), 1.0; inner_events = true))
        @test all(i -> n_events(i) == 1, stiff_integrators(prob_in))

        sol = solve_strang(prob)
        sol_in = solve_strang(prob_in)
        @test sol.retcode == sol_in.retcode == ReturnCode.Success
        @test sol.t == sol_in.t
        @test sol.u == sol_in.u
        # The data reach the solution.
        csys_other, _ = data_event_csys(0.0; scale = 2.0)
        @test solve_strang(ODEProblem(csys_other, SolverStrangSerial(Tsit5(), 1.0))).u[end] !=
              sol.u[end]

        # Each cell's solve starts from its own RHS (FSAL `fsalfirst` is evaluated
        # after the cell index is set), so the block layout does not change the result.
        for alg in (Tsit5(), TRBDF2())
            ref = solve_strang(ODEProblem(csys_b, SolverStrangSerial(alg, 1.0))).u[end]
            for n in (2, 7)
                prob_t = ODEProblem(csys_b, SolverStrangThreads(n, alg, 1.0))
                @test solve_strang(prob_t).u[end] == ref
            end
        end
    end

    @testset "data time inside a step" begin
        # Data times 1.5 + 4k fall inside the steps (1, 2), (5, 6), …: the stiff
        # integrators keep the event.
        csys_i, _ = data_event_csys(1.5)
        prob = ODEProblem(csys_i, SolverStrangSerial(Tsit5(), 1.0))
        @test all(i -> n_events(i) == 1, stiff_integrators(prob))
        @test window(prob) == window0(-2.5)
        @test solve_strang(prob).retcode == ReturnCode.Success
        prob_out = ODEProblem(csys_i, SolverStrangSerial(Tsit5(), 1.0; inner_events = false))
        @test all(i -> n_events(i) == 0, stiff_integrators(prob_out))
        @test window(prob_out) == window0(-2.5)
        @test solve_strang(prob_out).retcode == ReturnCode.Success
    end

    @testset "events that touch unknowns" begin
        # The outer integrator's `u` holds every grid cell, so an event that reads or
        # changes an unknown stays in the stiff integrators.
        _, dom = data_event_csys(0.0)
        function event_csys(kind)
            @parameters lon=0.0 lat=0.0 lev=1.0 k=1.0
            @variables c(t) = 1.0
            ev = if kind === :reset
                ModelingToolkit.SymbolicDiscreteCallback([4.0, 8.0] => [c ~ 0.5]; iv = t)
            elseif kind === :reads
                ModelingToolkit.SymbolicDiscreteCallback([4.0, 8.0],
                    ModelingToolkit.ImperativeAffect((m, o, ctx, integ) -> (; k = o.c),
                        modified = (; k), observed = (; c)))
            else
                ModelingToolkit.SymbolicDiscreteCallback([4.0, 8.0],
                    ModelingToolkit.ImperativeAffect((m, o, ctx, integ) -> (; k = 2.0),
                        modified = (; k)))
            end
            @named sys_ev = System([D(c) ~ -k * c * lev + lat + lon], t;
                discrete_events = [ev])
            return couple(sys_ev, dom)
        end
        events(kind) = n_events(only(stiff_integrators(
            ODEProblem(event_csys(kind), SolverStrangSerial(Tsit5(), 1.0)))))
        @test events(:reset) == 1
        @test events(:reads) == 1
        @test events(:params) == 0
    end

    @testset "split-step boundaries" begin
        on_boundary(τ, start, Δ) = EarthSciMLBase._on_split_boundary(τ, start, Δ)
        @test on_boundary(4.0, 0.0, 1.0)
        @test !on_boundary(4.5, 0.0, 1.0)
        @test on_boundary(-3.0, 1.0, 2.0)
        @test on_boundary(4.0, 0.0, 1.0f0)
        # The stiff step fires at 0.0 + 3 * 0.1 = 0.30000000000000004, so 0.3 is
        # strictly inside the step before it.
        @test !on_boundary(0.3, 0.0, 0.1)
        @test on_boundary(0.0 + 3 * 0.1, 0.0, 0.1)
        @test_throws ArgumentError ODEProblem(csys_b,
            SolverStrangSerial(Tsit5(), 1.0; inner_events = :yes))
    end

    @testset "stiff callback copied per block" begin
        pd = DiffEqCallbacks.PositiveDomain(zeros(1); save = false)
        prob = ODEProblem(csys_b, SolverStrangThreads(3, Tsit5(), 1.0; callback = pd))
        integs = stiff_integrators(prob)
        @test length(integs) == 3
        is_pd(c) = c.affect! isa DiffEqCallbacks.PositiveDomainAffect
        @test all(i -> count(is_pd, i.opts.callback.discrete_callbacks) == 1, integs)
        bufs = [only(c.affect!.u for c in i.opts.callback.discrete_callbacks if is_pd(c))
                for i in integs]
        @test allunique(objectid.(bufs))
        @test !any(b -> b === pd.affect!.u, bufs)
        prob_s = ODEProblem(csys_b, SolverStrangSerial(Tsit5(), 1.0; callback = pd))
        @test solve_strang(prob).u[end] == solve_strang(prob_s).u[end]
    end
end
