using EarthSciMLBase
using ModelingToolkit
using ModelingToolkit: t, D
using ModelingToolkit: t_nounits, D_nounits
using Test
using DynamicQuantities
using OrdinaryDiffEqTsit5

@testset "Composed System" begin
    struct SEqnCoupler
        sys::Any
    end
    function SEqn()
        @variables S(t), I(t), R(t)
        N = S + I + R
        @parameters β [unit = u"s^-1"]
        System([D(S) ~ -β * S * I / N], t; name = :seqn,
            metadata = Dict(CoupleType => SEqnCoupler))
    end

    struct IEqnCoupler
        sys::Any
    end
    function IEqn()
        @variables S(t), I(t), R(t)
        N = S + I + R
        @parameters β [unit = u"s^-1"]
        @parameters γ [unit = u"s^-1"]
        System([D(I) ~ β * S * I / N - γ * I], t; name = :ieqn,
            metadata = Dict(CoupleType => IEqnCoupler))
    end

    struct REqnCoupler
        sys::Any
    end
    function REqn()
        @variables I(t), R(t)
        @parameters γ [unit = u"s^-1"]
        System([D(R) ~ γ * I], t; name = :reqn,
            metadata = Dict(CoupleType => REqnCoupler))
    end

    function EarthSciMLBase.couple2(s::SEqnCoupler, i::IEqnCoupler)
        s, i = s.sys, i.sys
        ConnectorSystem([
                i.S ~ s.S,
                s.I ~ i.I], s, i)
    end

    function EarthSciMLBase.couple2(s::SEqnCoupler, r::REqnCoupler)
        s, r = s.sys, r.sys
        ConnectorSystem([s.R ~ r.R], s, r)
    end

    function EarthSciMLBase.couple2(i::IEqnCoupler, r::REqnCoupler)
        i, r = i.sys, r.sys
        ConnectorSystem([
                i.R ~ r.R,
                r.I ~ i.I], i, r)
    end

    seqn, ieqn, reqn = SEqn(), IEqn(), REqn()

    sir = couple(seqn, ieqn, reqn)

    sirfinal = convert(System, sir)

    have_eqs = equations(sirfinal)
    obs = ModelingToolkit.observed(sirfinal)

    # Check that the expected equations are present (allowing for equivalent simplifications)
    have_str = string(have_eqs)
    @test occursin("reqn₊γ", have_str) && occursin("reqn₊I", have_str) &&
          occursin("reqn₊R", have_str)
    @test occursin("seqn₊β", have_str) && occursin("seqn₊S", have_str) &&
          occursin("seqn₊I", have_str)
    @test occursin("ieqn₊β", have_str) && occursin("ieqn₊S", have_str) &&
          occursin("ieqn₊I", have_str)
    @test length(have_eqs) == 3

    @testset "Graph" begin
        using MetaGraphsNext

        seqn, ieqn, reqn = SEqn(), IEqn(), REqn()
        sir = couple(seqn, ieqn, reqn)

        g = graph(sir)
        l = collect(labels(g))
        el = collect(edge_labels(g))

        @test sort(l) == sort([:seqn, :ieqn, :reqn])
        @test length(el) == 3
    end
end

@testset "Composed System Permutations" begin
    struct ACoupler
        sys::Any
    end
    function A()
        @parameters j_unit=1 [unit = u"s^-1"]
        @variables j_NO2(t)=0.0149 [unit = u"s^-1"]
        eqs = [
            j_NO2 ~ j_unit
        ]
        System(eqs, t, [j_NO2], [j_unit]; name = :a,
            metadata = Dict(CoupleType => ACoupler))
    end

    struct BCoupler
        sys::Any
    end
    function B()
        @parameters jNO2=0.0149 [unit = u"s^-1"]
        @variables NO2(t) = 10.0
        eqs = [
            D(NO2) ~ -jNO2 * NO2
        ]
        System(eqs, t; name = :b, metadata = Dict(CoupleType => BCoupler))
    end

    function EarthSciMLBase.couple2(b::BCoupler, a::ACoupler)
        a, b = a.sys, b.sys
        b = param_to_var(b, :jNO2)
        ConnectorSystem([b.jNO2 ~ a.j_NO2], b, a)
    end

    struct CCoupler
        sys::Any
    end
    function C()
        @parameters emis=1 [unit = u"s^-1"]
        @variables NO2(t)=0.00014 [unit = u"s^-1"]
        eqs = [NO2 ~ emis]
        System(eqs, t, [NO2], [emis]; name = :c, metadata = Dict(CoupleType => CCoupler))
    end

    function EarthSciMLBase.couple2(b::BCoupler, c::CCoupler)
        b, c = b.sys, c.sys
        operator_compose(b, c, Dict(
            b.NO2 => c.NO2,
        ))
    end

    models = [couple(A(), B(), C())
              couple(C(), B(), A())
              couple(B(), A(), C())
              couple(C(), A(), B())
              couple(A(), C(), B())
              couple(B(), C(), A())]
    for (i, model) in enumerate(models)
        @testset "permutation $i" begin
            m = convert(System, model)
            eqstr = string(equations(m))
            @test occursin("b₊c_NO2(t)", eqstr)
            @test occursin("b₊jNO2(t)", eqstr)
            @test occursin("b₊NO2(t)", string(unknowns(m)))
            obstr = string(ModelingToolkit.observed(m))
            @test occursin("a₊j_NO2(t) ~ a₊j_unit", obstr)
            @test occursin("c₊NO2(t) ~ c₊emis", obstr)
            @test occursin("b₊jNO2(t) ~ a₊j_NO2(t)", obstr)
            @test occursin("b₊c_NO2(t) ~ c₊NO2(t)", obstr)
        end
    end

    @testset "Stable evaluation" begin
        sys = couple(A(), B(), C())
        s = convert(System, sys)
        eqs1 = string(equations(s))
        @test occursin("b₊c_NO2(t)", eqs1)
        eqs2 = string(equations(s))
        @test occursin("b₊c_NO2(t)", eqs2)
        @test eqs1 == eqs2
    end
end

mutable struct ParamTest
    y::Any
end
(t::ParamTest)(x) = t.y - x

function update_affect!(mod, obs, ctx, integ)
    mod.p.y = integ.t
    mod
end

@testset "Event filtering" begin
    p1 = ParamTest(1)
    tp1 = typeof(p1)
    @parameters (p_1::tp1)(..) = p1
    @discretes p_2(t_nounits) = 1
    @parameters (p_3::tp1)(..) = ParamTest(1)
    @discretes p_4(t_nounits) = 1
    @variables x(t_nounits) = 0
    @variables x2(t_nounits) = 0
    @variables x3(t_nounits)

    event1 = [1.0, 2, 3] => (f = update_affect!, modified = (p = p_1,))
    event2 = [
        1.0, 2, 3] => (f = (mod, obs, ctx, integ) -> (p_2 = 1,), modified = (p_2 = p_2,))
    event3 = [1.0, 2, 3] => (f = update_affect!, modified = (p = p_3,))
    event4 = [
        1.0, 2, 3] => (f = (mod, obs, ctx, integ) -> (p_4 = 1,), modified = (p_4 = p_4,))

    sys = System(
        [
            D_nounits(x) ~ p_1(x),
            D_nounits(x2) ~ p_2 - x2,
            x3 ~ p_3(x2) + p_4
        ],
        t_nounits; name = :test,
        discrete_events = [event1, event2, event3, event4]
    )

    prob = ODEProblem(mtkcompile(sys), [], (0, 100))
    sol = solve(prob, Tsit5(), abstol = 1e-8, reltol = 1e-8)
    @test sol[x][end] ≈ 3
    @test sol[x2][end] ≈ 1

    @testset "variable in equations" begin
        sys2 = mtkcompile(sys)
        @test EarthSciMLBase.var_in_eqs(p_1, equations(sys2)) == true
        @test EarthSciMLBase.var_in_eqs(p_2, equations(sys2)) == true
        @test EarthSciMLBase.var_in_eqs(p_3, equations(sys2)) == false
        @test EarthSciMLBase.var_in_eqs(p_4, equations(sys2)) == false
    end
end

@testset "Transient Equality" begin
    struct SourceCoupler
        sys::Any
    end
    function Source(; name = :Source)
        @parameters T_0 = 300
        @variables T(t)
        System([T ~ T_0], t_nounits, [T], [T_0]; name = name,
            metadata = Dict(CoupleType => SourceCoupler))
    end

    struct DestCoupler
        sys::Any
    end
    function Dest(; name = :Dest)
        @parameters T = 400
        @variables x(t)
        System([D_nounits(x) ~ T], t_nounits, [x], [T]; name = name,
            metadata = Dict(CoupleType => DestCoupler))
    end
    struct DestCoupler
        sys::Any
    end
    function Dest2(; name = :Dest2)
        @parameters T = 400
        @variables x(t)
        System([x ~ T], t_nounits, [x], [T]; name = name,
            metadata = Dict(CoupleType => DestCoupler))
    end

    function EarthSciMLBase.couple2(s::SourceCoupler, d::DestCoupler)
        s, d = s.sys, d.sys
        d = param_to_var(d, :T)
        ConnectorSystem([d.T ~ s.T], s, d)
    end

    src = Source()
    @named dst1 = Dest()
    @named dst2 = Dest2()

    for systems in [(src, dst1, dst2),
        (dst1, src, dst2),
        (dst2, src, dst1),
        (src, dst2, dst1),
        (dst1, dst2, src),
        (dst2, dst1, src)]
        coupled_sys = couple(systems...)

        sys = convert(System, coupled_sys)
        equations(sys)
        observed(sys)

        obs_lhss = [eq.lhs for eq in observed(sys)]
        @test contains(string(obs_lhss), "dst1₊T(t)")
        @test contains(string(obs_lhss), "dst2₊T(t)")
    end
end

@testset "sys_discrete_event" begin
    @parameters a=0 b=0
    @variables begin
        x(t_nounits) = 0
        y(t_nounits)
    end

    # Utility function to check if a variable is needed in the system,
    # i.e., if one of the state variables depends on it.
    function is_var_needed(var, sys)
        var = EarthSciMLBase.var2symbol(var)
        if var in EarthSciMLBase.var2symbol.(unknowns(sys))
            return true
        end
        exprs = [eq.rhs for eq in equations(sys)]
        needed_obs = ModelingToolkit.observed_equations_used_by(sys, exprs)
        needed_vars = getproperty.(observed(sys)[needed_obs], :lhs)
        return var in EarthSciMLBase.var2symbol.(needed_vars)
    end

    runcount1 = 0
    function sysevent1(sys)
        function f1!(mod, obs, ctx, integ)
            if is_var_needed(sys.sys1₊x, sys) # Only run if x is needed in the system.
                #global runcount1 += 1
                runcount1 += 1
                return (sys1₊a = 1,)
            end
            return (sys1₊a = mod.sys1₊a,)
        end
        return [3.0] => (f = f1!, modified = (sys1₊a = sys.sys1₊a,))
    end
    runcount2 = 0
    function sysevent2(sys)
        function f2!(mod, obs, ctx, integ)
            if is_var_needed(sys.sys2₊y, sys) # Only run if y is needed in the system.
                #global runcount2 += 1
                runcount2 += 1
                return (sys2₊b = 1,)
            end
            return (sys2₊b = mod.sys2₊b,)
        end
        return [5.0] => (f = f2!, modified = (sys2₊b = sys.sys2₊b,))
    end

    sys1 = System([D_nounits(x) ~ a], t_nounits, [x], [a]; name = :sys1,
        metadata = Dict(SysDiscreteEvent => sysevent1))
    sys2 = System([y ~ b], t_nounits, [y], [b]; name = :sys2,
        metadata = Dict(SysDiscreteEvent => sysevent2))

    model1 = couple(sys1, sys2)
    sys = convert(System, model1)

    @test length(ModelingToolkit.get_discrete_events(sys)) == 2

    sol = solve(ODEProblem(sys, [], (0, 10)), Tsit5())

    # Here the derivative of x is 0 until t = 3, then because of sysevent1 it becomes 1 for
    # the rest of the simulation, so the final value of x should be 7.
    # Because sys2.y is not a state variable, sysevent 2 does not run.
    @test sol[sys.sys1₊x][end] ≈ 7
    @test runcount1 == 1
    @test runcount2 == 0

    ### Now, try again after coupling the two variables together so x depends on y.

    # reset count.
    runcount1, runcount2 = 0, 0

    struct Couple1
        sys::Any
    end
    struct Couple2
        sys::Any
    end

    sys1 = System([D_nounits(x) ~ a], t_nounits, [x], [a]; name = :sys1,
        metadata = Dict(SysDiscreteEvent => sysevent1,
            CoupleType => Couple1))
    sys2 = System([y ~ b], t_nounits, [y], [b]; name = :sys2,
        metadata = Dict(SysDiscreteEvent => sysevent2,
            CoupleType => Couple2))

    function EarthSciMLBase.couple2(s1::Couple1, s2::Couple2)
        s1, s2 = s1.sys, s2.sys
        EarthSciMLBase.operator_compose(s1, s2, Dict(
            s1.x => s2.y
        ))
    end

    model1 = couple(sys1, sys2)
    sys = convert(System, model1)

    @test length(ModelingToolkit.get_discrete_events(sys)) == 2

    sol = solve(ODEProblem(sys, [], (0, 10)), Tsit5())

    # Here the derivative of x is 0 until t = 3, then because of sysevent1 it becomes 1 for
    # until t = 5, and then because of sysevent2 it become 2 for the rest of the simulation.
    @test sol[sys.sys1₊x][end] ≈ 12
    @test runcount1 == 1
    @test runcount2 == 1
end
