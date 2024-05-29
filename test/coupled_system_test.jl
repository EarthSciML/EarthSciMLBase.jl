using EarthSciMLBase
using ModelingToolkit
using Test
using Catalyst
using Unitful

@testset "Composed System" begin
    function SEqn()
        @variables S(t), I(t), R(t)
        D = Differential(t)
        N = S + I + R
        @parameters β [unit = u"s^-1"]
        @named seqn = ODESystem([D(S) ~ -β * S * I / N])
    end

    function IEqn()
        @variables S(t), I(t), R(t)
        D = Differential(t)
        N = S + I + R
        @parameters β [unit = u"s^-1"]
        @parameters γ [unit = u"s^-1"]
        @named ieqn = ODESystem([D(I) ~ β * S * I / N - γ * I])
    end

    function REqn()
        @variables I(t), R(t)
        D = Differential(t)
        @parameters γ [unit = u"s^-1"]
        @named reqn = ODESystem([D(R) ~ γ * I])
    end

    register_coupling(SEqn(), IEqn()) do s, i
        ConnectorSystem([
                i.S ~ s.S,
                s.I ~ i.I], s, i)
    end

    register_coupling(SEqn(), REqn()) do s, r
        ConnectorSystem([s.R ~ r.R], s, r)
    end

    register_coupling(IEqn(), REqn()) do i, r
        ConnectorSystem([
                i.R ~ r.R,
                r.I ~ i.I], i, r)
    end

    seqn, ieqn, reqn = SEqn(), IEqn(), REqn()

    sir = couple(seqn, ieqn, reqn)

    sirfinal = get_mtk(sir)

    sir_simple = structural_simplify(sirfinal)

    want_eqs = [
        Differential(t)(reqn.R) ~ reqn.γ * reqn.I,
        Differential(t)(seqn.S) ~ (-seqn.β * seqn.I * seqn.S) / (seqn.I + seqn.R + seqn.S),
        Differential(t)(ieqn.I) ~ (ieqn.β * ieqn.I * ieqn.S) / (ieqn.I + ieqn.R + ieqn.S) - ieqn.γ * ieqn.I,
    ]

    have_eqs = equations(sir_simple)
    for eq in want_eqs
        @test eq in have_eqs
    end
    for eq in have_eqs
        @test eq in want_eqs
    end

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
    function A()
        @parameters j_unit = 1 [unit = u"s^-1"]
        @variables j_NO2(t) = 0.0149 [unit = u"s^-1"]
        eqs = [
            j_NO2 ~ j_unit
        ]
        ODESystem(eqs, t, [j_NO2], [j_unit]; name=:a)
    end

    function B()
        @parameters jNO2 = 0.0149 [unit = u"s^-1"]
        @species NO2(t) = 10.0
        rxs = [
            Reaction(jNO2, [NO2], [], [1], [1])
        ]
        ReactionSystem(rxs, t; combinatoric_ratelaws=false, name=:b)
    end

    register_coupling(B(), A()) do b, a
        b = param_to_var(convert(ODESystem, b), :jNO2)
        ConnectorSystem([b.jNO2 ~ a.j_NO2], b, a)
    end

    function C()
        @parameters emis = 1 [unit = u"s^-1"]
        @variables NO2(t) = 0.00014 [unit = u"s^-1"]
        eqs = [NO2 ~ emis]
        ODESystem(eqs, t, [NO2], [emis]; name=:c)
    end

    register_coupling(B(), C()) do b, c
        @constants uu = 1
        operator_compose(convert(ODESystem, b), c, Dict(
            b.NO2 => c.NO2 => uu,
        ))
    end

    models = [
        couple(A(), B(), C())
        couple(C(), B(), A())
        couple(B(), A(), C())
        couple(C(), A(), B())
        couple(A(), C(), B())
        couple(B(), C(), A())
    ]
    for model ∈ models
        model_mtk = get_mtk(model)
        m = structural_simplify(model_mtk)
        eqstr = string(equations(m))
        @test occursin("b₊c_NO2(t)", eqstr)
        @test occursin("b₊jNO2(t)", eqstr)
        @test occursin("b₊NO2(t)", string(states(m)))
        obstr = string(observed(m))
        @test occursin("a₊j_NO2(t) ~ a₊j_unit", obstr)
        @test occursin("c₊NO2(t) ~ c₊emis", obstr)
        @test occursin("b₊jNO2(t) ~ a₊j_NO2(t)", obstr)
        @test occursin("b₊c_NO2(t) ~ c₊NO2(t)", obstr)
    end

    @testset "Stable evaluation" begin
        sys = couple(A(), B(), C())
        eqs1 = string(equations(get_mtk(sys)))
        @test occursin("b₊c_NO2(t)", eqs1)
        eqs2 = string(equations(get_mtk(sys)))
        @test occursin("b₊c_NO2(t)", eqs2)
        @test eqs1 == eqs2
    end
end