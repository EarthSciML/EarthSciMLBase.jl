using EarthSciMLBase
using ModelingToolkit

@testset "Composed System" begin
    @parameters t

    struct SEqn <: EarthSciMLODESystem
        sys::ODESystem
    
        function SEqn(t) 
            @variables S(t), I(t), R(t)
            D = Differential(t)
            N = S + I + R
            @parameters β
            @named seqn = ODESystem([D(S) ~ -β*S*I/N])
            new(seqn)
        end
    end
    
    struct IEqn <: EarthSciMLODESystem
        sys::ODESystem
    
        function IEqn(t) 
            @variables S(t), I(t), R(t)
            D = Differential(t)
            N = S + I + R
            @parameters β,γ
            @named ieqn = ODESystem([D(I) ~ β*S*I/N-γ*I])
            new(ieqn)
        end
    end
    
    struct REqn <: EarthSciMLODESystem
        sys::ODESystem
    
        function REqn(t) 
            @variables I(t), R(t)
            D = Differential(t)
            @parameters γ
            @named reqn = ODESystem([D(R) ~ γ*I])
            new(reqn)
        end
    end
    
    function Base.:(+)(s::SEqn, i::IEqn)::ComposedEarthSciMLSystem
        seqn = s.sys
        ieqn = i.sys
        ComposedEarthSciMLSystem(
            ConnectorSystem([
                ieqn.S ~ seqn.S,
                seqn.I ~ ieqn.I], s, i), 
            s, i,
        )
    end
    
    function Base.:(+)(s::SEqn, r::REqn)::ComposedEarthSciMLSystem
        seqn = s.sys
        reqn = r.sys
        ComposedEarthSciMLSystem(
            ConnectorSystem([seqn.R ~ reqn.R], s, r), 
            s, r,
        )
    end
    
    function Base.:(+)(i::IEqn, r::REqn)::ComposedEarthSciMLSystem
        ieqn = i.sys
        reqn = r.sys
        ComposedEarthSciMLSystem(
            ConnectorSystem([
                ieqn.R ~ reqn.R,
                reqn.I ~ ieqn.I], i, r), 
            i, r,
        )
    end
    
    seqn, ieqn, reqn = SEqn(t), IEqn(t), REqn(t)
    
    sir = seqn + ieqn + reqn
    
    sirfinal = get_mtk(sir)
    
    sir_simple = structural_simplify(sirfinal)

    want_eqs = [
        Differential(t)(reqn.R) ~ reqn.γ*reqn.I, 
        Differential(t)(seqn.S) ~ (-seqn.β*seqn.I*seqn.S) / (seqn.I + seqn.R + seqn.S),
        Differential(t)(ieqn.I) ~ (ieqn.β*ieqn.I*ieqn.S) / (ieqn.I + ieqn.R + ieqn.S) - ieqn.γ*ieqn.I,
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

        g = graph(sir)
        l = collect(labels(g))
        el = collect(edge_labels(g))

        @test sort(l) == sort([:seqn, :ieqn, :reqn])
        @test length(el) == 3
    end
end