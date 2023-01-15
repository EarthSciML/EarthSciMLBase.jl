using EarthSciMLBase
using ModelingToolkit

@testset "Composed System" begin
    @parameters t

    struct SEqn <: EarthSciMLSystem
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
    
    struct IEqn <: EarthSciMLSystem
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
    
    struct REqn <: EarthSciMLSystem
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
    
    sir_want = compose(ODESystem([
            ieqn.sys.S ~ seqn.sys.S,
            seqn.sys.I ~ ieqn.sys.I,
            seqn.sys.R ~ reqn.sys.R,
            ieqn.sys.R ~ reqn.sys.R,
            reqn.sys.I ~ ieqn.sys.I], t, [], [], name=:connection,
        ), seqn.sys, ieqn.sys, reqn.sys)
    
    hasalleq = true
    for eqwant ∈ equations(sir_want)
        haseq = false
        for eq ∈ equations(sirfinal)
            if isequal(eqwant, eq)
                haseq = true
                break
            end
        end
        if !haseq
            hasalleq = false
        end
    end
    @test hasalleq
    
    # Make sure it can be simplified.
    @testset "simplify" begin structural_simplify(sirfinal) end
end