export ICBC, constICBC, dims, domains

abstract type ICBC end

struct constICBC <: ICBC 
    val
    indepdomain::Symbolics.VarDomainPairing
    partialdomains::Vector{Symbolics.VarDomainPairing}
end

function (icbc::constICBC)(sys::ModelingToolkit.ODESystem, )
    dims = [domain.variables for domain in icbc.partialdomains]
    statevars = add_dims(states(sys), [dims...; icbc.indepdomain.variables])
    
    bcs = Equation[]
    
    for state ∈ statevars
        push!(bcs, state.val.f(dims..., icbc.indepdomain.domain.left) ~ icbc.val)
        for (i, domain) ∈ enumerate(icbc.partialdomains)
            for edge ∈ [domain.domain.left, domain.domain.right]
                args = Any[dims..., icbc.indepdomain.variables]
                args[i] = edge
                push!(bcs, state.val.f(args...) ~ icbc.val)
            end
        end
    end

    bcs
end

dims(icbc::constICBC) = [[domain.variables for domain in icbc.partialdomains]..., icbc.indepdomain.variables]

domains(icbc::constICBC) = [icbc.partialdomains..., icbc.indepdomain]

function Base.:(+)(sys::ModelingToolkit.ODESystem, icbc::ICBC)::ModelingToolkit.PDESystem
    dimensions = dims(icbc)
    statevars = states(sys)
    ps = [k => v for (k,v) in sys.defaults] # Add parameters and their default values
    if !all([p ∈ keys(sys.defaults) for p in parameters(sys)])
        error("All parameters in the system of equations must have default values.")
    end
    ivs = dims(icbc) # New dimensions are the independent variables.
    dvs = add_dims(statevars, dimensions) # Add new dimensions to dependent variables.
    eqs = Vector{Equation}([add_dims(eq, statevars, dimensions) for eq in equations(sys)]) # Add new dimensions to equations.
    PDESystem(eqs, icbc(sys), domains(icbc), ivs, dvs, ps, name=sys.name, defaults=sys.defaults)
end

Base.:(+)(icbc::ICBC, sys::ModelingToolkit.ODESystem)::ModelingToolkit.PDESystem = sys + icbc

function Base.:(+)(sys::Catalyst.ReactionSystem, icbc::ICBC)::ModelingToolkit.PDESystem
    convert(ODESystem, sys; combinatoric_ratelaws=false) + icbc
end

Base.:(+)(icbc::ICBC, sys::Catalyst.ReactionSystem)::ModelingToolkit.PDESystem = sys + icbc
