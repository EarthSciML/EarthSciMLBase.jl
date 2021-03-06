export AddDims

"""
```julia
add_dims(expression, vars, dims)
add_dims(equation, vars, dims)
add_dims(ModelingToolkit.AbstractSystem, dims)
add_dims(Catalyst.ReactionSystem, dims)
```

Add the given dimensions to each variable in `vars` in the given expression
or system. 
Each variable in `vars` must be unidimensional, i.e.
defined like `@variables u(t)` rather than `@variables u(..)`.

# Example:
```jldoctest
using EarthSciMLBase, ModelingToolkit

@parameters x y k t
@variables u(t) q(t)
exp = 2u + 3k*q + 1
EarthSciMLBase.add_dims(exp, [u, q], x, y, t)

# output
1 + 2u(x, y, t) + 3k*q(x, y, t)
```
"""
function add_dims(exp, vars, dims::Num...)
    newvars = add_dims(vars, dims...)
    @variables ๐ฆ๐temp # BUG(CT): If someone chooses ๐ฆ๐temp as a variable in their equation this will fail.
    for (var, newvar) โ zip(vars, newvars)
        # Replace variable with temporary variable, then replace temporary
        # variable with new variable.
        # TODO(CT): Should be able to directly substitute all variables at once but doesn't work.
        exp = substitute(exp, Dict(var => ๐ฆ๐temp))
        exp = substitute(exp, Dict(๐ฆ๐temp => newvar))
    end
    exp
end

function add_dims(vars, dims::Num...)
    syms = [Symbolics.tosymbol(x, escape=false) for x in vars]
    o = []
    for xx in syms
        newvar = (@variables $xx(..))[1]
        push!(o, newvar(dims...))
    end
    return o
end

function add_dims(eq::Equation, vars, dims::Num...)::Equation
    add_dims(eq.lhs, vars, dims...) ~ add_dims(eq.rhs, vars, dims...)
end

function add_dims(sys::ModelingToolkit.AbstractSystem, dims::Num...)::Vector{Equation}
    vars = states(sys)
    [add_dims(eq, vars, dims...) for eq in equations(sys)]
end

function add_dims(rxs::Catalyst.ReactionSystem, dims::Num...)::Vector{Equation}
    sys = convert(ODESystem, rxs; combinatoric_ratelaws=false)
    vars = states(sys)
    [add_dims(eq, vars, dims...) for eq in equations(sys)]
end


"""
```julia
AddDims(dims...)
```

Construct an object that can add dimensions to the variables in a 
system of equations. The object can be used using the `+` operator.

# Example:
```jldoctest
using ModelingToolkit

@parameters x y t k
@variables u(t) q(t)
Dt = Differential(t)

eqs = [
    Dt(u) ~ 2u + 3k*q + 1, 
    Dt(q) ~ 3u + k*q + 1
]
@named sys = ODESystem(eqs)
sys + AddDims(x, y, t)

# output
2-element Vector{Equation}:
 Differential(t)(u(x, y, t)) ~ 1 + 2u(x, y, t) + 3k*q(x, y, t)
 Differential(t)(q(x, y, t)) ~ 1 + k*q(x, y, t) + 3u(x, y, t)
```
"""
struct AddDims
    dims
    AddDims(dims::Num...) = new(dims)
end

Base.:(+)(a::Catalyst.ReactionSystem, b::AddDims)::Vector{Equation} = add_dims(a, b.dims...)
Base.:(+)(a::ModelingToolkit.AbstractSystem, b::AddDims)::Vector{Equation} = add_dims(a, b.dims...)
Base.:(+)(a::AddDims, b::ModelingToolkit.AbstractSystem)::Vector{Equation} = b + a