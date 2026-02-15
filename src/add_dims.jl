"""
```julia
add_dims(expression, vars, dims)
add_dims(equation, vars, dims)
```

Add the given dimensions to each variable in `vars` in the given expression
or equation.
Each variable in `vars` must be unidimensional, i.e.
defined like `@variables u(t)` rather than `@variables u(..)`.

# Example:

```jldoctest
using EarthSciMLBase, ModelingToolkit

@parameters x y k t
@variables u(t) q(t)
exp = 2u + 3k * q + 1
EarthSciMLBase.add_dims(exp, [u, q], [x, y, t])

# output

1 + 2u(x, y, t) + 3k*q(x, y, t)
```
"""
function add_dims(exp, vars::AbstractVector, dims::AbstractVector)
    newvars = add_dims(vars, dims)
    @variables 🦖🌋temp # BUG(CT): If someone chooses 🦖🌋temp as a variable in their equation this will fail.
    for (var, newvar) in zip(vars, newvars)
        # Replace variable with temporary variable, then replace temporary
        # variable with new variable.
        # TODO(CT): Should be able to directly substitute all variables at once but doesn't work.
        exp = substitute_in_deriv(exp, Dict(var => 🦖🌋temp))
        exp = substitute_in_deriv(exp, Dict(🦖🌋temp => newvar))
    end
    exp
end

function add_dims(vars::AbstractVector, dims::AbstractVector)
    syms = [Symbolics.tosymbol(x, escape = false) for x in vars]
    o = Num[]
    for (sym, var) in zip(syms, vars)
        newvar = (@variables $sym(..))[1]
        newvar = add_metadata(newvar, var)
        push!(o, newvar(dims...))
    end
    return o
end

function add_dims(eq::Equation, vars::AbstractVector, dims::AbstractVector)::Equation
    add_dims(eq.lhs, vars, dims) ~ add_dims(eq.rhs, vars, dims)
end
