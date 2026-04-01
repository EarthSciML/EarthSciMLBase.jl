"""
```julia
add_dims(expression, vars, dims)
add_dims(equation, vars, dims)
```

Add the given dimensions to each variable in `vars` in the given expression
or equation.

Variables that already have some spatial dimensions are extended with any
missing target dimensions. For example, a variable `v(t, x)` promoted
with dims `[t, x, y]` becomes `v(t, x, y)`. Variables that already have
all target dimensions are left unchanged. This allows components that
construct spatially-varying variables directly to be composed with
components that rely on automatic dimension expansion.

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
        isequal(var, newvar) && continue # Skip already-spatial variables.
        # Replace variable with temporary variable, then replace temporary
        # variable with new variable.
        # TODO(CT): Should be able to directly substitute all variables at once but doesn't work.
        exp = substitute_in_deriv(exp, Dict(var => 🦖🌋temp))
        exp = substitute_in_deriv(exp, Dict(🦖🌋temp => newvar))
    end
    exp
end

function add_dims(vars::AbstractVector, dims::AbstractVector)
    o = Num[]
    dim_syms = Set(Symbol.(dims))
    for var in vars
        args = Symbolics.arguments(Symbolics.unwrap(var))
        nargs = length(args)
        if nargs > 1
            # Variable already has some spatial dimensions.
            # Check if it already has ALL target dimensions.
            var_arg_syms = Set(Symbol.(args))
            missing_dims = setdiff(dim_syms, var_arg_syms)
            if isempty(missing_dims)
                # All target dims present — keep as-is
                push!(o, var)
            else
                # Some dims missing — extend with missing ones
                sym = Symbolics.tosymbol(var, escape = false)
                newvar = (@variables $sym(..))[1]
                newvar = add_metadata(newvar, var)
                # Preserve existing arg order, append missing dims in the order they appear in dims
                new_args = Any[args...]
                for d in dims
                    if Symbol(d) in missing_dims
                        push!(new_args, d)
                    end
                end
                push!(o, newvar(new_args...))
            end
        else
            sym = Symbolics.tosymbol(var, escape = false)
            newvar = (@variables $sym(..))[1]
            newvar = add_metadata(newvar, var)
            push!(o, newvar(dims...))
        end
    end
    return o
end

function add_dims(eq::Equation, vars::AbstractVector, dims::AbstractVector)::Equation
    add_dims(eq.lhs, vars, dims) ~ add_dims(eq.rhs, vars, dims)
end
