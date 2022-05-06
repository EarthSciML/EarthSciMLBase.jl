"""
```julia
    a::Symbolics.Equation + b::Symbolics.Equation
```

Add two equations. If the left-hand-sides of the two equations are 
the same, a single equation is returned with the two right-hand-sides
added together. If the left-hand-sides of the two equations are different,
a vector of the two equations is returned.
"""
function Base.:(+)(a::Equation, b::Equation)::Vector{Equation}
    if isequal(a.lhs, b.lhs)
        return [Equation(a.lhs, a.rhs + b.rhs)]
    else
        return [a, b]
    end
end

"""
```julia
    a::Vector{Symbolics.Equation} + b::Vector{Symbolics.Equation}
```

Add two vectors of equations using the same rules as when adding two individual
equations.
"""
function Base.:(+)(a::Vector{Equation}, b::Vector{Equation})::Vector{Equation}
    o::Vector{Equation} = []
    matchedy = zeros(Bool, length(a))
    for x in b
        matchedx = false
        for (i, y) in enumerate(a)
            if !matchedy[i] && isequal(x.lhs, y.lhs)
                matchedy[i] = true
                matchedx = true
                push!(o, Equation(x.lhs, x.rhs + y.rhs))
                break
            end
        end
        if !matchedx
            push!(o, x)
        end
    end
    for (i, m) ∈ enumerate(matchedy)
        if !m
            push!(o, a[i])
        end
    end
    o
end

Base.:(+)(a::Vector{Equation}, b::Equation)::Vector{Equation} = a + [b]
Base.:(+)(a::Equation, b::Vector{Equation})::Vector{Equation} = [a] + b