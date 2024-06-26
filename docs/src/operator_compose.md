# Operator Composition

There are a lot of cases where there are two different "processes" or "operators" that change the same variable. 
For example, CO2 in the atmosphere can be emitted by human activity, and it can also be absorbed by the ocean.
In models, typically the emission and removal are considered separate processes which are represented by separate
model components.
However, when we want to combine these two components into a single model, we need to be able to compose them together.

We can use the [`operator_compose`](@ref operator_compose) function for this. It composes to systems of equations together by adding the right-hand side terms together of equations that have matching left-hand sides.
The left hand sides of two equations will be considered matching if:

1. They are both time derivatives of the same variable.
2. The first one is a time derivative of a variable and the second one is the variable itself.
3. There is an entry in the optional `translate` dictionary that maps the dependent variable in the first system to the dependent variable in the second system, e.g. `Dict(sys1.sys.x => sys2.sys.y)`.
4. There is an entry in the optional `translate` dictionary that maps the dependent variable in the first system to the dependent variable in the second system, with a conversion factor, e.g. `Dict(sys1.sys.x => sys2.sys.y => 6)`.

Perhaps we can make this somewhat clearer with some examples.

## Examples

### Example with matching variable time derivatives

The example below shows that when we `operator_compose` two systems together that are both equal to `D(x) = p`, the resulting system is equal to `D(x) = 2p`.

```@example operator_compose
using EarthSciMLBase
using ModelingToolkit

@parameters t

function ExampleSys(t)
    @variables x(t)
    @parameters p
    D = Differential(t)
    ODESystem([D(x) ~ p], t; name=:Docs₊ExampleSys)
end

ExampleSys(t)
```

```@example operator_compose
function ExampleSys2(t)
    @variables x(t)
    @parameters p
    D = Differential(t)
    ODESystem([D(x) ~ 2p], t; name=:Docs₊ExampleSys2)
end

ExampleSys2(t)
```

```@example operator_compose
sys1 = ExampleSys(t)
sys2 = ExampleSys2(t)

register_coupling(ExampleSys(t), ExampleSys2(t)) do sys1, sys2
    operator_compose(sys1, sys2)
end

combined = couple(sys1, sys2)

combined_mtk = get_mtk(combined)
```

The simplified equation should be D(x) = p + sys2_xˍt:
```@example operator_compose
combined_simplified = structural_simplify(combined_mtk)
```

 where sys2_xˍt is also equal to p:
```@example operator_compose
observed(combined_simplified)
```

### Example with non-matching variables

This example demonstrates a case where one variable in the first system is equal to another variable in the second system:


```@example operator_compose
function ExampleSys3(t)
    @variables y(t)
    @parameters p
    D = Differential(t)
    ODESystem([D(y) ~ p], t; name=:Docs₊ExampleSys3)
end

sys1 = ExampleSys(t)
sys2 = ExampleSys3(t)

register_coupling(ExampleSys(t), ExampleSys3(t)) do sys1, sys2
    operator_compose(sys1, sys2, Dict(sys1.x => sys2.y))
end

combined = couple(sys1, sys2)
combined_simplified = structural_simplify(get_mtk(combined))
```

```@example operator_compose
observed(combined_simplified)
```

### Example with a non-ODE system

In the second case above, we might have a variable in the second system that is equal to a rate, but it is not a time derivative.
This could happen if we are extracting emissions from a file, and those emissions are already in units of kg/s, or something similar. The example below demonstrates this case. 
(Note that this case can also be used with the conversion factors shown in the last example.)

```@example operator_compose
function ExampleSysNonODE(t)
    @variables y(t)
    @parameters p
    ODESystem([y ~ p], t; name=:Docs₊ExampleSysNonODE)
end

sys1 = ExampleSys(t)
sys2 = ExampleSysNonODE(t)

register_coupling(ExampleSys(t), ExampleSysNonODE(t)) do sys1, sys2
    operator_compose(sys1, sys2, Dict(sys1.x => sys2.y))
end

combined = couple(sys1, sys2)
sys_combined = structural_simplify(get_mtk(combined))
```

```@example operator_compose
observed(sys_combined)
```

### Example with non-matching variables and a conversion factor

Finally, this last example shows the fourth case, where a conversion factor is included in the translation dictionary.

```@example operator_compose
function ExampleSysNonODE2(t)
    @variables y(t)
    @parameters p
    ODESystem([y ~ p], t; name=:Docs₊ExampleSysNonODE2)
end

sys1 = ExampleSys(t)
sys2 = ExampleSysNonODE2(t)

register_coupling(ExampleSys(t), ExampleSysNonODE2(t)) do sys1, sys2
    operator_compose(sys1, sys2, Dict(sys1.x => sys2.y => 6.0))
end

combined = couple(sys1, sys2)
combined_simplified = structural_simplify(get_mtk(combined))
```

```@example operator_compose
observed(combined_simplified)
```
