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
3. There is an entry in the optional `translate` dictionary or array that maps the dependent variable in the first system to the dependent variable in the second system, e.g. `[sys1.sys.x => sys2.sys.y]`.
4. There is an entry in the optional `translate` dictionary or array that maps the dependent variable in the first system to the dependent variable in the second system, with a conversion factor, e.g. `[sys1.sys.x => sys2.sys.y => 6]`.

Perhaps we can make this somewhat clearer with some examples.

## Examples

### Example with matching variable time derivatives

The example below shows that when we `operator_compose` two systems together that are both equal to `D(x) = p`, the resulting system is equal to `D(x) = 2p`.

```@example operator_compose
using EarthSciMLBase
using ModelingToolkit
using ModelingToolkit: t_nounits, D_nounits
t = t_nounits
D = D_nounits


struct ExampleSysCoupler sys end
function ExampleSys()
    @variables x(t)
    @parameters p
    ODESystem([D(x) ~ p], t; name=:ExampleSys,
        metadata=Dict(:coupletype=>ExampleSysCoupler))
end

ExampleSys()
```

```@example operator_compose
struct ExampleSys2Coupler sys end
function ExampleSys2()
    @variables x(t)
    @parameters p
    ODESystem([D(x) ~ 2p], t; name=:ExampleSys2,
        metadata=Dict(:coupletype=>ExampleSys2Coupler))
end

ExampleSys2()
```

```@example operator_compose
sys1 = ExampleSys()
sys2 = ExampleSys2()

function EarthSciMLBase.couple2(sys1::ExampleSysCoupler, sys2::ExampleSys2Coupler)
    sys1, sys2 = sys1.sys, sys2.sys
    operator_compose(sys1, sys2)
end

combined = couple(sys1, sys2)

combined_mtk = convert(ODESystem, combined)
```

The simplified equation should be D(x) = p + sys2_xˍt:

 where sys2_xˍt is also equal to p:
```@example operator_compose
observed(combined_mtk)
```

### Example with non-matching variables

This example demonstrates a case where one variable in the first system is equal to another variable in the second system:


```@example operator_compose
struct ExampleSys3Coupler sys end
function ExampleSys3()
    @variables y(t)
    @parameters p
    ODESystem([D(y) ~ p], t; name=:ExampleSys3,
        metadata=Dict(:coupletype=>ExampleSys3Coupler))
end

sys1 = ExampleSys()
sys2 = ExampleSys3()

function EarthSciMLBase.couple2(sys1::ExampleSysCoupler, sys2::ExampleSys3Coupler)
    sys1, sys2 = sys1.sys, sys2.sys
    operator_compose(sys1, sys2, [sys1.x => sys2.y])
end

combined = couple(sys1, sys2)
combined_simplified = convert(ODESystem, combined)
```

```@example operator_compose
observed(combined_simplified)
```

### Example with a non-ODE system

In the second case above, we might have a variable in the second system that is equal to a rate, but it is not a time derivative.
This could happen if we are extracting emissions from a file, and those emissions are already in units of kg/s, or something similar. The example below demonstrates this case. 
(Note that this case can also be used with the conversion factors shown in the last example.)

```@example operator_compose
struct ExampleSysNonODECoupler sys end
function ExampleSysNonODE()
    @variables y(t)
    @parameters p
    ODESystem([y ~ p], t; name=:ExampleSysNonODE,
        metadata=Dict(:coupletype=>ExampleSysNonODECoupler))
end

sys1 = ExampleSys()
sys2 = ExampleSysNonODE()

function EarthSciMLBase.couple2(sys1::ExampleSysCoupler, sys2::ExampleSysNonODECoupler)
    sys1, sys2 = sys1.sys, sys2.sys
    operator_compose(sys1, sys2, [sys1.x => sys2.y])
end

combined = couple(sys1, sys2)
sys_combined = convert(ODESystem, combined)
```

```@example operator_compose
observed(sys_combined)
```

### Example with non-matching variables and a conversion factor

Finally, this last example shows the fourth case, where a conversion factor is included in the translation dictionary or array.

```@example operator_compose
struct ExampleSysNonODE2Coupler sys end
function ExampleSysNonODE2()
    @variables y(t)
    @parameters p
    ODESystem([y ~ p], t; name=:Docs₊ExampleSysNonODE2,
        metadata=Dict(:coupletype=>ExampleSysNonODE2Coupler))
end

sys1 = ExampleSys()
sys2 = ExampleSysNonODE2()

function EarthSciMLBase.couple2(sys1::ExampleSysCoupler, sys2::ExampleSysNonODE2Coupler)
    sys1, sys2 = sys1.sys, sys2.sys
    operator_compose(sys1, sys2, [sys1.x => sys2.y => 6.0])
end

combined = couple(sys1, sys2)
combined_simplified = convert(ODESystem, combined)
```

```@example operator_compose
observed(combined_simplified)
```

!warning
    The `operator_compose` function will not work correctly if any of the variables to be 
    composed are part of a `NonlinearSystem` rather than an `ODESystem`. The reason for this
    is because `operator_compose` works by matching the left-hand sides of the equations in
    the two systems, but `NonlinearSystem`s move all of the terms to the right-hand side of
    the equation when they are created.