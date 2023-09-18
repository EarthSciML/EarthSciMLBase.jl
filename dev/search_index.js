var documenterSearchIndex = {"docs":
[{"location":"comp_viz/","page":"Composition and Visualization","title":"Composition and Visualization","text":"CurrentModule = EarthSciMLBase","category":"page"},{"location":"comp_viz/#Example-using-composition-and-visualization","page":"Composition and Visualization","title":"Example using composition and visualization","text":"","category":"section"},{"location":"comp_viz/","page":"Composition and Visualization","title":"Composition and Visualization","text":"using EarthSciMLBase\nusing ModelingToolkit\nusing MetaGraphsNext\nusing CairoMakie, GraphMakie\n\n@parameters t\n\nstruct SEqn <: EarthSciMLODESystem\n    sys::ODESystem\n\n    function SEqn(t) \n        @variables S(t), I(t), R(t)\n        D = Differential(t)\n        N = S + I + R\n        @parameters β\n        @named seqn = ODESystem([D(S) ~ -β*S*I/N])\n        new(seqn)\n    end\nend\n\nstruct IEqn <: EarthSciMLODESystem\n    sys::ODESystem\n\n    function IEqn(t) \n        @variables S(t), I(t), R(t)\n        D = Differential(t)\n        N = S + I + R\n        @parameters β,γ\n        @named ieqn = ODESystem([D(I) ~ β*S*I/N-γ*I])\n        new(ieqn)\n    end\nend\n\nstruct REqn <: EarthSciMLODESystem\n    sys::ODESystem\n\n    function REqn(t) \n        @variables I(t), R(t)\n        D = Differential(t)\n        @parameters γ\n        @named reqn = ODESystem([D(R) ~ γ*I])\n        new(reqn)\n    end\nend\n\nfunction Base.:(+)(s::SEqn, i::IEqn)::ComposedEarthSciMLSystem\n    seqn = s.sys\n    ieqn = i.sys\n    ComposedEarthSciMLSystem(\n        ConnectorSystem([\n            ieqn.S ~ seqn.S,\n            seqn.I ~ ieqn.I], s, i), \n        s, i,\n    )\nend\n\nfunction Base.:(+)(s::SEqn, r::REqn)::ComposedEarthSciMLSystem\n    seqn = s.sys\n    reqn = r.sys\n    ComposedEarthSciMLSystem(\n        ConnectorSystem([seqn.R ~ reqn.R], s, r), \n        s, r,\n    )\nend\n\nfunction Base.:(+)(i::IEqn, r::REqn)::ComposedEarthSciMLSystem\n    ieqn = i.sys\n    reqn = r.sys\n    ComposedEarthSciMLSystem(\n        ConnectorSystem([\n            ieqn.R ~ reqn.R,\n            reqn.I ~ ieqn.I], i, r), \n        i, r,\n    )\nend\n\nseqn, ieqn, reqn = SEqn(t), IEqn(t), REqn(t)\n\nsir = seqn + ieqn + reqn\n\ng = graph(sir)\n\nf, ax, p = graphplot(g; ilabels=labels(g))\nhidedecorations!(ax); hidespines!(ax); ax.aspect = DataAspect()\n\nf","category":"page"},{"location":"api/","page":"API Reference","title":"API Reference","text":"CurrentModule = EarthSciMLBase","category":"page"},{"location":"api/#API-Index","page":"API Reference","title":"API Index","text":"","category":"section"},{"location":"api/","page":"API Reference","title":"API Reference","text":"","category":"page"},{"location":"api/#API-Documentation","page":"API Reference","title":"API Documentation","text":"","category":"section"},{"location":"api/","page":"API Reference","title":"API Reference","text":"Modules = [EarthSciMLBase]","category":"page"},{"location":"api/#EarthSciMLBase.AbstractEarthSciMLSystem","page":"API Reference","title":"EarthSciMLBase.AbstractEarthSciMLSystem","text":"One or more ModelingToolkit systems of equations. EarthSciML uses custom types to allow  automatic composition of different systems together.\n\n\n\n\n\n","category":"type"},{"location":"api/#EarthSciMLBase.Advection","page":"API Reference","title":"EarthSciMLBase.Advection","text":"Apply advection to a model.\n\n\n\n\n\n","category":"type"},{"location":"api/#EarthSciMLBase.ComposedEarthSciMLSystem","page":"API Reference","title":"EarthSciMLBase.ComposedEarthSciMLSystem","text":"A system for composing together other systems using the + operator.\n\nThe easiest way to explain how this works and why we might want to do it is with an example. The following example is modified from the ModelingToolkit.jl documentation  here:\n\nExample:\n\nusing EarthSciMLBase\nusing ModelingToolkit\n\n# Set up our independent variable time, which will be shared by all systems.\n@parameters t\n\n# Create three systems which we will compose together.\nstruct SEqn <: EarthSciMLODESystem\n    sys::ODESystem\n\n    function SEqn(t) \n        @variables S(t), I(t), R(t)\n        D = Differential(t)\n        N = S + I + R\n        @parameters β\n        @named seqn = ODESystem([D(S) ~ -β*S*I/N])\n        new(seqn)\n    end\nend\n\nstruct IEqn <: EarthSciMLODESystem\n    sys::ODESystem\n\n    function IEqn(t) \n        @variables S(t), I(t), R(t)\n        D = Differential(t)\n        N = S + I + R\n        @parameters β,γ\n        @named ieqn = ODESystem([D(I) ~ β*S*I/N-γ*I])\n        new(ieqn)\n    end\nend\n\nstruct REqn <: EarthSciMLODESystem\n    sys::ODESystem\n\n    function REqn(t) \n        @variables I(t), R(t)\n        D = Differential(t)\n        @parameters γ\n        @named reqn = ODESystem([D(R) ~ γ*I])\n        new(reqn)\n    end\nend\n\n\n# Create functions to allow us to compose the systems together using the `+` operator.\nfunction Base.:(+)(s::SEqn, i::IEqn)::ComposedEarthSciMLSystem\n    seqn = s.sys\n    ieqn = i.sys\n    ComposedEarthSciMLSystem(\n        ConnectorSystem([\n            ieqn.S ~ seqn.S,\n            seqn.I ~ ieqn.I], s, i), \n        s, i,\n    )\nend\n\nfunction Base.:(+)(s::SEqn, r::REqn)::ComposedEarthSciMLSystem\n    seqn = s.sys\n    reqn = r.sys\n    ComposedEarthSciMLSystem(\n        ConnectorSystem([seqn.R ~ reqn.R], s, r), \n        s, r,\n    )\nend\n\nfunction Base.:(+)(i::IEqn, r::REqn)::ComposedEarthSciMLSystem\n    ieqn = i.sys\n    reqn = r.sys\n    ComposedEarthSciMLSystem(\n        ConnectorSystem([\n            ieqn.R ~ reqn.R,\n            reqn.I ~ ieqn.I], i, r), \n        i, r,\n    )\nend\n\n# Instantiate our three systems.\nseqn, ieqn, reqn = SEqn(t), IEqn(t), REqn(t)\n\n# Compose the systems together using the `+` operator. This is the fancy part!\nsir = seqn + ieqn + reqn\n\n# Finalize the system for solving.\nsirfinal = get_mtk(sir)\n\n# Show the equations in our combined system.\nequations(structural_simplify(sirfinal))\n\n# output\n3-element Vector{Equation}:\n Differential(t)(reqn₊R(t)) ~ reqn₊γ*reqn₊I(t)\n Differential(t)(seqn₊S(t)) ~ (-seqn₊β*seqn₊I(t)*seqn₊S(t)) / (seqn₊I(t) + seqn₊R(t) + seqn₊S(t))\n Differential(t)(ieqn₊I(t)) ~ (ieqn₊β*ieqn₊I(t)*ieqn₊S(t)) / (ieqn₊I(t) + ieqn₊R(t) + ieqn₊S(t)) - ieqn₊γ*ieqn₊I(t)\n\n\n\n\n\n","category":"type"},{"location":"api/#EarthSciMLBase.ConnectorSystem","page":"API Reference","title":"EarthSciMLBase.ConnectorSystem","text":"A connector for two systems.\n\neqs\nfrom\nto\n\n\n\n\n\n","category":"type"},{"location":"api/#EarthSciMLBase.ConstantWind","page":"API Reference","title":"EarthSciMLBase.ConstantWind","text":"Construct a constant wind velocity model component with the given wind speed(s), which should include units. For example, ConstantWind(t, 1u\"m/s\", 2u\"m/s\").\n\n\n\n\n\n","category":"type"},{"location":"api/#EarthSciMLBase.DomainInfo","page":"API Reference","title":"EarthSciMLBase.DomainInfo","text":"Domain information for a ModelingToolkit.jl PDESystem.  It can be used with the + operator to add initial and boundary conditions  and coordinate transforms to a ModelingToolkit.jl ODESystem or Catalyst.jl ReactionSystem.\n\nNOTE: The independent variable (usually time) must be first in the list of initial and boundary conditions.\n\npartial_derivative_func: Function that returns spatial derivatives of the partially-independent variables, optionally performing a coordinate transformation first.\nCurrent function options are:\npartialderivatives_identity (the default): Returns partial derivatives without performing any coordinate transforms.\npartialderivatives_lonlat2xymeters: Returns partial derivatives after transforming any variables named lat and lon\nfrom degrees to cartesian meters, assuming a spherical Earth.\n\nicbc: The sets of initial and/or boundary conditions.\n\n\n\n\n\n","category":"type"},{"location":"api/#EarthSciMLBase.EarthSciMLODESystem","page":"API Reference","title":"EarthSciMLBase.EarthSciMLODESystem","text":"A type for actual implementations of ODE systems.\n\n\n\n\n\n","category":"type"},{"location":"api/#EarthSciMLBase.ICBCcomponent","page":"API Reference","title":"EarthSciMLBase.ICBCcomponent","text":"Initial and boundary condition components that can be combined to  create an DomainInfo object.\n\n\n\n\n\n","category":"type"},{"location":"api/#EarthSciMLBase.MeanWind","page":"API Reference","title":"EarthSciMLBase.MeanWind","text":"A model component that represents the mean wind velocity, where t is the independent variable, iv is the independent variable, and pvars is the partial dependent variables for the domain.\n\n\n\n\n\n","category":"type"},{"location":"api/#EarthSciMLBase.constBC","page":"API Reference","title":"EarthSciMLBase.constBC","text":"Construct constant boundary conditions equal to the value  specified by val.\n\nval: The value of the constant boundary conditions.\npartialdomains: The partial domains, e.g. [x ∈ Interval(x_min, x_max), y ∈ Interval(y_min, y_max)].\n\n\n\n\n\n","category":"type"},{"location":"api/#EarthSciMLBase.constIC","page":"API Reference","title":"EarthSciMLBase.constIC","text":"Construct constant initial conditions equal to the value  specified by val.\n\nval: The value of the constant initial conditions.\nindepdomain: The independent domain, e.g. t ∈ Interval(t_min, t_max).\n\n\n\n\n\n","category":"type"},{"location":"api/#EarthSciMLBase.periodicBC","page":"API Reference","title":"EarthSciMLBase.periodicBC","text":"Construct periodic boundary conditions for the given partialdomains. Periodic boundary conditions are defined as when the value at one side of the domain is set equal to the value at the other side, so  that the domain \"wraps around\" from one side to the other.\n\npartialdomains: The partial domains, e.g. [x ∈ Interval(x_min, x_max), y ∈ Interval(y_min, y_max)].\n\n\n\n\n\n","category":"type"},{"location":"api/#EarthSciMLBase.zerogradBC","page":"API Reference","title":"EarthSciMLBase.zerogradBC","text":"Construct zero-gradient boundary conditions for the given partialdomains.\n\npartialdomains: The partial domains, e.g. [x ∈ Interval(x_min, x_max), y ∈ Interval(y_min, y_max)].\n\n\n\n\n\n","category":"type"},{"location":"api/#EarthSciMLBase.add_dims-Tuple{Any, AbstractVector, AbstractVector}","page":"API Reference","title":"EarthSciMLBase.add_dims","text":"add_dims(expression, vars, dims)\nadd_dims(equation, vars, dims)\n\nAdd the given dimensions to each variable in vars in the given expression or equation.  Each variable in vars must be unidimensional, i.e. defined like @variables u(t) rather than @variables u(..).\n\nExample:\n\nusing EarthSciMLBase, ModelingToolkit\n\n@parameters x y k t\n@variables u(t) q(t)\nexp = 2u + 3k*q + 1\nEarthSciMLBase.add_dims(exp, [u, q], [x, y, t])\n\n# output\n1 + 2u(x, y, t) + 3k*q(x, y, t)\n\n\n\n\n\n","category":"method"},{"location":"api/#EarthSciMLBase.add_metadata-Tuple{Any, Any}","page":"API Reference","title":"EarthSciMLBase.add_metadata","text":"Add the units and description in the variable from to the variable to.\n\n\n\n\n\n","category":"method"},{"location":"api/#EarthSciMLBase.dims-Tuple{EarthSciMLBase.ICcomponent}","page":"API Reference","title":"EarthSciMLBase.dims","text":"dims(\n    icbc::EarthSciMLBase.ICcomponent\n) -> Vector{Symbolics.Num}\n\n\nReturns the dimensions of the independent and partial domains associated with these  initial or boundary conditions.\n\n\n\n\n\n","category":"method"},{"location":"api/#EarthSciMLBase.domains-Tuple{EarthSciMLBase.ICcomponent}","page":"API Reference","title":"EarthSciMLBase.domains","text":"domains(icbc::EarthSciMLBase.ICcomponent) -> Vector\n\n\nReturns the domains associated with these initial or boundary conditions.\n\n\n\n\n\n","category":"method"},{"location":"api/#EarthSciMLBase.get_mtk-Tuple{AbstractEarthSciMLSystem}","page":"API Reference","title":"EarthSciMLBase.get_mtk","text":"\n\njulia get_mtk(     sys::AbstractEarthSciMLSystem ) -> ModelingToolkit.AbstractSystem\n\n\n\n\nReturn the ModelingToolkit version of this system.\n\n\n\n\n\n","category":"method"},{"location":"api/#EarthSciMLBase.graph-Tuple{ComposedEarthSciMLSystem}","page":"API Reference","title":"EarthSciMLBase.graph","text":"Create a graph from a ComposedEarthSciMLSystem using the MetaGraphsNext package.\n\n\n\n\n\n","category":"method"},{"location":"api/#EarthSciMLBase.icbc-Tuple{DomainInfo, AbstractVector}","page":"API Reference","title":"EarthSciMLBase.icbc","text":"icbc(di, states)\n\n\nReturn a vector of equations that define the initial and boundary conditions for the  given state variables.\n\n\n\n\n\n","category":"method"},{"location":"api/#EarthSciMLBase.ivar-Tuple{DomainInfo}","page":"API Reference","title":"EarthSciMLBase.ivar","text":"ivar(di::DomainInfo) -> Any\n\n\nReturn the independent variable associated with these  initial and boundary conditions.\n\n\n\n\n\n","category":"method"},{"location":"api/#EarthSciMLBase.operator_compose-Tuple{EarthSciMLODESystem, EarthSciMLODESystem}","page":"API Reference","title":"EarthSciMLBase.operator_compose","text":"operator_compose(\n    a::EarthSciMLODESystem,\n    b::EarthSciMLODESystem\n) -> ComposedEarthSciMLSystem\n\n\nCompose to systems of equations together by adding the right-hand side terms together of equations that have matching left-hand sides,  where the left-hand sides of both equations are derivatives of the same variable.\n\nThe example below shows that when we operator_compose two systems together that are both equal to D(x) = p, the resulting system is equal to D(x) = 2p.\n\nExample\n\nusing EarthSciMLBase\nusing ModelingToolkit\n\n@parameters t\n\nstruct ExampleSys <: EarthSciMLODESystem\n    sys::ODESystem\n\n    function ExampleSys(t; name)\n        @variables x(t)\n        @parameters p\n        D = Differential(t)\n        new(ODESystem([D(x) ~ p], t; name))\n    end\nend\n\n@named sys1 = ExampleSys(t)\n@named sys2 = ExampleSys(t)\n\ncombined = operator_compose(sys1, sys2)\n\ncombined_mtk = get_mtk(combined)\n\n# The simplified equation should be D(x) = p + sys2_xˍt, where sys2_xˍt is also equal to p.\nequations(structural_simplify(combined_mtk))\n\n# output\n1-element Vector{Equation}:\n Differential(t)(sys1₊x(t)) ~ sys1₊p + sys1₊sys2_xˍt(t)\n\n\n\n\n\n","category":"method"},{"location":"api/#EarthSciMLBase.param_to_var-Tuple{ModelingToolkit.AbstractSystem, Vararg{Symbol}}","page":"API Reference","title":"EarthSciMLBase.param_to_var","text":"Replace the parameter p in the system sys with a new variable that has  the same name, units, and description as p.\n\nThis can be useful to replace a parameter that does not change in time in a model component  with one specified by another system that does change in time (or space). For example, the  code below specifies a first-order loss equation, and then changes the temperature (which  determines the loss rate) with a temperature value that varies in time.\n\nusing ModelingToolkit, EarthSciMLBase, Unitful\n\n# Specify the original system with constant temperature.\nstruct Loss <: EarthSciMLODESystem\n    sys::ODESystem\n    Loss(sys::ModelingToolkit.ODESystem) = new(sys)\n    function Loss(t)\n        @variables A(t)=1 [unit=u\"kg\"]\n        @parameters k=1 [unit=u\"s^-1\"]\n        @parameters T=300 [unit=u\"K\"]\n        @constants T₀=300 [unit=u\"K\"]\n        eq = Differential(t)(A) ~ -k*exp(T/T₀) * A\n        new(ODESystem([eq]; name=:loss))\n    end\nend\n\n# Specify the temperature that varies in time.\nstruct Temperature <: EarthSciMLODESystem\n    sys::ODESystem\n    function Temperature(t)\n        @variables T(t)=300 [unit=u\"K\"]\n        @constants Tc=1.0 [unit=u\"K/s\"]\n        @constants tc=1.0 [unit=u\"s\"]\n        eq = Differential(t)(T) ~ sin(t/tc)*Tc\n        new(ODESystem([eq]; name=:temperature))\n    end\nend\n\n# Specify how to compose the two systems using `param_to_var`.\nfunction Base.:(+)(loss::Loss, temp::Temperature)\n    loss = Loss(param_to_var(loss.sys, :T))\n    losseqn = loss.sys\n    teqn = temp.sys\n    ComposedEarthSciMLSystem(\n        ConnectorSystem([losseqn.T ~ teqn.T], loss, temp), \n        loss, temp,\n    )\nend\n\n# Create the system components and the composed system.\n@variables t [unit=u\"s\", description=\"time\"]\nl = Loss(t)\nt = Temperature(t)\nvariable_loss = l+t\n\nequations(get_mtk(variable_loss))\n\n# output\n3-element Vector{Equation}:\n loss₊T(t) ~ temperature₊T(t)\n Differential(t)(loss₊A(t)) ~ -loss₊k*loss₊A(t)*exp(loss₊T(t) / loss₊T₀)\n Differential(t)(temperature₊T(t)) ~ temperature₊Tc*sin(t / temperature₊tc)\n\n\n\n\n\n","category":"method"},{"location":"api/#EarthSciMLBase.partialderivatives_identity-Tuple{AbstractVector}","page":"API Reference","title":"EarthSciMLBase.partialderivatives_identity","text":"partialderivatives_identity(pvars)\n\n\nReturn the partial derivative operators corresponding to each of the given partial-independent variables.\n\n\n\n\n\n","category":"method"},{"location":"api/#EarthSciMLBase.partialderivatives_lonlat2xymeters-Tuple{AbstractVector}","page":"API Reference","title":"EarthSciMLBase.partialderivatives_lonlat2xymeters","text":"partialderivatives_lonlat2xymeters(pvars; default_lat)\n\n\nReturn the partial derivative operators corresponding to each of the given partial-independent variables after converting variables named lon and lat from degrees to x and y meters,  assuming they represent longitude and latitude on a spherical Earth.\n\n\n\n\n\n","category":"method"},{"location":"api/#EarthSciMLBase.pvars-Tuple{DomainInfo}","page":"API Reference","title":"EarthSciMLBase.pvars","text":"pvars(di::DomainInfo) -> Any\n\n\nReturn the partial independent variables associated with these  initial and boundary conditions.\n\n\n\n\n\n","category":"method"},{"location":"example_icbc/","page":"Initial and Boundary Conditions","title":"Initial and Boundary Conditions","text":"CurrentModule = EarthSciMLBase","category":"page"},{"location":"example_icbc/#Initial-and-Boundary-condition-example","page":"Initial and Boundary Conditions","title":"Initial and Boundary condition example","text":"","category":"section"},{"location":"example_icbc/","page":"Initial and Boundary Conditions","title":"Initial and Boundary Conditions","text":"using EarthSciMLBase\nusing ModelingToolkit, DomainSets\n\n# Set up ODE system\n@parameters x y t\n@variables u(t) v(t)\nDt = Differential(t)\n\nx_min = y_min = t_min = 0.0\nx_max = y_max = 1.0\nt_max = 11.5\n\neqs = [\n    Dt(u) ~ √abs(v),\n    Dt(v) ~ √abs(u),\n]\n\n@named sys = ODESystem(eqs)\n\n# Create constant initial conditions = 16.0 and boundary conditions = 4.0.\nicbc = DomainInfo(\n    constIC(4.0, t ∈ Interval(t_min, t_max)),\n    constBC(16.0, \n        x ∈ Interval(x_min, x_max),\n        y ∈ Interval(y_min, y_max),\n    ),\n)\n\n# Convert to PDESystem and add constant initial and boundary conditions.\npdesys = sys + icbc\n\npdesys.bcs","category":"page"},{"location":"example_all_together/","page":"All together","title":"All together","text":"CurrentModule = EarthSciMLBase","category":"page"},{"location":"example_all_together/#Example-using-different-components-of-EarthSciMLBase-together","page":"All together","title":"Example using different components of EarthSciMLBase together","text":"","category":"section"},{"location":"example_all_together/","page":"All together","title":"All together","text":"using EarthSciMLBase\nusing ModelingToolkit, Catalyst, DomainSets, MethodOfLines, DifferentialEquations\nusing Plots\n\n# Create our independent variable `t` and our partially-independent variables `x` and `y`.\n@parameters t x y\n\n# Create our ODE systems of equations as subtypes of `EarthSciMLODESystem`.\n# Creating our system in this way allows us to convert it to a PDE system \n# using just the `+` operator as shown below.\n\n# Our first example system is a simple reaction system.\nstruct ExampleSys1 <: EarthSciMLODESystem\n    sys\n    function ExampleSys1(t; name)\n        @species c₁(t)=5.0 c₂(t)=5.0\n        new(convert(ODESystem, ReactionSystem(\n            [Reaction(2.0, [c₁], [c₂])],\n            t; name=name,\n        ), combinatoric_ratelaws=false))\n    end\nend\n\n# Our second example system is a simple ODE system,\n# with the same two variables.\nstruct ExampleSys2 <: EarthSciMLODESystem\n    sys\n    function ExampleSys2(t; name)\n        @variables c₁(t)=5.0 c₂(t)=5.0\n        @parameters p₁=1.0 p₂=0.5\n        D = Differential(t)\n        new(ODESystem(\n            [D(c₁) ~ -p₁, D(c₂) ~ p₂],\n            t; name=name,\n        ))\n    end\nend\n\n# Specify what should happen when we couple the two systems together.\n# In this case, we want the the derivative of the composed system to \n# be equal to the sum of the derivatives of the two systems.\n# We can do that using the `operator_compose` function \n# from this package.\nfunction Base.:(+)(sys1::ExampleSys1, sys2::ExampleSys2)\n    operator_compose(sys1, sys2)\nend\n\n# Once we specify all of the above, it is simple to create our \n# two individual systems and then couple them together. \n@named sys1 = ExampleSys1(t)\n@named sys2 = ExampleSys2(t)\nsys = sys1 + sys2\n\n# At this point we have an ODE system that is composed of two other ODE systems.\n# We can inspect its equations and observed variables using the `equations` and `observed` functions.\nsimplified_sys = structural_simplify(get_mtk(sys))\nequations(simplified_sys)\nobserved(simplified_sys)\n\n# We can also run simulations using this system:\nodeprob = ODEProblem(simplified_sys, [], (0.0,10.0), [])\nodesol = solve(odeprob)\nplot(odesol)\n\n# Once we've confirmed that our model works in a 0D \"box model\" setting,\n# we can expand it to 1, 2, or 3 dimensions using by adding in initial \n# and boundary conditions.\n# We will also add in advection using constant-velocity wind fields\n# add the same time.\nx_min = y_min = t_min = 0.0\nx_max = y_max = t_max = 1.0\ndomain = DomainInfo(\n    constIC(4.0, t ∈ Interval(t_min, t_max)),\n    periodicBC(x ∈ Interval(x_min, x_max)),\n    zerogradBC(y ∈ Interval(y_min, y_max)),\n)\n\nsys_pde = sys + domain + ConstantWind(t, 1.0, 1.0) + Advection()\n\n# Now we can inspect this new system that we've created:\nsys_pde_mtk = get_mtk(sys_pde)\nequations(sys_pde_mtk)\nsys_pde_mtk.dvs\n\n# Finally, we can run a simulation using this system:\ndiscretization = MOLFiniteDifference([x=>10, y=>10], t, approx_order=2)\n@time pdeprob = discretize(sys_pde_mtk, discretization)\n@time pdesol = solve(pdeprob, Tsit5(), saveat=0.1)\n\n# Plot the solution.\ndiscrete_x, discrete_y, discrete_t = pdesol[x], pdesol[y], pdesol[t]\n@variables sys1₊c₁(..) sys1₊c₂(..)\nsolc1, solc2 = pdesol[sys1₊c₁(t, x, y)], pdesol[sys1₊c₂(t, x, y)]\nanim = @animate for k in 1:length(discrete_t)\n    p1 = heatmap(solc1[k, 1:end-1, 1:end-1], title=\"c₁ t=\\$(discrete_t[k])\", clim=(0,4.0), lab=:none)\n    p2 = heatmap(solc2[k, 1:end-1, 1:end-1], title=\"c₂ t=\\$(discrete_t[k])\", clim=(0,7.0), lab=:none)\n    plot(p1, p2, layout=(1,2), size=(800,400))\nend\ngif(anim, fps = 8)","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = EarthSciMLBase","category":"page"},{"location":"#EarthSciMLBase:-Utilities-for-Symbolic-Earth-Science-Modeling-and-Machine-Learning","page":"Home","title":"EarthSciMLBase: Utilities for Symbolic Earth Science Modeling and Machine Learning","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for EarthSciMLBase.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This package contains utilities for constructing Earth Science models in Julia using ModelingToolkit.jl.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"using Pkg\nPkg.add(\"EarthSciMLBase\")","category":"page"},{"location":"#Feature-Summary","page":"Home","title":"Feature Summary","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package contains types and functions designed to simplify the process of constructing and composing symbolically-defined Earth Science model components together.","category":"page"},{"location":"#Feature-List","page":"Home","title":"Feature List","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Abstract types (based on AbstractEarthSciMLSystem) for wrapping ModelingToolkit.jl equation systems\nOperations to compose AbstractEarthSciMLSystems together using the + operator.\nOperations to add intitial and boundary conditions to systems and to turn ODE systems into PDE systems.\nOperations to add Advection terms to systems.","category":"page"},{"location":"#Contributing","page":"Home","title":"Contributing","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Please refer to the SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages for guidance on PRs, issues, and other matters relating to contributing.","category":"page"},{"location":"#Reproducibility","page":"Home","title":"Reproducibility","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg # hide\nPkg.status() # hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"</details>","category":"page"},{"location":"","page":"Home","title":"Home","text":"<details><summary>and using this machine and Julia version.</summary>","category":"page"},{"location":"","page":"Home","title":"Home","text":"using InteractiveUtils # hide\nversioninfo() # hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"</details>","category":"page"},{"location":"","page":"Home","title":"Home","text":"<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg # hide\nPkg.status(;mode = PKGMODE_MANIFEST) # hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"</details>","category":"page"},{"location":"","page":"Home","title":"Home","text":"You can also download the \n<a href=\"","category":"page"},{"location":"","page":"Home","title":"Home","text":"using TOML\nusing Markdown\nversion = TOML.parse(read(\"../../Project.toml\",String))[\"version\"]\nname = TOML.parse(read(\"../../Project.toml\",String))[\"name\"]\nlink = Markdown.MD(\"https://github.com/EarthSciML/\"*name*\".jl/tree/gh-pages/v\"*version*\"/assets/Manifest.toml\")","category":"page"},{"location":"","page":"Home","title":"Home","text":"\">manifest</a> file and the\n<a href=\"","category":"page"},{"location":"","page":"Home","title":"Home","text":"using TOML\nusing Markdown\nversion = TOML.parse(read(\"../../Project.toml\",String))[\"version\"]\nname = TOML.parse(read(\"../../Project.toml\",String))[\"name\"]\nlink = Markdown.MD(\"https://github.com/EarthSciML/\"*name*\".jl/tree/gh-pages/v\"*version*\"/assets/Project.toml\")","category":"page"},{"location":"","page":"Home","title":"Home","text":"\">project</a> file.","category":"page"},{"location":"example_advection/","page":"Advection","title":"Advection","text":"CurrentModule = EarthSciMLBase","category":"page"},{"location":"example_advection/#Advection-Example","page":"Advection","title":"Advection Example","text":"","category":"section"},{"location":"example_advection/","page":"Advection","title":"Advection","text":"using EarthSciMLBase\nusing DomainSets, MethodOfLines, ModelingToolkit, Plots, DifferentialEquations\n\n# Create our independent variable `t` and our partially-independent variable `x`.\n@parameters t, x\n\n# Create our ODE system of equations as a subtype of `EarthSciMLODESystem`.\n# Creating our system in this way allows us to convert it to a PDE system \n# using just the `+` operator as shown below.\nstruct ExampleSys <: EarthSciMLODESystem\n    sys::ODESystem\n    function ExampleSys(t; name)\n        @variables y(t)\n        @parameters p=2.0\n        D = Differential(t)\n        new(ODESystem([D(y) ~ p], t; name))\n    end\nend\n@named sys = ExampleSys(t)\n\n# Create our initial and boundary conditions.\ndomain = DomainInfo(constIC(0.0, t ∈ Interval(0, 1.0)), constBC(1.0, x ∈ Interval(0, 1.0)))\n\n# Convert our ODE system to a PDE system and add advection to each of the state variables.\n# We're also adding a constant wind in the x-direction, with a speed of 1.0.\nsys_advection = sys + domain + ConstantWind(t, 1.0) + Advection()\nsys_mtk = get_mtk(sys_advection)\n\n# Discretize the system and solve it.\ndiscretization = MOLFiniteDifference([x=>10], t, approx_order=2)\n@time prob = discretize(sys_mtk, discretization)\n@time sol = solve(prob, Tsit5(), saveat=0.1)\n\n# Plot the solution.\ndiscrete_x = sol[x]\ndiscrete_t = sol[t]\n@variables sys₊y(..)\nsoly = sol[sys₊y(t, x)]\nanim = @animate for k in 1:length(discrete_t)\n    plot(soly[k, 1:end], title=\"t=\\$(discrete_t[k])\", ylim=(0,2.5), lab=:none)\nend\ngif(anim, fps = 8)","category":"page"}]
}
