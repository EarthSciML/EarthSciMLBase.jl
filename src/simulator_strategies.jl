export SimulatorIMEX, run!

"""
SimulatorStrategy is an abstract type that defines the strategy for running a simulation.
Each SimulatorStrategy should implement a method of:

```julia
run!(st::SimulatorStrategy, s::Simulator{T}) where T
```
"""
abstract type SimulatorStrategy end

"""
$(TYPEDSIGNATURES)

Run the simulation.
"""
function run!(st::SimulatorStrategy, simulator)
    error("Not implemented.")
end

"Return a SciMLOperator to apply the MTK system to each column of s.u after reshaping to a matrix."
function mtk_op(s::Simulator)
    mtkf = ODEFunction(s.sys_mtk)
    II = CartesianIndices(size(s.u)[2:4])
    function setp!(p, j) # Set the parameters for the jth grid cell.
        ii = II[j]
        for (jj, g) ∈ enumerate(s.grid) # Set the coordinates of this grid cell.
            p[s.pvidx[jj]] = g[ii[jj]]
        end
    end
    function f(du, u::AbstractMatrix, p, t) # In-place, matrix
        @inbounds for j ∈ 1:size(u, 2)
            col = view(u, :, j)
            ddu = view(du, :, j)
            setp!(p, j)
            @inline mtkf(ddu, col, p, t)
        end
    end
    function f(u::AbstractMatrix, p, t) # Out-of-place, matrix
        function ff(u, p, t, j)
            setp!(p, j)
            mtkf(u, p, t)
        end
        @inbounds @views mapreduce(jcol -> ff(jcol[2], p, t, jcol[1]), hcat, enumerate(eachcol(u)))
    end
    
    indata = reshape(s.u, size(s.u, 1), :)
    fo = FunctionOperator(f, indata, batch=true, p=s.p)

    ncols = size(indata, 2)
    # Rehape the input vector to a matrix, then apply the FunctionOperator.
    #op = ScalarOperator(1.0) * TensorProductOperator(I(ncols), fo)
    op = TensorProductOperator(I(ncols), fo)
    cache_operator(op, s.u[:])
end

function mtk_func(s::Simulator)
    b = repeat([length(states(s.sys_mtk))], length(s.u) ÷ size(s.u, 1))
    j = BlockBandedMatrix{Float64}(undef, b, b, (0,0)) # Jacobian prototype
    ODEFunction(mtk_op(s); jac_prototype=j)
end

"""
A simulator strategy based on implicit-explicit (IMEX) time integration.
See [here](https://docs.sciml.ai/DiffEqDocs/stable/types/split_ode_types/)
for additional information.

`alg` is the ODE solver algorithm to use, which should be chosen
from the Split ODE Solver algorithms listed [here](https://docs.sciml.ai/DiffEqDocs/stable/solvers/split_ode_solve/#split_ode_solve).
In most cases, it is recommended to use a Jacobian-free Newton-Krylov linear solution method
as described [here](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/advanced_ode_example/#Using-Jacobian-Free-Newton-Krylov).

!!! caution
    This strategy does not currently work. Do not use. See [here](https://github.com/SciML/OrdinaryDiffEq.jl/issues/2078) for more details.

$(FIELDS)
"""
struct SimulatorIMEX <: SimulatorStrategy 
    alg
end


"""
$(TYPEDSIGNATURES)

Run the simulation.
`kwargs` are passed to the ODEProblem and ODE solver constructors.
"""
function run!(s::Simulator, st::SimulatorIMEX; kwargs...)
    f1 = mtk_func(s)

    @assert length(s.sys.ops) > 0 "Operators must be defined to use the `SimulatorIMEX` strategy. For no operators, try `SimulatorFused` instead."
    # Combine the non-stiff operators into a single operator.
    # This works because SciMLOperators can be added together.
    f2 = sum([get_scimlop(op, s) for op ∈ s.sys.ops])

    start, finish = time_range(s.domaininfo)
    prob = SplitODEProblem(f1, f2, s.u, (start, finish), s.p, callback=CallbackSet(s.sys.callbacks...), kwargs...)
    solve(prob, st.alg, save_on=false, save_start=false, save_end=false,
        initialize_save=false; kwargs...)
end