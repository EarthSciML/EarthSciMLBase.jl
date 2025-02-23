# Move coordinate variables from the parameter list of a function to the argument list.
# This is useful for creating functions that take coordinates as arguments.
function rewrite_coord_func(x, coord_args)
    if @capture(x, function (args__)
        body_
    end)
        return :(function ($(args...), $(coord_args...))
            $body
        end)
    elseif @capture(x, a_ = b_)
        if (a in coord_args) && (b == 1)
            return nothing
        end
    end
    return x
end

function _add_coord_args(ex, coord_args)
    ex = MacroTools.postwalk(x -> rewrite_coord_func(x, coord_args), ex)
end

function gen_coord_func(sys, expr, coord_args; eval_expression=false, eval_module=@__MODULE__)
    fexpr = ModelingToolkit.generate_custom_function(sys, expr, expression=Val{true})
    if fexpr isa Tuple
        fexpr = EarthSciMLBase._add_coord_args.(fexpr, (coord_args,))
        f = ModelingToolkit.eval_or_rgf.(fexpr; eval_expression, eval_module)
        f = ModelingToolkit.GeneratedFunctionWrapper{(2, 6, true)}(f[1], f[2])
    else
        fexpr = EarthSciMLBase._add_coord_args(fexpr, coord_args)
        f = ModelingToolkit.eval_or_rgf(fexpr; eval_expression, eval_module)
    end
    return f
end

function _get_coord_args(sys, domain)
    coords = EarthSciMLBase.coord_params(sys, domain)
    # Create constants to replace coordinates. We will replace these with arguments later.
    coord_args = Symbol.(nameof.(coords), (:_arg,))
    coords, coord_args
end

function _prepare_coord_sys(sys, domain)
    coords, coord_args = _get_coord_args(sys, domain)
    coord_arg_consts = [only(@constants $(ca) = 1) for ca in coord_args]
    coord_arg_consts = add_metadata.(coord_arg_consts, coords; exclude_default=true)
    sys_coord = substitute(sys, Dict(coords .=> coord_arg_consts))
    @named obs = ODESystem(substitute(ModelingToolkit.observed(sys),
            Dict(coords .=> coord_arg_consts)),
        ModelingToolkit.get_iv(sys_coord))
    sys_coord = copy_with_change(sys_coord, eqs=[equations(sys_coord); equations(obs)],
        unknowns=unique([unknowns(sys_coord); unknowns(obs)]),
        parameters=unique([parameters(sys_coord); parameters(obs)]))
    return structural_simplify(sys_coord), coord_args
end

RuntimeGeneratedFunctions.init(@__MODULE__)

function build_coord_ode_function(sys_coord, coord_args; kwargs...)
    exprs = [eq.rhs for eq in equations(sys_coord)]
    gen_coord_func(sys_coord, exprs, coord_args; kwargs...)
end

function build_coord_jac_function(sys_coord, coord_args; sparse=false, kwargs...)
    jac_expr = ModelingToolkit.calculate_jacobian(sys_coord, sparse=sparse; kwargs...)
    gen_coord_func(sys_coord, jac_expr, coord_args; kwargs...)
end

function build_coord_tgrad_function(sys_coord, coord_args; kwargs...)
    tgrad_expr = ModelingToolkit.calculate_tgrad(sys_coord; kwargs...)
    gen_coord_func(sys_coord, tgrad_expr, coord_args; kwargs...)
end

"""
Create a function to return the observed function for a system with coordinates.
For more information see the documentation for `ModelingToolkit.build_explicit_observed_function`.
"""
function build_coord_observed_function(sys_coord, coord_args, vars; kwargs...)
    o = ModelingToolkit.observed(sys_coord)
    idxs = [findfirst((x) -> Symbol(x.lhs) == Symbol(v), o) for v in vars]
    exprs = [o[i].rhs for i in idxs]
    gen_coord_func(sys_coord, exprs, coord_args; kwargs...)
end

function _mtk_grid_func(sys_mtk, mtkf, domain)
    nrows = length(unknowns(sys_mtk))
    II = CartesianIndices(tuple(size(domain)...))
    c1, c2, c3 = grid(domain)
    function f(du::AbstractVector, u::AbstractVector, p, t) # In-place
        u = reshape(u, nrows, :)
        du = reshape(du, nrows, :)
        for j ∈ 1:size(u, 2)
            col = view(u, :, j)
            ddu = view(du, :, j)
            mtkf(ddu, col, p, t, c1[II[j][1]], c2[II[j][2]], c3[II[j][3]])
        end
        nothing
    end
    function f(u, p, t) # Out-of-place
        u = reshape(u, nrows, :)
        function ff(u, p, t, j)
            mtkf(u, p, t, c1[II[j][1]], c2[II[j][2]], c3[II[j][3]])
        end
        du = @views mapreduce(jcol -> ff(jcol[2], p, t, jcol[1]), hcat, enumerate(eachcol(u)))
        reshape(du, :)
    end
    return f
end

# Return a function to apply the MTK system to each column of u after reshaping to a matrix.
function mtk_grid_func(sys_mtk::ODESystem, domain::DomainInfo{T}, u0;
    sparse=false, tgrad=false, vjp=true) where {T}

    sys_mtk, coord_args = _prepare_coord_sys(sys_mtk, domain)

    mtkf_coord = build_coord_ode_function(sys_mtk, coord_args)
    jac_coord = build_coord_jac_function(sys_mtk, coord_args; sparse=sparse)

    f = _mtk_grid_func(sys_mtk, mtkf_coord, domain)

    ncells = reduce(*, length.(grid(domain)))
    nvars = length(unknowns(sys_mtk))

    if !sparse
        single_jac_prototype = Matrix{eltype(u0)}(undef, nvars, nvars)
    else
        single_jac_prototype = ODEFunction(sys_mtk, tgrad=tgrad, jac=true, sparse=sparse).jac_prototype
    end
    jac_prototype = BlockDiagonal([similar(single_jac_prototype) for _ in 1:ncells])
    jf = mtk_jac_grid_func(sys_mtk, jac_coord, domain)

    kwargs = []
    if tgrad
        tgf = build_coord_tgrad_function(sys_mtk, coord_args)
        tg = mtk_tgrad_grid_func(sys_mtk, tgf, domain)
        push!(kwargs, :tgrad => tg)
    end
    if vjp
        vj = mtk_vjp_grid_func(sys_mtk, jac_coord, domain)
        push!(kwargs, :vjp => vj)
    end
    ODEFunction(f; jac_prototype=jac_prototype, jac=jf, kwargs...), sys_mtk, coord_args
end

# Create a function to calculate the gridded Jacobian.
# ngrid is the number of grid cells.
function mtk_jac_grid_func(sys_mtk, jacf, domain)
    nvar = length(unknowns(sys_mtk))
    II = CartesianIndices(tuple(size(domain)...))
    c1, c2, c3 = grid(domain)
    function jac(out, u, p, t) # In-place
        u = reshape(u, nvar, :)
        blks = blocks(out)
        for r ∈ 1:size(u, 2)
            _u = view(u, :, r)
            jacf(blks[r], _u, p, t, c1[II[r][1]], c2[II[r][2]], c3[II[r][3]])
        end
        nothing
    end
    function jac(u, p, t) # Out-of-place
        u = reshape(u, nvar, :)
        BlockDiagonal([
            begin
                _u = view(u, :, r)
                jacf(_u, p, t, c1[II[r][1]], c2[II[r][2]], c3[II[r][3]])
            end for r ∈ 1:size(u, 2)
        ])
    end
end

# Create a function to calculate the gridded time gradient.
# ngrid is the number of grid cells.
function mtk_tgrad_grid_func(sys_mtk, tgradf, domain)
    nvar = length(unknowns(sys_mtk))
    II = CartesianIndices(tuple(size(domain)...))
    c1, c2, c3 = grid(domain)
    function tgrad(out, u, p, t) # In-place
        u = reshape(u, nvar, :)
        # FIXME(CT): I believe the tgrad output should be a vector and not a clone of the Jacobian.
        # This may be a bug in the DifferentialEquations.jl interface.
        blks = blocks(out)
        for r ∈ 1:size(u, 2)
            _u = view(u, :, r)
            tgradf(blks[r], _u, p, t, c1[II[r][1]], c2[II[r][2]], c3[II[r][3]])
        end
    end
end

function mtk_vjp_grid_func(sys_mtk, mtkf, domain)
    nvar = length(unknowns(sys_mtk))
    II = CartesianIndices(tuple(size(domain)...))
    c1, c2, c3 = grid(domain)
    function vjp(vJ, v, u, p, t)
        u = reshape(u, nvar, :)
        J = Matrix{eltype(u)}(undef, nvar, nvar)
        for r ∈ 1:size(u, 2)
            _u = view(u, :, r)
            vJr = view(vJ, (r-1)*nvar+1:r*nvar, :)
            vr = view(v, (r-1)*nvar+1:r*nvar, :)
            mtkf.jac(J, _u, p, t, c1[II[j][1]], c2[II[j][2]], c3[II[j][3]])
            mul!(vJr, J', vr)
        end
    end
    function vjp(v, u, p, t)
        u = reshape(u, nvar, :)
        vcat([
            begin
                _u = view(u, :, r)
                vr = view(v, (r-1)*nvar+1:r*nvar, :)
                J = mtkf.jac(_u, p, t, c1[II[j][1]], c2[II[j][2]], c3[II[j][3]])
                J' * vr
            end for r ∈ 1:size(u, 2)
        ]...)
    end
    return vjp
end
