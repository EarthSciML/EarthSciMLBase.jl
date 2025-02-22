function _add_coord_args(ex, coord_args)
    ex = MacroTools.postwalk(x -> @capture(x, function (args__)
            body_
        end) ?
                                  :(function ($(args...), $(coord_args...))
            $body
        end) : x, ex)

    ex = string(ex)
    for ca in coord_args
        ex = replace(ex, "$ca = NaN" => "")
    end
    Meta.parse(ex)
end

function _get_coord_args(sys, domain)
    coords = EarthSciMLBase.coord_params(sys, domain)
    # Create constants to replace coordinates. We will replace these with arguments later.
    coord_args = Symbol.(nameof.(coords), (:_arg,))
    coords, coord_args
end

function _prepare_coord_sys(sys, domain)
    coords, coord_args = _get_coord_args(sys, domain)
    coord_arg_consts = [only(@constants $(ca) = NaN) for ca in coord_args]
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
RuntimeGeneratedFunctions.init(ModelingToolkit)

# Build a function that takes the coordinates as arguments.
function _build_mtk_coord_arg_function(sys, coord_args, gen_f, iip; kwargs...)
    f_expr = gen_f(sys; kwargs...)
    idx = iip ? 2 : 1
    ex = _add_coord_args(f_expr[idx], coord_args)
    f = @RuntimeGeneratedFunction(ModelingToolkit, ex)
    return f
end

"""
Create a function to return the observed function for a system with coordinates.
For more information see the documentation for `ModelingToolkit.build_explicit_observed_function`.
"""
function build_coord_observed_function(sys_coord, coord_args, vars, iip; kwargs...)
    idx = iip ? 2 : 1
    ex = ModelingToolkit.build_explicit_observed_function(sys_coord, vars; return_inplace=true, kwargs...)[idx]
    ex = RuntimeGeneratedFunctions.get_expression(ex)

    ex = MacroTools.postwalk(x -> @capture(ex, (args__,) -> body_) ? :(($(args...), $(coord_args...)) -> $body) : x, ex)

    ex = string(ex)
    for ca in coord_args
        ex = replace(ex, "$ca = NaN" => "")
    end
    ex = Meta.parse(ex)
    @RuntimeGeneratedFunction(ModelingToolkit, ex)
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
function mtk_grid_func(sys_mtk::ODESystem, domain::DomainInfo{T}, u0, iip;
    sparse=false, tgrad=false, vjp=true) where {T}

    sys_mtk, coord_args = _prepare_coord_sys(sys_mtk, domain)

    mtkf_coord = _build_mtk_coord_arg_function(sys_mtk, coord_args, ModelingToolkit.generate_function, iip)
    jac_coord = _build_mtk_coord_arg_function(sys_mtk, coord_args, ModelingToolkit.generate_jacobian, iip; sparse=sparse)

    f = _mtk_grid_func(sys_mtk, mtkf_coord, domain)

    ncells = reduce(*, length.(grid(domain)))
    nvars = length(unknowns(sys_mtk))
    single_jac_prototype = ODEFunction(sys_mtk, tgrad=tgrad, jac=true, sparse=sparse).jac_prototype
    if isnothing(single_jac_prototype)
        single_jac_prototype = Matrix{eltype(u0)}(undef, nvars, nvars)
    end
    jac_prototype = BlockDiagonal([similar(single_jac_prototype) for _ in 1:ncells])
    jf = mtk_jac_grid_func(sys_mtk, jac_coord, domain)

    kwargs = []
    if tgrad
        tgf = _build_mtk_coord_arg_function(sys_mtk, coord_args, ModelingToolkit.generate_tgrad, iip)
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
