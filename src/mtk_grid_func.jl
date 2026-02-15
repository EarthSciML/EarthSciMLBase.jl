# Move coordinate variables from the parameter list of a function to the argument list.
# This is useful for creating functions that take coordinates as arguments.
function rewrite_coord_func(x, coord_args, idv::Symbol)
    if @capture(x, function (args__)
        body_
    end)
        return :(function ($(args...), $(coord_args...))
            $body
        end)
    elseif @capture(x, (a_)(b_))
        if (a isa _CoordTmpF) && (b == idv)
            return :($(coord_args[a.idx]))
        end
    end
    return x
end

function rewrite_broadcast(x)
    if @capture(x, (a_)[b_] = (c_))
        if a == :ˍ₋out # Copying data to output array
            return :(@view($a[$b, :]) .= $c)
        end
    elseif @capture(x, (a_)[b_])
        if a == :__mtk_arg_1 # Accessing u array
            return :(@view($a[$b, :]))
        end
    elseif @capture(x, (f_)(args__)) # Function call
        return :($f.($(args...)))
    elseif @capture(x, if a_
        b_
    else
        c_
    end) # if expression
        return :(ifelse.($a, $b, $c))
    end
    return x
end

function _add_coord_args(ex, coord_args, idv::Symbol, ::MapAlgorithm)
    ex = MacroTools.postwalk(x -> rewrite_coord_func(x, coord_args, idv), ex)
end

function _add_coord_args(ex, coord_args, idv::Symbol, ::MapReactant)
    ex = MacroTools.postwalk(x -> rewrite_coord_func(x, coord_args, idv), ex)
    ex = MacroTools.postwalk(x -> rewrite_broadcast(x), ex)
end

function gen_coord_func(sys, expr, coord_args, alg::MapAlgorithm = MapBroadcast();
        eval_expression = false, eval_module = @__MODULE__)
    idv = var2symbol(ModelingToolkit.get_iv(sys))
    fexpr = ModelingToolkit.generate_custom_function(sys, expr, expression = Val{true})
    if fexpr isa Tuple
        fexpr = _add_coord_args.(fexpr, (coord_args,), (idv,), (alg,))
        f = ModelingToolkit.eval_or_rgf.(fexpr; eval_expression, eval_module)
        f = ModelingToolkit.GeneratedFunctionWrapper{(2, 6, true)}(f[1], f[2])
    else
        fexpr = _add_coord_args(fexpr, coord_args, idv, alg)
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

# Dummy function for temporarily replacing coordinate variables.
struct _CoordTmpF
    coord::Any
    idx::Int
end
(::_CoordTmpF)(t) = Inf
(x::_CoordTmpF)(u::DynamicQuantities.AbstractQuantity) = ModelingToolkit.get_unit(x.coord)
@register_symbolic (coordtmpf::_CoordTmpF)(t)
Base.nameof(x::_CoordTmpF) = Symbol(:coord, x.idx)
Symbolics.derivative(::_CoordTmpF, args::NTuple{1, Any}, ::Val{1}) = 0.0

function _prepare_coord_sys(sys, domain)
    coords, coord_args = _get_coord_args(sys, domain)
    t = ModelingToolkit.get_iv(sys)
    coord_tmps = [_CoordTmpF(coords[i], i)(t) for i in eachindex(coords)]
    sys_coord = substitute(sys, Dict(coords .=> coord_tmps))
    @named obs = System(
        Vector{ModelingToolkit.Equation}(substitute(ModelingToolkit.observed(sys),
            Dict(coords .=> coord_tmps))),
        ModelingToolkit.get_iv(sys_coord))
    sys_coord = copy_with_change(sys_coord,
        eqs = [equations(sys_coord); equations(obs)],
        discrete_events = ModelingToolkit.get_discrete_events(sys),
        continuous_events = ModelingToolkit.get_continuous_events(sys)
    )
    return mtkcompile(sys_coord), coord_args
end

RuntimeGeneratedFunctions.init(@__MODULE__)

function build_coord_ode_function(sys_coord, coord_args, MA::MapAlgorithm = MapBroadcast();
        kwargs...)
    exprs = [eq.rhs for eq in equations(sys_coord)]
    gen_coord_func(sys_coord, exprs, coord_args, MA; kwargs...)
end

function build_coord_jac_function(sys_coord, coord_args, MA::MapAlgorithm = MapBroadcast();
        sparse = false, kwargs...)
    jac_expr = ModelingToolkit.calculate_jacobian(sys_coord, sparse = sparse; kwargs...)
    gen_coord_func(sys_coord, jac_expr, coord_args, MA; kwargs...)
end

function build_coord_tgrad_function(
        sys_coord, coord_args, MA::MapAlgorithm = MapBroadcast();
        kwargs...)
    tgrad_expr = ModelingToolkit.calculate_tgrad(sys_coord; kwargs...)
    # Substitute time derivatives of _CoordTmpF terms with 0.0.
    # _CoordTmpF represents coordinate parameters that are constant in time,
    # but expand_derivatives doesn't resolve their derivatives in Symbolics v7.
    subs = Dict()
    for e in tgrad_expr
        for v in Symbolics.get_variables(e)
            if Symbolics.iscall(v) && Symbolics.operation(v) isa Symbolics.Differential
                inner = Symbolics.arguments(v)[1]
                if Symbolics.iscall(inner) && Symbolics.operation(inner) isa _CoordTmpF
                    subs[Symbolics.wrap(v)] = 0.0
                end
            end
        end
    end
    if !isempty(subs)
        tgrad_expr = Symbolics.substitute.(tgrad_expr, (subs,))
    end
    gen_coord_func(sys_coord, tgrad_expr, coord_args, MA; kwargs...)
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

function _mtk_grid_func(sys_mtk, mtkf, domain::DomainInfo{ET, AT},
        alg::MA) where {ET, AT, MA <: MapAlgorithm}
    nrows = length(unknowns(sys_mtk))
    c1, c2, c3 = concrete_grid(domain)
    function f(du::AbstractVector, u::AbstractVector, p, t) # In-place
        u = reshape(u, nrows, :)
        du = reshape(du, nrows, :)
        function f(j, du, u, p, t, c1, c2, c3)
            mtkf(view(du, :, j), view(u, :, j), p, t, c1[j], c2[j], c3[j])
        end
        map_closure_to_range(f, 1:size(u, 2), alg, du, u, p, t, c1, c2, c3)
        nothing
    end
    function f(u, p, t) # Out-of-place
        u = reshape(u, nrows, :)
        du = Vector{Vector{eltype(u)}}(undef, size(u, 2))
        f(j, u, p, t, c1, c2, c3) = du[j] = mtkf(view(u, :, j), p, t, c1[j], c2[j], c3[j])
        map_closure_to_range(f, 1:size(u, 2), alg, u, p, t, c1, c2, c3)
        reshape(hcat(du...), :)
    end
    return f
end

# Broadcast-based ODE function for use with Reactant.jl.
function _mtk_grid_func(sys_mtk, mtkf, domain::DomainInfo{ET, AT},
        ::MapReactant) where {ET, AT}
    nrows = length(unknowns(sys_mtk))
    c1, c2, c3 = concrete_grid(domain)
    function f(du, u, p, t) # In-place
        u = reshape(u, nrows, :)
        du = reshape(du, nrows, :)
        mtkf(du, u, p, t, c1, c2, c3)
    end
    function f(u, p, t) # Out-of-place
        u = reshape(u, nrows, :)
        @info u, p, t, c1, c2, c3
        @info mtkf
        du = mtkf(u, p, t, c1, c2, c3)
        reshape(hcat(du...), :)
    end
    return f
end

# Return a function to apply the MTK system to each column of u after reshaping to a matrix.
function mtk_grid_func(
        sys_mtk::System, domain::DomainInfo{T, AT}, u0,
        alg::MA = MapBroadcast(),
        jac_type::JT = BlockDiagonalJacobian();
        sparse = false, tgrad = false, vjp = true) where {
        T, AT, MA <: MapAlgorithm, JT <: JacobianType}
    sys_mtk, coord_args = _prepare_coord_sys(sys_mtk, domain)

    mtkf_coord = build_coord_ode_function(sys_mtk, coord_args, alg)
    jac_coord = build_coord_jac_function(sys_mtk, coord_args, alg; sparse = sparse)

    f = _mtk_grid_func(sys_mtk, mtkf_coord, domain, alg)

    nvars = length(unknowns(sys_mtk))

    jac_prototype = build_jacobian(jac_type, nvars, domain, alg, sparse)
    jf = mtk_jac_grid_func(sys_mtk, jac_coord, domain, jac_type, alg)

    kwargs = []
    if tgrad
        tgf = build_coord_tgrad_function(sys_mtk, coord_args, alg)
        tg = mtk_tgrad_grid_func(sys_mtk, tgf, domain, alg)
        push!(kwargs, :tgrad => tg)
    end
    if vjp
        vj = mtk_vjp_grid_func(sys_mtk, jac_coord, domain, alg)
        push!(kwargs, :vjp => vj)
    end
    ODEFunction(f; jac_prototype = jac_prototype, jac = jf, kwargs...), sys_mtk, coord_args
end

# Create a function to calculate the gridded time gradient.
# ngrid is the number of grid cells.
function mtk_tgrad_grid_func(sys_mtk, tgradf, domain, alg = MapBroadcast())
    nvar = length(unknowns(sys_mtk))
    II = CartesianIndices(tuple(size(domain)...))
    c1, c2, c3 = grid(domain)
    function tgrad(out, u, p, t) # In-place
        u = reshape(u, nvar, :)
        # FIXME(CT): I believe the tgrad output should be a vector and not a clone of the Jacobian.
        # This may be a bug in the DifferentialEquations.jl interface.
        for r in 1:size(u, 2)
            _u = view(u, :, r)
            tgradf(EarthSciMLBase.block(out, r), _u, p, t,
                c1[II[r][1]], c2[II[r][2]], c3[II[r][3]])
        end
    end
end

function mtk_vjp_grid_func(sys_mtk, mtkf, domain, alg = MapBroadcast())
    nvar = length(unknowns(sys_mtk))
    II = CartesianIndices(tuple(size(domain)...))
    c1, c2, c3 = grid(domain)
    function vjp(vJ, v, u, p, t)
        u = reshape(u, nvar, :)
        J = Matrix{eltype(u)}(undef, nvar, nvar)
        for r in 1:size(u, 2)
            _u = view(u, :, r)
            vJr = view(vJ, ((r - 1) * nvar + 1):(r * nvar), :)
            vr = view(v, ((r - 1) * nvar + 1):(r * nvar), :)
            mtkf.jac(J, _u, p, t, c1[II[j][1]], c2[II[j][2]], c3[II[j][3]])
            mul!(vJr, J', vr)
        end
    end
    function vjp(v, u, p, t)
        u = reshape(u, nvar, :)
        vcat([begin
                  _u = view(u, :, r)
                  vr = view(v, ((r - 1) * nvar + 1):(r * nvar), :)
                  J = mtkf.jac(_u, p, t, c1[II[j][1]], c2[II[j][2]], c3[II[j][3]])
                  J' * vr
              end
              for r in 1:size(u, 2)]...)
    end
    return vjp
end
