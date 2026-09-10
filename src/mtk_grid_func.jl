# Post-codegen AST rewrite for Reactant: turns scalar `u`/`out` access and
# scalar operations into broadcasted equivalents.  Coordinate arguments are
# handled natively by `build_function_wrapper` (see `gen_coord_func`); this
# rewrite only needs to handle array broadcasting.
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

# Move coordinate references from the generated code's body to the argument list.
# Rewrites the generated function's AST *after* codegen: (1) appends the
# coordinate argument symbols to the signature; (2) replaces every literal
# `_CoordTmpF` call site with the matching coordinate argument. Operating
# post-codegen (rather than substituting the symbolic expressions before codegen)
# preserves the `===` node sharing that the SymbolicUtils fast_toexpr backend
# relies on for deduplication, which keeps the generated body from blowing up on
# large mechanisms (large sparse Jacobians).
function rewrite_coord_func(x, coord_args, idv::Symbol)
    if @capture(x, function (args__)
        body_
    end)
        return :(function ($(args...), $(coord_args...))
            $body
        end)
    elseif @capture(x, (a_)(b_))
        # Under the fast_toexpr backend the independent variable may be renamed to
        # a hashed argument symbol (`__argₛᵧₘ<hash>`), so we do NOT require
        # `b == idv` — a literal `_CoordTmpF` instance in call position is
        # unambiguous by itself.
        if a isa _CoordTmpF
            return :($(coord_args[a.idx]))
        end
    end
    return x
end

# Loud build-time guard: a `_CoordTmpF` left in the generated code would silently
# evaluate to `Inf` at runtime (the `(::_CoordTmpF)(t) = Inf` fallback below). If
# the codegen backend ever changes shape such that the postwalk pattern above no
# longer matches, fail the build instead.
function _assert_no_coord_tmp(ex)
    ok = Ref(true)
    MacroTools.postwalk(ex) do x
        x isa _CoordTmpF && (ok[] = false)
        x
    end
    ok[] || error(
        "gen_coord_func: an unrewritten _CoordTmpF coordinate placeholder remains in " *
        "the generated code — the post-codegen AST rewrite pattern no longer matches " *
        "this codegen backend. Coordinates would evaluate to Inf at runtime.")
    return ex
end

function _add_coord_args(ex, coord_args, idv::Symbol, ::MapAlgorithm)
    ex = MacroTools.postwalk(x -> rewrite_coord_func(x, coord_args, idv), ex)
    _assert_no_coord_tmp(ex)
end

function _add_coord_args(ex, coord_args, idv::Symbol, ::MapReactant)
    ex = MacroTools.postwalk(x -> rewrite_coord_func(x, coord_args, idv), ex)
    _assert_no_coord_tmp(ex)
    ex = MacroTools.postwalk(x -> rewrite_broadcast(x), ex)
end

"""
Generate a scalar/in-place function that accepts the three coordinate values as
trailing arguments.  The generated signature is `(u, p, t, c1, c2, c3)` (out-of-place)
or `(out, u, p, t, c1, c2, c3)` (in-place), matching the arity expected by the
downstream grid-evaluation machinery.
"""
function gen_coord_func(sys, expr, coord_args, alg::MapAlgorithm = MapBroadcast();
        eval_expression = false, eval_module = @__MODULE__)
    idv = var2symbol(ModelingToolkit.get_iv(sys))
    fexpr = ModelingToolkit.generate_custom_function(sys, expr, expression = Val{true})
    if fexpr isa Tuple
        fexpr = _add_coord_args.(fexpr, (coord_args,), (idv,), (alg,))
        f = ModelingToolkit.eval_or_rgf.(fexpr; eval_expression, eval_module)
        return ModelingToolkit.GeneratedFunctionWrapper{(2, 6, true)}(f[1], f[2])
    else
        fexpr = _add_coord_args(fexpr, coord_args, idv, alg)
        return ModelingToolkit.eval_or_rgf(fexpr; eval_expression, eval_module)
    end
end

function _get_coord_args(sys, domain)
    coords = EarthSciMLBase.coord_params(sys, domain)
    # Symbolic names for the trailing coord arguments of the generated function.
    coord_args = Symbol.(nameof.(coords), (:_arg,))
    coords, coord_args
end

# Placeholder symbolic function used to keep coordinate-valued expressions
# flowing through `mtkcompile` without adding the coordinates to the parameter
# list.  The underlying coordinate is never evaluated directly — every
# `_CoordTmpF(coord, i)(t)` term is substituted with a scalar placeholder
# parameter before code generation in `gen_coord_func`.  The scalar-arg method
# returns `Inf` so that any stray, unsubstituted call at runtime produces a
# loud, recognisable failure rather than silently returning a plausible value.
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
