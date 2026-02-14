export JacobianType, BlockDiagonalJacobian, BlockDiagonalOperatorJacobian

"""
A type for specifying the type that should be used for the Jacobian Matrix.
"""
abstract type JacobianType end

"""
Specify that a block-diagonal Jacobian matrix should be used.

This should allow efficient factorization on the CPU, but may not be fast on the
GPU unless the size of the blocks is large, because it currently each block has to be
factorized in series on the GPU rather than all blocks being factored in parallel.
"""
struct BlockDiagonalJacobian <: JacobianType end

"""
Specify that a block-diagonal Jacobian matrix should be used, and it should be wrapped in a
SciMLOperator.

This allows the use of Krylov methods for factorization which should parallelize well
on the GPU.
"""
struct BlockDiagonalOperatorJacobian <: JacobianType end

# Create a function to calculate the gridded Jacobian.
function mtk_jac_grid_func(
        sys_mtk, jacf, domain, ::BlockDiagonalJacobian, alg = MapBroadcast())
    nvar = length(unknowns(sys_mtk))
    c1, c2, c3 = concrete_grid(domain)
    function jac(J::AbstractBlockDiagonal, u, p, t) # In-place
        u = reshape(u, nvar, :)
        function calcJ(r, u, J, p, t, c1, c2, c3)
            jacf(view(J,:,:,r), view(u, :, r), p, t, c1[r], c2[r], c3[r])
            return nothing
        end
        map_closure_to_range(calcJ, 1:size(u, 2), alg, u, J.data, p, t, c1, c2, c3)
        nothing
    end
    function jac(u, p, t) # Out-of-place
        u = reshape(u, nvar, :)
        calcJ(r, u, p, t, c1, c2, c3) = jacf(view(u, :, r), p, t, c1[r], c2[r], c3[r])
        blocks = map_closure_to_range(calcJ, 1:size(u, 2), alg, u, p, t, c1, c2, c3)
        blocks3d = reduce((x, y) -> cat(x, y; dims = 3), blocks)
        return BlockDiagonal(blocks3d, alg)
    end
end

function mtk_jac_grid_func(
        sys_mtk, jacf, domain, jt::BlockDiagonalOperatorJacobian, alg = MapBroadcast())
    jacbc = mtk_jac_grid_func(sys_mtk, jacf, domain, BlockDiagonalJacobian(), alg)
    function jac(Jop::SMO.AbstractSciMLOperator, u, p, t) # In-place
        J = Jop.u
        jacbc(J, u, p, t)
        SMO.update_coefficients!(Jop, J, nothing, 0.0)
        nothing
    end
    function jac(u, p, t) # Out-of-place
        J = jacbc(u, p, t)
        build_jacobian(jt, J, alg)
    end
end

"""
Build the jacobian matrix.
"""
function build_jacobian(
        ::BlockDiagonalJacobian, nvars, domain::DomainInfo, alg, sparse::Bool)
    if sparse
        error("Sparse Jacobian not yet implemented for MTK grid functions.")
    end
    ncells = reduce(*, length.(grid(domain)))
    BlockDiagonal(init_array(domain, nvars, nvars, ncells), alg)
end
function build_jacobian(::BlockDiagonalJacobian, X::AbstractArray, alg)
    BlockDiagonal(X, alg)
end
function build_jacobian(
        ::BlockDiagonalOperatorJacobian, nvars, domain::DomainInfo, alg, sparse::Bool)
    # Functions for the SciMLOperator.
    opfunc!(w, v, u, p, t) = mul!(w, u, v)
    opfunc!(v, u, p, t) = u * v
    if sparse
        error("Sparse Jacobian not yet implemented for MTK grid functions.")
    end
    ncells = reduce(*, length.(grid(domain)))
    x = BlockDiagonal(init_array(domain, nvars, nvars, ncells), alg)
    SMO.FunctionOperator(opfunc!, b, o, u = x, p = nothing, t = 0.0, islinear = true,
        isconstant = true)
end
function build_jacobian(::BlockDiagonalOperatorJacobian, X::AbstractArray, alg)
    opfunc!(w, v, u, p, t) = mul!(w, u, v)
    opfunc!(v, u, p, t) = u * v
    x = BlockDiagonal(X, alg)
    SMO.FunctionOperator(opfunc!, b, o, u = x, p = nothing, t = 0.0, islinear = true,
        isconstant = true)
end
