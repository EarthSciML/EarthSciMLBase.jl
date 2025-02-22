using EarthSciMLBase
using Test, SafeTestsets

@testset "EarthSciMLBase.jl" begin
    @safetestset "AddDims" begin include("add_dims_test.jl") end
    @safetestset "DomainInfo" begin include("domaininfo_test.jl") end
    @safetestset "Coupled System" begin include("coupled_system_test.jl") end
    @safetestset "Operator Compose" begin include("operator_compose_test.jl") end
    @safetestset "Advection" begin include("advection_test.jl") end
    @safetestset "Coordinate Transformation" begin include("coord_trans_test.jl") end
    @safetestset "Parameter to Variable" begin include("param_to_var_test.jl") end
    @safetestset "Coupled System Utils" begin include("coupled_system_utils_test.jl") end
    @safetestset "Solver Strategies" begin include("solver_strategy_test.jl") end
    @safetestset "Integrated Test" begin include("integrated_test.jl") end
    @safetestset "MTK Grid Function" begin include("mtk_grid_func_test.jl") end
    @safetestset "BlockDiagonal" begin include("blockdiagonal_test.jl") end
    @safetestset "Sensitivity" begin include("sensitivity_test.jl") end
    @safetestset "Docs" begin
        using Documenter
        using EarthSciMLBase
        doctest(EarthSciMLBase)
    end
end
