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
    @safetestset "Simulator Utils" begin include("simulator_utils_test.jl") end
    @safetestset "Simulator" begin include("simulator_test.jl") end
    @safetestset "Docs" begin 
        using Documenter
        using EarthSciMLBase
        doctest(EarthSciMLBase) 
    end
end

