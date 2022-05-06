using EarthSciMLBase
using Test, SafeTestsets

@testset "EarthSciMLBase.jl" begin
    @safetestset "Composition" begin include("composition_test.jl") end
    @safetestset "AddDims" begin include("add_dims_test.jl") end
end
