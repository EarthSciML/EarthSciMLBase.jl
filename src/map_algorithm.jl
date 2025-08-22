export MapAlgorithm, MapBroadcast, MapThreads, MapKernel, map_closure_to_range

"""
A type to specify the algorithm used for performing a computation
across a range of grid cells.
"""
abstract type MapAlgorithm end

"""
Perform computations by broadcasting.
"""
struct MapBroadcast <: MapAlgorithm end
"""
Perform computations in parallel using multi-threading.
"""
struct MapThreads <: MapAlgorithm end
"""
Perform computations on CPU or GPU using AcceleratedKernels.jl.

kwargs are passed to the AcceleratedKernels.jl `foreachindex` function.
"""
struct MapKernel <: MapAlgorithm
    kwargs::Dict{Symbol, Any}
    function MapKernel(; kwargs...)
        new(kwargs)
    end
end

function map_closure_to_range(f, range, ::MapAlgorithm = MapThreads(), args...; kwargs...)
    ThreadsX.map(range) do i
        f(i, args...; kwargs...)
    end
end
function map_closure_to_range(f, range, ::MapBroadcast, args...; kwargs...)
    f2(i) = f(i, args...; kwargs...)
    f2.(range)
end
function map_closure_to_range(f, range, mk::MapKernel, args...; kwargs...)
    bknd = if (length(args) > 0) && (args[1] isa AbstractArray)
        AK.get_backend(args[1])
    else
        error("No backend specified for MapKernel. Please provide an array as the first argument.")
    end
    AK.foreachindex(range, bknd; mk.kwargs...) do i
        f(i, args...; kwargs...)
    end
end
