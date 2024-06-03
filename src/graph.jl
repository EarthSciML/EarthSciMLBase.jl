export graph

"""
Create a graph from a CoupledSystem using the MetaGraphsNext package.
"""
function graph(sys::CoupledSystem)::MetaGraphsNext.MetaGraph
    g = MetaGraphsNext.MetaGraph(
        Graphs.Graph();
        label_type=Symbol,
        vertex_data_type=ModelingToolkit.ODESystem,
        edge_data_type=ConnectorSystem,
    )
    systems = copy(sys.systems)
    hashes = systemhash.(systems)
    for sys ∈ systems # First do nodes
        g[nameof(sys)] = sys
    end
    for (i, a) ∈ enumerate(systems) # Now do edges.
        for (j, b) ∈ enumerate(systems)
            if (hashes[i], hashes[j]) ∈ keys(coupling_registry)
                f = coupling_registry[hashes[i], hashes[j]]
                cs = f(deepcopy(a), deepcopy(b))
                g[nameof(a), nameof(b)] = cs
            end
        end
    end
    g
end