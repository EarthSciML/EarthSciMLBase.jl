export graph

"""
Create a graph from a CoupledSystem using the MetaGraphsNext package.
"""
function graph(sys::CoupledSystem)::MetaGraphsNext.MetaGraph
    g = MetaGraphsNext.MetaGraph(
        Graphs.Graph();
        label_type = Symbol,
        vertex_data_type = ModelingToolkit.System,
        edge_data_type = ConnectorSystem
    )
    systems = copy(sys.systems)
    for sys in systems # First do nodes
        g[nameof(sys)] = sys
    end
    for a in systems # Now do edges.
        for b in systems
            a_t, b_t = get_coupletype(a), get_coupletype(b)
            if hasmethod(couple2, (a_t, b_t))
                cs = couple2(a_t(deepcopy(a)), b_t(deepcopy(b)))
                g[nameof(a), nameof(b)] = cs
            end
        end
    end
    g
end
