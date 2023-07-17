export graph

"""
Create a graph from a ComposedEarthSciMLSystem using the MetaGraphsNext package.
"""
function graph(sys::ComposedEarthSciMLSystem)::MetaGraphsNext.MetaGraph
    g = MetaGraphsNext.MetaGraph(
        Graphs.Graph();
        label_type=Symbol,
        vertex_data_type=EarthSciMLODESystem,
        edge_data_type=ConnectorSystem,
    )
    for sys ∈ sys.systems # First do nodes
        if isa(sys, EarthSciMLODESystem)
            g[nameof(sys.sys)] = sys
        end
    end
    for sys ∈ sys.systems # Now do edges.
        if isa(sys, ConnectorSystem)
            g[nameof(sys.from.sys), nameof(sys.to.sys)] = sys
        end
    end
    g
end