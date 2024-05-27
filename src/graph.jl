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
        g[nameof(sys.sys)] = sys
    end
    for sysa ∈ sys.systems # Now do edges.
        for sysb ∈ sys.systems
            if applicable(couple, sysa, sysb)
                g[nameof(sysa.sys), nameof(sysb.sys)] = couple(sysa, sysb)
            end
        end
    end
    g
end