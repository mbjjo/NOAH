


# Function that reads a .inp file and returns Pandas dataframes with all nodes, links, subcatchments, and rain gauges
def swmm_model_inventory(inpFile):
    import pandas as pd
    import pyswmm
    
    node_id_list = list()
    junction_list = list()
    outfall_list = list()
    storage_list = list()
    
    link_id_list = list()
    connections_list = list()
    conduit_list = list()
    orifice_list = list()
    outlet_list = list()
    pump_list = list()
    weir_list = list()
    
    raingauge_id_list = list()
    
    subcatchment_id_list = list()
    subcatchment_connection_list = list()
    
    
    with pyswmm.Simulation(inpFile) as sim:    
        allNodes = pyswmm.Nodes(sim)
        for node in allNodes:
            node_id_list.append(node.nodeid)
            #temp_node = pyswmm.Nodes(sim)[node_list[i]]
            junction_list.append(node.is_junction())
            outfall_list.append(node.is_outfall())
            storage_list.append(node.is_storage())
            
        allLinks = pyswmm.Links(sim)
        for link in allLinks:
            link_id_list.append(link.linkid)
            connections_list.append(link.connections)
            conduit_list.append(link.is_conduit())
            orifice_list.append(link.is_orifice())
            outlet_list.append(link.is_outlet())
            pump_list.append(link.is_pump())
            weir_list.append(link.is_weir())
            
        allRainGauges = pyswmm.RainGages(sim)
        for rg in allRainGauges:
            raingauge_id_list.append(rg.raingageid)
            
        allSubcatchments = pyswmm.Subcatchments(sim)
        for subcatchment in allSubcatchments:
            subcatchment_id_list.append(subcatchment.subcatchmentid)        
            subcatchment_connection_list.append(subcatchment.connection)    
        
    
    #node_type = ["" for x in range(len(node_id_list))]
    nodes_overview = pd.DataFrame({"id": node_id_list,
                                  "type": ["" for x in range(len(node_id_list))]})
    nodes_overview.loc[junction_list,"type"] = "junction"
    nodes_overview.loc[outfall_list,"type"] = "outfall"
    nodes_overview.loc[storage_list,"type"] = "storage"
    
     
    #links_type = ["" for x in range(len(link_id_list))]
    links_overview = pd.DataFrame({"id": link_id_list,
                                  "type": ["" for x in range(len(link_id_list))],
                                  "upstream_node": [connections_list[x][0] for x in range(len(connections_list))],
                                  "downstream_node": [connections_list[x][1] for x in range(len(connections_list))]})
    links_overview.loc[conduit_list,"type"] = "conduit"
    links_overview.loc[orifice_list,"type"] = "orifice"
    links_overview.loc[outlet_list,"type"] = "outlet"
    links_overview.loc[pump_list,"type"] = "pump"
    links_overview.loc[weir_list,"type"] = "weir"
    
    subcatchments_overview = pd.DataFrame({"id": subcatchment_id_list,
                                           "connection_node": [subcatchment_connection_list[x][1] for x in range(len(subcatchment_connection_list))]})
    
    return(nodes_overview, links_overview, subcatchments_overview, raingauge_id_list)



### Flow trace function -----------
    
# based on network only - inlet/outlet node status - assumes that user has correctly drawn pipes

# first run "swmm_model_inventory" function to get all links

# Function that goes through network and finds all nodes upstream of a chosen node.
# (does not consider if an upstream node has another downstream connection)
def backwards_network_trace(model_nodes, model_links, model_subs, key_node):
    import numpy as np
    import pandas as pd
    
    upstream_nodes = [key_node]
    uninvestigated_nodes = [key_node] # turn into list
    upstream_links = []
    
    while len(uninvestigated_nodes) > 0:
        present_node = uninvestigated_nodes.pop()
        upstream_idx = np.where(model_links["downstream_node"] == present_node)
        immediately_upstream_nodes = model_links["upstream_node"].loc[upstream_idx].tolist()
        immediately_upstream_links = model_links["id"].loc[upstream_idx].tolist()
        for up_node in immediately_upstream_nodes:
            if up_node not in upstream_nodes: 
                upstream_nodes.extend([up_node])
                uninvestigated_nodes.extend([up_node])
        
        for up_link in immediately_upstream_links:
            if up_link not in upstream_links: 
                upstream_links.extend([up_link])
    
    upstream_nodes_idx = np.isin(model_nodes["id"], upstream_nodes)
    upstream_nodes_df = model_nodes[upstream_nodes_idx].reset_index(drop = True)
    
    upstream_links_idx = np.isin(model_links["id"], upstream_links)
    upstream_links_df = model_links[upstream_links_idx].reset_index(drop = True)
    
    upstream_subs_idx = np.isin(model_subs["connection_node"], upstream_nodes)
    upstream_subs_df = model_subs[upstream_subs_idx].reset_index(drop = True)

    return (upstream_nodes_df, upstream_links_df, upstream_subs_df)    


# def backwards_network_trace(model_links, key_node):
#     import numpy as np

#     upstream_nodes = [key_node]
#     uninvestigated_nodes = [key_node] # turn into list
    
#     while len(uninvestigated_nodes) > 0:
#         present_node = uninvestigated_nodes.pop()
#         upstream_idx = np.where(model_links["downstream_node"] == present_node)
#         immediately_upstream = model_links["upstream_node"].loc[upstream_idx].tolist()
#         for up_node in immediately_upstream:
#             if up_node not in upstream_nodes: 
#                 upstream_nodes.extend([up_node])
#                 uninvestigated_nodes.extend([up_node])
#     return(upstream_nodes)

# Function that identifies all downstream nodes from a user specified "key node"
def forwards_network_trace(model_links, key_node):
    import numpy as np

    downstream_nodes = [key_node]
    uninvestigated_nodes = [key_node] # turn into list
    
    while len(uninvestigated_nodes) > 0:
        present_node = uninvestigated_nodes.pop()
        upstream_idx = np.where(model_links["upstream_node"] == present_node)
        immediately_downstream = model_links["downstream_node"].loc[upstream_idx].tolist()
        for up_node in immediately_downstream:
            if up_node not in downstream_nodes: 
                downstream_nodes.extend([up_node])
                uninvestigated_nodes.extend([up_node])
    return(downstream_nodes)


# Function that goes through network and finds all nodes connected upwardly of a chosen node.
# (does consider if an upstream node has another downstream connection)
def backwards_connected_network_trace(model_links, key_node):
    import numpy as np
    
    upstream_nodes = [key_node]
    uninvestigated_nodes = [key_node] # turn into list
    
    present_node = uninvestigated_nodes.pop()
    upstream_idx = np.where(model_links["downstream_node"] == present_node)
    immediately_upstream = model_links["upstream_node"].loc[upstream_idx].tolist()
    for up_node in immediately_upstream:
        if up_node not in upstream_nodes: 
            upstream_nodes.extend([up_node])
            uninvestigated_nodes.extend([up_node])
                
    while len(uninvestigated_nodes) > 0:
        present_node = uninvestigated_nodes.pop()
        upstream_idx = np.where(model_links["downstream_node"] == present_node)
        immediately_upstream = model_links["upstream_node"].loc[upstream_idx].tolist()
        downstream_idx = np.where(model_links["upstream_node"] == present_node)
        immediately_downstream = model_links["downstream_node"].loc[downstream_idx].tolist()
        connected_nodes = immediately_upstream + immediately_downstream
        for con_node in connected_nodes:
            if con_node not in upstream_nodes: 
                upstream_nodes.extend([con_node])
                uninvestigated_nodes.extend([con_node])
    return(upstream_nodes)








