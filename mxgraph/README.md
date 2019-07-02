Graph API
-----------------------
CSRMat

 - Parameters:
    - end_points (np.ndarray) - Shape=(nnz,) 
    - ind_ptr (np.ndarray) - Shape=(#row+1,)
    - row_ids (np.ndarray) - Shape=(#row, )
    - col_ids (np.ndarray) - Shape=(#col, )
    - values  (np.ndarray or None) - Shape(nnz,) Default is True. If None, then all the value will be 1.
    - force_contiguous (bool) - Default is True.
 - staticmethod:
    - from_spy(mat) : Param - mat (sp.csr_matrix)
    - load(fname): Param - fname (str)
 - property:
    - size --> return #end_points
    - nnz --> return #nnz
    - shape --> return (#row, #col)
    - node_pair_indices --> return (2, size)
    - node_pair_ids --> return (2, size)
    - row_degrees --> return (#row, )
    - col_degrees --> return (#col, )
    - T --> transpose a CSRMat
 - function:
    - row_id_to_ind(node_ids)
    - col_id_to_ind(node_ids)
    - save(fname)
    - submat(row_indices, col_indices)
    - submat_by_id(row_ids, col_ids)
    - fetch_edges_by_ind(node_pair_indices)
    - fetch_edges_by_id(node_pair_ids)
    - remove_edges_by_ind(node_pair_indices)
    - remove_edges_by_id(node_pair_ids)
    - summary()
    - info()
    
HeterGraph
 - Parameters:
    - features (dict) - {node_key : np.ndarray (#node, fea_dim)} if None then set dict value=np.zeros(#node,)
    - node_ids (dict or None) - {node_key, np.ndarray (#node, )}
    - meta_graph (dict or None) - {node_key1: {node_key2: 1} }
    - csr_mat_dict (dict or None) - {(node_key1, node_key2) : CSRMat}
    
 - staticmethod:
    - load(dir_name): Param - dir_name (str)
 
 - property:
    - node_names --> return list
    - node_id_rmaps --> return np.array (Generate node_id_to_ind mappings)

 - function:
    - features_by_id(key, node_ids)
    - get_multi_link_structure() --> return dict 
        - multi_link_structure = {(node_key1, node_key2): len(multi_link) or None}
            - the keys are the same as those in *csr_mat_dict*
    - save(dir_name)
    - node_id_to_ind(key, node_ids)
    - fetch_edges_by_id(src_key, dst_key, node_pair_ids)
    - fetch_edges_by_ind(src_key, dst_key, node_pair_indices)
    - remove_edges_by_id(src_key, dst_key, node_pair_ids)
    - remove_edges_by_ind(src_key, dst_key, node_pair_indices)
    - sel_node_by_id(key, node_id)
    - gen_nd_features(ctx)
    - summary()


