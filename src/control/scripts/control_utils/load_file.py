#!/usr/bin/env python3

import networkx as nx
import os
import yaml
from path.path_functions import get_indices_within_distance

class LoadData:
    def __init__(self):
        pass

    def load_path_file(self, range_data, path, segment_dist_map, default_dist):
        range_index = {}

        for segment, seg_info in range_data.items():
            segment_type = seg_info.get('type', 'midpoint')
            coords_list = seg_info.get('coords', [])

            if segment_type == 'range':
                if len(coords_list) != 2:
                    range_index[segment] = (-1, -1)
                    continue

                start_coords = coords_list[0]
                end_coords   = coords_list[1]

                start_indices = get_indices_within_distance(path, start_coords[0], start_coords[1])
                end_indices   = get_indices_within_distance(path, end_coords[0], end_coords[1])

                if not start_indices or not end_indices:
                    range_index[segment] = (-1, -1)
                    continue

                start_index = min(start_indices, key=lambda idx:
                                (path[idx][0] - start_coords[0])**2 + (path[idx][1] - start_coords[1])**2)
                end_index = min(end_indices, key=lambda idx:
                                (path[idx][0] - end_coords[0])**2 + (path[idx][1] - end_coords[1])**2)

                range_index[segment] = (start_index, end_index)

            elif segment_type == 'midpoint':
                dist = segment_dist_map.get(segment, default_dist)
                node_indices = []

                for (nx, ny) in coords_list:
                    near_inds = get_indices_within_distance(path, nx, ny, dist)
                    node_indices.extend(near_inds)

                node_indices = sorted(set(node_indices))
                range_index[segment] = node_indices

            else:
                dist = segment_dist_map.get(segment, default_dist)
                node_indices = []

                for (nx, ny) in coords_list:
                    near_inds = get_indices_within_distance(path, nx, ny, dist)
                    node_indices.extend(near_inds)

                node_indices = sorted(set(node_indices))
                range_index[segment] = node_indices

        return range_index

    def load_graphml_file(self, file_path):
        try:
            g = nx.read_graphml(file_path)
            return g
        except Exception as e:
            return None
        
    def load_range_data_file(self, file_path, graph):
        if not os.path.exists(file_path):
            return {}

        try:
            with open(file_path, 'r') as f:
                yaml_data = yaml.safe_load(f)

            range_data = {}

            for segment, seg_data in yaml_data.items():
                segment_type = seg_data.get('type', 'midpoint')

                if 'nodes' not in seg_data:
                    continue
                nodes_list = seg_data['nodes']

                node_coords = []

                for sublist in nodes_list:
                    if not isinstance(sublist, list):
                        continue

                    if len(sublist) == 1:
                        node_id = str(sublist[0])
                        if node_id not in graph.nodes:
                            continue

                        x = float(graph.nodes[node_id].get('x', 0.0))
                        y = float(graph.nodes[node_id].get('y', 0.0))
                        node_coords.append((x, y))

                    elif len(sublist) == 2:
                        node_id1 = str(sublist[0])
                        node_id2 = str(sublist[1])

                        if (node_id1 not in graph.nodes) or (node_id2 not in graph.nodes):
                            continue

                        x1 = float(graph.nodes[node_id1].get('x', 0.0))
                        y1 = float(graph.nodes[node_id1].get('y', 0.0))
                        x2 = float(graph.nodes[node_id2].get('x', 0.0))
                        y2 = float(graph.nodes[node_id2].get('y', 0.0))

                        if segment_type == "midpoint":
                            mid_x = (x1 + x2) / 2.0
                            mid_y = (y1 + y2) / 2.0
                            node_coords.append((mid_x, mid_y))
                        elif segment_type == "range":
                            node_coords.append((x1, y1))
                            node_coords.append((x2, y2))

                range_data[segment] = {
                    "type": segment_type,
                    "coords": node_coords
                }

            return range_data

        except Exception as e:
            return {}
        
    def load_node(self, graph, node_id):
        node_id = str(node_id)
        
        node_data = graph.nodes[node_id]

        x = float(node_data.get('x', 0.0))
        y = float(node_data.get('y', 0.0))

        return x, y
