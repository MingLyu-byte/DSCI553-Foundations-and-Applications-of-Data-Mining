import os
import sys

import pyspark
import json
import datetime
import random
import math
import itertools
import time
from collections import Counter
from collections import defaultdict


def Girvan_Newman(root, adjacent_vertices, vertices):
    tree = {}
    children_dict = {}
    parent_dict = defaultdict(set)
    num_path = {}
    level = 1

    tree[0] = root
    level_vertices = adjacent_vertices[root]
    children_dict[root] = level_vertices
    num_path[root] = 1
    used_vertices = {root}

    for child in level_vertices:
        parent_dict[child].add(root)

    while level_vertices != set():
        tree[level] = level_vertices
        used_vertices = used_vertices.union(level_vertices)
        cur_vertices = set()
        for vertex in level_vertices:
            adj_ver_new = adjacent_vertices[vertex].difference(used_vertices)
            children_dict[vertex] = adj_ver_new

            for k in adj_ver_new:
                parent_dict[k].add(vertex)

            parents = parent_dict[vertex]
            if parents:
                num_path[vertex] = sum([num_path[i] for i in parents])
            else:
                num_path[vertex] = 1
            cur_vertices = cur_vertices.union(adj_ver_new)

        level_vertices = cur_vertices
        level += 1

    vertex_value = defaultdict(float)
    for vertex in vertices:
        vertex_value[vertex] = 1

    edge_t = {}
    while level != 1:
        for vertex in tree[level - 1]:
            parents = parent_dict[vertex]
            total_path = num_path[vertex]
            for parent in parents:
                weight = num_path[parent] / total_path
                edge_t[tuple(sorted((vertex, parent)))] = weight * vertex_value[vertex]
                vertex_value[parent] += edge_t[tuple(sorted((vertex, parent)))]

        level -= 1
    return [(k, v) for k, v in edge_t.items()]


def find_single_community(root, adjacent_vertices):
    neighbour_vertices = adjacent_vertices[root]
    if not neighbour_vertices:
        return {root}
    queue = [root]
    visited = {root}

    while queue:
        cur = queue.pop(0)
        cur_neighobours = adjacent_vertices[cur]
        for neighobour in cur_neighobours:
            if neighobour not in visited:
                visited.add(neighobour)
                queue.append(neighobour)

    return visited


def find_all_communities(vertices, adjacent_vertices):
    root = random.sample(vertices, 1)[0]
    communities = []
    used_nodes = find_single_community(root, adjacent_vertices)
    nodes_left = vertices.difference(used_nodes)
    communities.append(used_nodes)
    while True:
        cur_community = find_single_community(random.sample(nodes_left, 1)[0], adjacent_vertices)
        communities.append(cur_community)
        used_nodes = used_nodes.union(cur_community)
        nodes_left = nodes_left.difference(cur_community)
        if not nodes_left:
            break

    return communities


def calculate_modularity(communities, m, A_matrix, degree_dict):
    modularity = 0.0
    for community in communities:
        cur_modularity = 0.0
        for i in community:
            for j in community:
                cur_modularity += A_matrix[(i, j)] - degree_dict[i] * degree_dict[j] / (2 * m)
        modularity += cur_modularity
    return modularity / (2 * m)


if __name__ == '__main__':
    input_thresh_hold = int(sys.argv[1])
    input_file_path = sys.argv[2]
    betweeness_output_path = sys.argv[3]
    community_output_path = sys.argv[4]

    start = time.time()
    sc = pyspark.SparkContext.getOrCreate()
    sc.setLogLevel("WARN")
    start = time.time()
    input_lines = sc.textFile(input_file_path).map(lambda x: x.strip().split(",")) \
        .filter(lambda x: x[0] != 'user_id') \
        .map(lambda x: (str(x[0]), str(x[1])))

    distinct_user = input_lines.map(lambda row: row[0]).distinct().collect()
    user_dict = input_lines.groupByKey() \
        .map(lambda x: (x[0], list(set(x[1])))) \
        .collectAsMap()

    edges = set()
    vertices = set()
    for c in itertools.combinations(distinct_user, 2):
        p1 = c[0]
        p2 = c[1]
        if len(set(user_dict[p1]).intersection(set(user_dict[p2]))) >= input_thresh_hold:
            edges.add((p1, p2))
            vertices.add(p1)
            vertices.add(p2)

    adjacent_vertices = defaultdict(set)
    for p in edges:
        adjacent_vertices[p[0]].add(p[1])
        adjacent_vertices[p[1]].add(p[0])

    betweenness = sc.parallelize(vertices) \
        .map(lambda x: Girvan_Newman(x, adjacent_vertices, vertices)) \
        .flatMap(lambda x: [p for p in x]) \
        .reduceByKey(lambda a, b: a + b) \
        .map(lambda x: (x[0], x[1] / 2)) \
        .sortBy(lambda x: (-x[1], x[0]))

    betweenness_output = betweenness.map(lambda x: (x[0], round(x[1], 5))) \
        .collect()

    f_betweeness = open(betweeness_output_path, "w")
    for i in betweenness_output:
        f_betweeness.write(str(i[0]) + "," + str(i[1]) + "\n")
    f_betweeness.close()

    degree_dict = {}
    for k, v in adjacent_vertices.items():
        degree_dict[k] = len(v)

    A_matrix = defaultdict(float)
    for e in edges:
        i = e[0]
        j = e[1]
        A_matrix[(i, j)] = 1
        A_matrix[(j, i)] = 1
    m = len(edges)
    edges_left = m
    max_modularity = -10000
    best_communities = None
    betweenness = betweenness.collect()

    while True:
        highest_score = betweenness[0][1]
        for pair in betweenness:
            if highest_score == pair[1]:
                p1 = pair[0][0]
                p2 = pair[0][1]
                adjacent_vertices[p1].remove(p2)
                adjacent_vertices[p2].remove(p1)
                edges_left -= 1
            else:
                break

        cur_communities = find_all_communities(vertices, adjacent_vertices)
        cur_modularity = calculate_modularity(cur_communities, m, A_matrix, degree_dict)
        if cur_modularity > max_modularity:
            max_modularity = cur_modularity
            best_communities = cur_communities

        if edges_left == 0:
            break

        betweenness = sc.parallelize(vertices) \
            .map(lambda x: Girvan_Newman(x, adjacent_vertices, vertices)) \
            .flatMap(lambda x: [p for p in x]) \
            .reduceByKey(lambda a, b: a + b) \
            .map(lambda x: (x[0], x[1] / 2)) \
            .sortBy(lambda x: (-x[1], x[0])) \
            .collect()

    f_communties = open(community_output_path, "w")
    best_communities = sc.parallelize(best_communities) \
        .map(lambda x: sorted(x)) \
        .sortBy(lambda x: (len(x), x)) \
        .collect()

    for i in best_communities:
        f_communties.write(str(i)[1:-1] + "\n")
    f_communties.close()

    print("Duration:", time.time() - start)
