from tqdm import tqdm

import torch
from collections import deque

def calculate_depth(adj):
    num_rooms = adj.shape[0]
    depths = torch.full((num_rooms, num_rooms), -1)  # 초기값은 -1로 설정
    
    for start_node in tqdm(range(num_rooms)):
        visited = [-1] * num_rooms  # 방문 여부를 체크하며 초기값 -1
        queue = deque([(start_node, 0)])  # (현재 노드, 현재 깊이)
        
        while queue:
            current_node, current_depth = queue.popleft()
            if visited[current_node] == -1:  # 아직 방문하지 않은 노드만 처리
                visited[current_node] = current_depth
                for neighbor in range(num_rooms):
                    if adj[current_node, neighbor] == 1 and visited[neighbor] == -1:
                        queue.append((neighbor, current_depth + 1))
        
        # 방문된 노드들의 depth를 저장
        depths[start_node] = torch.tensor(visited)
    
    return depths

def find_nodeName2index(node_name: str, room_list:list) -> int:
    for i, item in enumerate(room_list):
            if item == node_name:
                return i

def find_depth(start_node, end_node, room_list, depths):
    if type(start_node) != int:
        start_node = find_nodeName2index(node_name, room_list)

    if type(end_node) != int:
        end_node = find_nodeName2index(end_node, room_list)
    
    return depths[start_node][end_node].item()
