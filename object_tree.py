# referenced by https://github.com/hendrycks/jiminy-cricket/blob/main/extras/object_tree/object_tree_examples.ipynb
from pprint import pprint
import argparse
import time
import os
import json
import sys
from collections import deque
import ast
sys.path.insert(0,'..')

from adjustText import adjust_text
import torch
import numpy as np
import matplotlib.pyplot as plt

# local directory
from extras.object_tree.annotated_env_with_object_tree import AnnotatedEnvWithObjectTree
from game_info import game_info
from count_depth import calculate_depth, find_nodeName2index, find_depth

def main1(game):
    annotated_env = AnnotatedEnvWithObjectTree(f'./extras/object_tree/annotated_games_with_object_tree/{game}', seed=1)
    print(annotated_env.reset()[0])

    nodes = annotated_env.get_object_tree_nodes()
    print(len(nodes.keys()))

    rooms = [x for x in list(nodes.keys()) if nodes[x]['room'] == True]
    objects = [x for x in list(nodes.keys()) if nodes[x]['room'] == False]
    print(f'Number of rooms: {len(rooms)}')
    print(f'Number of objects: {len(objects)}')

    print(f'List of objects in {game}:')
    print('  '.join(sorted(objects)))

    from IPython import embed; embed()

    print('\nObject tree entry for MAILBOX')
    print(nodes['MAILBOX'])

    observation, reward, done, info = annotated_env.step('open mailbox')
    print(observation)

    observation, reward, done, info = annotated_env.step('take leaflet')
    print(observation)

    nodes = annotated_env.get_object_tree_nodes()
    print(nodes['MAILBOX'])

    # Object and room dicts
    annotated_env.object_dicts['MAILBOX']
    annotated_env.room_dicts['WEST-OF-HOUSE']

    # Valid action generator
    start_time = time.time()
    valid_actions = annotated_env.get_valid_actions(nodes)
    print(f'Time taken: {time.time() - start_time}')
    print(f'Number of raw action candidates: {len(valid_actions)}')

    start_time = time.time()
    valid_actions = annotated_env.get_valid_actions(nodes, state_change_detection=True)
    print(f'Time taken: {time.time() - start_time}')
    print(f'Number of raw action candidates: {len(valid_actions)}')

    print(valid_actions)


def main2(game, args):
    '''Example application: Visualizing the game state'''
    annotated_env = AnnotatedEnvWithObjectTree(f'./extras/object_tree/annotated_games_with_object_tree/{game}')
    nodes = annotated_env.get_object_tree_nodes()
    print (annotated_env.get_anytree())

    # Now let's use force-directed graph drawing to make a prettier visualization.
    room_list = sorted(list(annotated_env.room_dicts.keys()))
    num_rooms = len(room_list)

    print (f"room_list => {room_list}")

    adj = torch.zeros(num_rooms, num_rooms)
    direction_list = []
    ideal_directions = {'NORTH': np.array([0, 1]), 'SOUTH': np.array([0, -1]), 'EAST': np.array([1, 0]),
                        'WEST': np.array([-1, 0]), 'NE': np.array([1, 1]), 'NW': np.array([-1, 1]),
                        'SE': np.array([1, -1]), 'SW': np.array([-1, -1])}

    # Step 1:
    for i in range(num_rooms):
        directions = annotated_env.room_dicts[room_list[i]]['directions']
        for dir_tuple in directions:
            direction, link_type, destination = dir_tuple
            if link_type == 'PER':
                continue  # ignore PER connections for now
            if type(destination) == list:
                destination = destination[0]  # ignore if-else syntax for now
            direction_list.append((room_list[i], destination, direction))  # for imposing direction loss
            destination_idx = room_list.index(destination)
            adj[i, destination_idx] = 1
            adj[destination_idx, i] = 1
    adj = adj * (1 - torch.eye(num_rooms))

    # Step 2: check each node (rooms)'s depth using adjacent matrix
    if game=="zork1" or game=="zork3" or game=="sorcerer":
        # Count items which item equals 0 or 1
        num_ones = torch.sum(adj == 1).item()
        num_zeros = torch.sum(adj == 0).item()
        print(f"Number of 1s: {num_ones}")
        print(f"Number of 0s: {num_zeros}")

        print (len(room_list) , adj.shape)
        assert adj.shape[0] == adj.shape[1] == len(room_list)
        
        print ("Find depths (function call) !")
        depths = calculate_depth(adj)
        # check if the matrix is symmetric
        print (torch.equal(depths, depths.T))

        # find index that is 'WEST-OF-HOUSE'
        start_node = 'WEST-OF-HOUSE' if game=="zork1" else "ZORK2-STAIR" if game=="zork3" else "TWISTED-FOREST"
        start_node_index = find_nodeName2index(node_name=start_node, room_list=room_list)
        print (f"{start_node}'s index is {start_node_index}")
        
        # load txt file
        file_path = f'node-names-in-game/{game}_analysis.txt'
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
        data_list = ast.literal_eval(content)

        results = {}
        for node_name in data_list:
            results[node_name] = find_depth(
                start_node=start_node_index, end_node=node_name, 
                room_list=room_list, depths=depths
            )
        
        # save results (for each node_rooms, find the depth from start_node to end_node)
        save_path = "results/" + os.path.splitext(file_path)[0].split('/')[1] + ".json"
        with open(save_path, 'w', encoding='utf-8') as file:
            json.dump(results, file, indent=4, ensure_ascii=False)

        if game=="zork1":
            common_rooms = ['WEST-OF-HOUSE', 'EAST-OF-HOUSE', 'NORTH-OF-HOUSE', 
                'SOUTH-OF-HOUSE', 'KITCHEN', 'LIVING-ROOM', 'CELLAR', 'FOREST-1', 
                'TROLL-ROOM', 'EAST-OF-CHASM', 'GALLERY', 'ATTIC']
        elif game=="sorcerer":
            common_rooms = ['TWISTED-FOREST', 'YOUR-QUARTERS', 'HALLWAY-1', 
                'HALLWAY-2', 'LOBBY', 'STORE-ROOM', 'LIBRARY', 'BELBOZ-QUARTERS', 
                'HELISTAR-QUARTERS', 'SERVANT-QUARTERS', 'APPRENTICE-QUARTERS']
        elif game=="zork3":
            common_rooms = ['ZORK2-STAIR', 'JUNCTION', 'CLEARING', 'CREEPY-CRAWL', 
                'SHADOW-1', 'SHADOW-2', 'FLATHEAD-OCEAN', 'CLIFF', 'FOGGY-ROOM', 'SHADOW-8', 
                'SHADOW-3', 'DAMP-PASSAGE', 'SHADOW-7', 'SHADOW-4', 'SHADOW-5', 'CLIFF-BASE', 
                'SHADOW-6', 'SLOPE']

        uncommon_rooms = [element for element in room_list if element not in common_rooms]
        common_rooms_dict, uncommon_rooms_dict = {}, {}
        for node_name in room_list:
            depth = find_depth(
                start_node=start_node_index, end_node=node_name, 
                room_list=room_list, depths=depths
            )

            if node_name in common_rooms:
                common_rooms_dict[node_name] = depth
            elif node_name in uncommon_rooms:
                uncommon_rooms_dict[node_name] = depth
            else:
                raise NotImplementedError()

        # calculate average at common_rooms & uncommon_rooms
        common_avg = sum(common_rooms_dict.values()) / len(common_rooms_dict.values())
        common_rooms_dict["Avg."] = common_avg

        uncommon_avg = sum(uncommon_rooms_dict.values()) / len(uncommon_rooms_dict.values())
        uncommon_rooms_dict["Avg."] = uncommon_avg

        # save common & uncommon depths and "average"
        save_common_path = f"record-common-uncommon-avg/{game}-common.json"
        with open(save_common_path, 'w', encoding='utf-8') as file:
            json.dump(common_rooms_dict, file, indent=4, ensure_ascii=False)

        save_uncommon_path = f"record-common-uncommon-avg/{game}-uncommon.json"
        with open(save_uncommon_path, 'w', encoding='utf-8') as file:
            json.dump(uncommon_rooms_dict, file, indent=4, ensure_ascii=False)


        # load txt file
        game_scores_result_path = f'record-game-scores-results/{game}.txt'
        with open(game_scores_result_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
        data_dict = ast.literal_eval(content)

        common_node_dict, uncommon_node_dict = {}, {}
        total_common_score, total_uncommon_score = 0, 0
        number_of_common_rooms, number_of_uncommon_rooms = 0, 0
        for node_name, scores in data_dict.items():
            depth = find_depth(
                start_node=start_node_index, end_node=node_name, 
                room_list=room_list, depths=depths
            )
            if node_name in common_rooms:
                common_node_dict[node_name] = {
                    "depth": depth,
                    "scores": scores,
                }
                total_common_score += sum(scores)
                number_of_common_rooms += 1
            elif node_name in uncommon_rooms:
                uncommon_node_dict[node_name] = {
                    "depth": depth,
                    "scores": scores,
                }
                total_uncommon_score += sum(scores)
                number_of_uncommon_rooms += 1

        save_path = f"analysis-game-scores/{game}-common.json"
        with open(save_path, 'w', encoding='utf-8') as file:
            json.dump(common_node_dict, file, indent=4, ensure_ascii=False)
        
        save_path = f"analysis-game-scores/{game}-uncommon.json"
        with open(save_path, 'w', encoding='utf-8') as file:
            json.dump(uncommon_node_dict, file, indent=4, ensure_ascii=False)

        save_path = f"analysis-game-scores/{game}-result.txt"
        text_data = f"""
        the number of common_rooms during game execution: {number_of_common_rooms}, total-common-score (sum): {total_common_score}
        the number of uncommon_rooms during game execution: {number_of_uncommon_rooms}, total-uncommon-score (sum): {total_uncommon_score}
        """
        with open(save_path, 'w', encoding='utf-8') as file:
            file.write(text_data)

        # from IPython import embed; embed()
        if args.only_execute_adj:
            return

    # Step 3:
    all_coords = []
    best_loss = np.inf
    for k in range(args.num):
        print('#', k)
        coords = torch.randn(num_rooms, 2)
        if k > 0:
            # initialize from previous best coords and add a random perturbation
            coords = 0.1 * coords + torch.FloatTensor(best_coords)
        initial_coords = coords.clone().numpy()
        coords.requires_grad_(True)
        optimizer = torch.optim.SGD([coords], lr=0.001, weight_decay=0, momentum=0.9)
        for i in range(args.step):
            pairwise_distances = (coords.unsqueeze(0) - coords.unsqueeze(1)).norm(p=2, dim=-1)

            """
            total loss : 5
            
            (1) Attraction Loss: 연결된 방들 간 거리를 줄이는 손실
            (2) Repulsion Loss: 모든 노드 간 거리 유지(충돌 방지)
            (3) Edge Length Regularization: 연결 간 최소/최대 거리 제약
            (4) Directional Loss: 방 간의 방향 정보를 유지하기 위한 손실
            (5) Maximum Distance Regularization: 모든 노드 간 최대 거리 제한
            """

            # main graph drawing losses
            attraction_loss = (pairwise_distances.pow(2) * adj).sum()  # spring-like behavior
            repulsion_loss = ((pairwise_distances + 1e-5).pow(-1) * (1 - torch.eye(len(adj)))).sum()

            # impose a maximum edge length of 5
            edge_length_reg1 = ((adj * pairwise_distances).clamp(min=5) - 5).pow(2).sum()

            # impose a minimum edge length of 1
            edge_length_reg2 = (-1 * ((adj * pairwise_distances).clamp(max=2) - 1)).pow(2).sum()

            # impose a max allowed distance between all nodes
            max_dist_reg = (pairwise_distances.clamp(min=20) - 20).pow(2).sum()

            # impose directional losses
            dir_losses = []
            for tmp in direction_list:
                room1, room2, direction = tmp
                ideal_dir = ideal_directions.get(direction)
                if ideal_dir is None:
                    continue
                else:
                    ideal_dir = torch.FloatTensor(ideal_dir)
                idx1 = room_list.index(room1)
                idx2 = room_list.index(room2)
                curr_dir = coords[idx2] - coords[idx1]

                if (curr_dir * ideal_dir).sum().item() != 0:
                    cosine_dist = 1 - ((curr_dir * ideal_dir).sum() / (curr_dir.norm(p=2) * ideal_dir.norm(p=2)))
                else:
                    cosine_dist = 1
                dir_losses.append(cosine_dist)
            dir_loss = sum(dir_losses)

            loss = attraction_loss + repulsion_loss + edge_length_reg1 + edge_length_reg2 + max_dist_reg + 20*dir_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(loss.item())

        print('final loss:', loss.item())
        all_coords.append(coords.detach().numpy())
        if loss.item() < best_loss:
            print('new best loss!')
            best_loss = loss.item()
            best_coords = coords.detach().numpy()
            best_initial_coords = initial_coords

    # Step 4:
    # Getting node colors
    node_colors = []
    for i in range(len(room_list)):
        num_children = len(nodes[room_list[i]]['children'])
        color1 = np.array([0, 0, 0])  # black
        color2 = np.array([1, 0.5, 0.2])  # orange
        weight = torch.sigmoid(4 * (torch.ones(1) * num_children) - 3).numpy()[0]
        # higher weight of color2 for larger num_children
        room_color = color2 * weight + color1 * (1 - weight)
        node_colors.append(room_color)

    # Step 5: draw figure
    plt.figure(figsize=(18,18))
    fig = plt.figure(figsize=(18,18))
    ax = fig.add_subplot(1, 1, 1)

    coords_np = best_coords
    ax.scatter(coords_np[:, 0], coords_np[:, 1], c=node_colors)
    for i in range(len(coords_np)):
        for j in range(i, len(coords_np)):
            if adj[i, j] == 1:
                ax.plot([coords_np[i, 0], coords_np[j, 0]], [coords_np[i, 1], coords_np[j, 1]], c='black')


    texts = []
    for i in range(len(coords_np)):
        texts.append(ax.text(coords_np[i, 0], coords_np[i, 1], room_list[i]))
    adjust_text(texts, only_move={'points':'y', 'texts':'y'})

    plt.axis('off')

    import datetime
    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    fig.savefig(f'figure/{game}_{time_string}.jpg', format='jpg')



if __name__ == '__main__':
    # 1. create parser
    parser = argparse.ArgumentParser()

    # 2. add arguments to parser
    parser.add_argument('--num',
                        type=int,
                        default=10,
                        help="~")
    parser.add_argument('--step',
                        type=int,
                        default=600,
                        help="~")
    parser.add_argument('--only-execute-adj', 
                        action="store_true", 
                        help="only execute code that used adjacent matrix")

    # 3. parse arguments
    args = parser.parse_args()

    # 4. use arguments
    print (args)


    # check the possible games
    path = "./extras/object_tree/annotated_games_with_object_tree"
    print (os.listdir(path))

    # game_list = os.listdir(path)
    # game_list = ['sorcerer']
    game_list = ['sorcerer', 'zork1', 'zork2', 'zork3']
    
    for game in game_list:
        print (f"\n\n==================\nstart game: {game} \n==================\n\n")
        main2(game, args)