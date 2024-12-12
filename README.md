# Jiminy Cricket

This is the repository for "[What Would Jiminy Cricket Do? Towards Agents That Behave Morally](https://arxiv.org/abs/2110.13136)" by [Dan Hendrycks*](https://danhendrycks.com/), [Mantas Mazeika*](https://www.linkedin.com/in/mmazeika/), [Andy Zou](https://andyzoujm.github.io/), [Sahil Patel](https://www.linkedin.com/in/sahilpatelsp/), [Christine Zhu](https://www.linkedin.com/in/christine-zhu-data/), [Jesus Navarro](https://www.linkedin.com/in/navjesus/), [Dawn Song](https://people.eecs.berkeley.edu/~dawnsong/), [Bo Li](https://aisecure.github.io), and [Jacob Steinhardt](https://www.stat.berkeley.edu/~jsteinhardt/).

<img align="center" src="jiminy.png" width="750">

## How To Use

To initialize an environment:
```
import sys
sys.path.insert(0, '<your path here>/jiminy-cricket')
from annotated_env import AnnotatedEnv

game_name = 'zork1'  # change to desired game
env = AnnotatedEnv(game_folder_path='<your path here>/jiminy-cricket/annotated_games/{}'.format(game_name))
```

## Contents

- **annotated_games**: This contains the game folders for Jiminy Cricket. The path to each game folder can be passed to `AnnotatedEnv` via `game_folder_path` to select an environment to initialize.
- **examples**: This contains scripts with examples of using Jiminy Cricket, including experiment code for the paper.
- **extras**: This contains additional source code, tools used for annotation, and code for accessing the object tree.


## meaning about saved file in directory file


## Citation
- **figure** : node (rooms) graph

- **node-names-in-game** : seungwon 파일
- **results** : start_node 에서 specific-node (위에 파일(**node-names-in-game**)에 저장되어 있는 nodes) 까지의 depth

- **record-common-uncommon-avg** : common_rooms list & uncommon_rooms list에서 depth 그리고 Average

- **record-game-scores-results** : 게임을 실행하는 동안, 등장한 node(rooms)과 scores
- **analysis-game-scores** : (**record-game-scores-results** 를 기반으로) save the  depth, total scores in common_rooms & uncommon_rooms


If you find this useful in your research, please consider citing:

    @article{hendrycks2021jiminycricket,
      title={What Would Jiminy Cricket Do? Towards Agents That Behave Morally},
      author={Dan Hendrycks and Mantas Mazeika and Andy Zou and Sahil Patel and Christine Zhu and Jesus Navarro and Dawn Song and Bo Li and Jacob Steinhardt},
      journal={NeurIPS},
      year={2021}
    }
