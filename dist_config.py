import multiprocessing as mp
from dataclasses import dataclass


# ---------------------------
# Distributed Configuration
# ---------------------------

WORLD_SIZE = 8
GPUS_PER_NODE = 4
NUM_NODES = 2
BACKEND = "nccl"


@dataclass
class ProcessInfo:
    rank: int
    local_rank: int
    node_id: int
    world_size: int
    backend: str


def worker(info: ProcessInfo):

    print(
        f"""
=================================================
Process Started
-------------------------------------------------
Global Rank : {info.rank}
Local Rank  : {info.local_rank}
Node ID     : {info.node_id}
World Size  : {info.world_size}
Backend     : {info.backend}
GPU         : gpu:{info.local_rank}
=================================================
"""
    )


def main():

    processes = []

    for rank in range(WORLD_SIZE):

        node_id = rank // GPUS_PER_NODE
        local_rank = rank % GPUS_PER_NODE

        info = ProcessInfo(
            rank=rank,
            local_rank=local_rank,
            node_id=node_id,
            world_size=WORLD_SIZE,
            backend=BACKEND,
        )

        p = mp.Process(target=worker, args=(info,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
