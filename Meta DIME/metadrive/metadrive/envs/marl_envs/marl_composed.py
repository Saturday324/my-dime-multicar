import copy
from math import floor

from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.road_network import Road
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.manager.base_manager import BaseManager
from metadrive.manager.spawn_manager import SpawnManager
from metadrive.utils import Config


MAComposedConfig = dict(
    num_agents=20,
    map="XTXTS",
    map_config=dict(
        exit_length=60,
        lane_num=2,
    ),
)


def _unique_roads(roads):
    unique = []
    seen = set()
    for road in roads:
        key = (road.start_node, road.end_node)
        if key in seen:
            continue
        seen.add(key)
        unique.append(road)
    return unique


class ComposedSpawnManager(SpawnManager):
    """Spawn and respawn agents from entrances collected across all sub-blocks."""

    def __init__(self):
        BaseManager.__init__(self)
        self.initialized = True
        self.num_agents = self.engine.global_config["num_agents"]
        self.exit_length = self.engine.global_config["map_config"]["exit_length"] - FirstPGBlock.ENTRANCE_LENGTH
        assert self.exit_length >= self.RESPAWN_REGION_LONGITUDE, (
            "The exist length {} should greater than minimal longitude interval {}.".format(
                self.exit_length, self.RESPAWN_REGION_LONGITUDE
            )
        )
        self.lane_num = self.engine.global_config["map_config"]["lane_num"]
        self.spawn_roads = []
        self.safe_spawn_places = {}
        self.need_update_spawn_places = True
        self.spawn_places_used = []
        self._init_agent_configs = copy.copy(self.engine.global_config["agent_configs"])
        self.available_agent_configs = []

    def _get_current_map(self):
        current_map = self.engine.current_map
        if current_map is not None:
            return current_map

        map_manager = getattr(self.engine, "map_manager", None)
        if map_manager is None:
            return None

        if getattr(map_manager, "current_map", None) is not None:
            return map_manager.current_map

        maps = getattr(map_manager, "maps", None)
        if maps is not None:
            return maps.get(self.engine.global_seed, None)

        return None

    def _get_spawn_roads(self):
        current_map = self._get_current_map()
        assert current_map is not None, "Composed map is not initialized before spawn manager reset"

        roads = []
        for block in getattr(current_map, "blocks", []):
            roads.extend(block.get_respawn_roads())
        roads = _unique_roads(roads)
        assert len(roads) > 0, "No spawn roads found for composed multi-agent map"
        return roads

    def _auto_fill_spawn_roads_with_real_lanes(self, spawn_roads):
        current_map = self._get_current_map()
        assert current_map is not None, "Composed map is not initialized before computing spawn slots"

        road_infos = []
        total_capacity = 0
        for road in spawn_roads:
            lanes = road.get_lanes(current_map.road_network)
            lane_num = len(lanes)
            lane_length = min(l.length for l in lanes)
            num_slots = max(int(floor(lane_length / SpawnManager.RESPAWN_REGION_LONGITUDE)), 1)
            road_infos.append((road, lane_num, lane_length, num_slots))
            total_capacity += lane_num * num_slots

        if self.num_agents is not None:
            assert self.num_agents > 0 or self.num_agents == -1
            assert self.num_agents <= total_capacity, (
                "Too many agents! We only accept {} agents, but you have {} agents!".format(
                    total_capacity, self.num_agents
                )
            )

        agent_configs = []
        safe_spawn_places = []
        for road, lane_num, lane_length, num_slots in road_infos:
            for lane_idx in range(lane_num):
                for slot_idx in range(num_slots):
                    long = 0.5 * self.RESPAWN_REGION_LONGITUDE + slot_idx * self.RESPAWN_REGION_LONGITUDE
                    long = min(long, lane_length - 1.0)
                    lane_tuple = road.lane_index(lane_idx)
                    agent_configs.append(
                        Config(
                            dict(
                                identifier="|".join((str(s) for s in lane_tuple + (slot_idx, ))),
                                config={
                                    "spawn_lane_index": lane_tuple,
                                    "spawn_longitude": long,
                                    "spawn_lateral": 0,
                                },
                            ),
                            unchangeable=True,
                        )
                    )
                    if slot_idx == 0:
                        safe_spawn_places.append(copy.deepcopy(agent_configs[-1]))

        return agent_configs, safe_spawn_places

    def _route_exists(self, current_map, spawn_lane_index, destination_node):
        if destination_node is None or destination_node == spawn_lane_index[0]:
            return False
        try:
            checkpoints = current_map.road_network.shortest_path(spawn_lane_index, destination_node)
        except Exception:
            return False
        return len(checkpoints) >= 2

    def reset(self):
        spawn_roads = self._get_spawn_roads()
        agent_configs, safe_spawn_places = self._auto_fill_spawn_roads_with_real_lanes(spawn_roads)
        self.available_agent_configs = agent_configs
        self.safe_spawn_places = {place["identifier"]: place for place in safe_spawn_places}
        self.spawn_roads = spawn_roads
        self.engine.global_config["spawn_roads"] = copy.deepcopy(spawn_roads)
        current_map = self._get_current_map()
        if current_map is not None:
            current_map.spawn_roads = spawn_roads
        return super(ComposedSpawnManager, self).reset()

    def update_destination_for(self, agent_id, vehicle_config):
        current_map = self._get_current_map()
        assert current_map is not None, "Composed map is not initialized before destination assignment"

        start_road = Road(*vehicle_config["spawn_lane_index"][:2])
        candidate_roads = [road for road in self.spawn_roads if road != start_road]
        self.np_random.shuffle(candidate_roads)

        for candidate_road in candidate_roads:
            destination_node = (-candidate_road).end_node
            if self._route_exists(current_map, vehicle_config["spawn_lane_index"], destination_node):
                vehicle_config["destination"] = destination_node
                return vehicle_config

        # Fall back to the default MetaDrive route assignment if none of the collected entrances is reachable.
        vehicle_config["destination"] = None
        return vehicle_config


class MultiAgentComposedEnv(MultiAgentMetaDrive):
    @staticmethod
    def default_config() -> Config:
        return MultiAgentMetaDrive.default_config().update(copy.deepcopy(MAComposedConfig), allow_add_new_key=True)

    def setup_engine(self):
        # Skip the default MultiAgentMetaDrive SpawnManager initialization because it assumes
        # only the first-block spawn road is available, which is too restrictive for composed maps.
        MetaDriveEnv.setup_engine(self)
        self.engine.register_manager("spawn_manager", ComposedSpawnManager())
