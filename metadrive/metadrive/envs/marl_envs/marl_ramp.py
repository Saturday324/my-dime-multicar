import copy
from math import floor

from metadrive.component.map.pg_map import PGMap
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.ramp import InRampOnStraight, OutRampOnStraight
from metadrive.component.road_network import Road
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.manager.base_manager import BaseManager
from metadrive.manager.pg_map_manager import PGMapManager
from metadrive.manager.spawn_manager import SpawnManager
from metadrive.utils import Config

MAInRampConfig = dict(
    num_agents=12,
    map_config=dict(
        exit_length=60,
        lane_num=2,
        length=30,
        extension_length=30,
    ),
    top_down_camera_initial_x=105,
    top_down_camera_initial_y=18,
    top_down_camera_initial_z=110,
)

MAOutRampConfig = dict(
    num_agents=12,
    map_config=dict(
        exit_length=60,
        lane_num=2,
        length=30,
        extension_length=30,
    ),
    top_down_camera_initial_x=105,
    top_down_camera_initial_y=18,
    top_down_camera_initial_z=110,
)


def _build_ramp_block(current_map, first_block, ramp_cls, parent_node_path, physics_world):
    ramp_block = ramp_cls(
        1,
        first_block.get_socket(index=0),
        current_map.road_network,
        random_seed=1,
        ignore_intersection_checking=False,
    )

    extra_config = {}
    for key in ("length", "extension_length"):
        if key in current_map.config and current_map.config[key] is not None:
            extra_config[key] = current_map.config[key]

    ramp_block.construct_block(parent_node_path, physics_world, extra_config=extra_config or None)
    return ramp_block


def _find_terminal_positive_roads(block, excluded_roads=None):
    excluded_pairs = {
        (road.start_node, road.end_node) for road in (excluded_roads or [])
    }
    graph = block.block_network.graph
    terminal_roads = []
    for start_node, to_dict in graph.items():
        if start_node.startswith(Road.NEGATIVE_DIR):
            continue
        for end_node in to_dict.keys():
            if end_node.startswith(Road.NEGATIVE_DIR):
                continue
            if (start_node, end_node) in excluded_pairs:
                continue
            if end_node not in graph:
                terminal_roads.append(Road(start_node, end_node))
    return terminal_roads


class _BaseMARampMap(PGMap):
    RAMP_CLS = None

    def _generate(self):
        length = self.config["exit_length"]

        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        assert len(self.road_network.graph) == 0, "These Map is not empty, please create a new map to read config"

        first_block = FirstPGBlock(
            self.road_network,
            self.config[self.LANE_WIDTH],
            self.config[self.LANE_NUM],
            parent_node_path,
            physics_world,
            length=length,
        )
        self.blocks.append(first_block)

        ramp_block = _build_ramp_block(self, first_block, self.RAMP_CLS, parent_node_path, physics_world)
        self.blocks.append(ramp_block)
        self._configure_scene(first_block, ramp_block)

    def _configure_scene(self, first_block, ramp_block):
        raise NotImplementedError

    def configure_runtime_scene(self):
        first_block, ramp_block = self.blocks[0], self.blocks[1]
        self._configure_scene(first_block, ramp_block)


class MAInRampMap(_BaseMARampMap):
    RAMP_CLS = InRampOnStraight

    def _configure_scene(self, first_block, ramp_block):
        self.spawn_roads = list(first_block.get_respawn_roads())
        self.spawn_roads.extend(ramp_block.get_respawn_roads())
        self.destination_node = ramp_block.get_socket(0).positive_road.end_node


class MAOutRampMap(_BaseMARampMap):
    RAMP_CLS = OutRampOnStraight

    def _configure_scene(self, first_block, ramp_block):
        self.spawn_roads = list(first_block.get_respawn_roads())
        socket_road = ramp_block.get_socket(0).positive_road
        terminal_roads = _find_terminal_positive_roads(ramp_block, excluded_roads=[socket_road])
        assert len(terminal_roads) == 1, f"Expected one out-ramp exit road, got {len(terminal_roads)}"
        self.destination_node = terminal_roads[0].end_node


class RampSpawnManager(SpawnManager):
    def __init__(self):
        BaseManager.__init__(self)
        self.initialized = True
        self.num_agents = self.engine.global_config["num_agents"]
        self.exit_length = (self.engine.global_config["map_config"]["exit_length"] - FirstPGBlock.ENTRANCE_LENGTH)
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
        assert current_map is not None, "Ramp map is not initialized before spawn manager reset"
        roads = list(getattr(current_map, "spawn_roads", []))
        assert len(roads) > 0, "No spawn roads found for ramp map"
        return roads

    def _auto_fill_spawn_roads_with_real_lanes(self, spawn_roads):
        current_map = self._get_current_map()
        assert current_map is not None, "Ramp map is not initialized before computing spawn slots"
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
                for j in range(num_slots):
                    long = 1 / 2 * self.RESPAWN_REGION_LONGITUDE + j * self.RESPAWN_REGION_LONGITUDE
                    long = min(long, lane_length - 1.0)
                    lane_tuple = road.lane_index(lane_idx)
                    agent_configs.append(
                        Config(
                            dict(
                                identifier="|".join((str(s) for s in lane_tuple + (j, ))),
                                config={
                                    "spawn_lane_index": lane_tuple,
                                    "spawn_longitude": long,
                                    "spawn_lateral": 0,
                                },
                            ),
                            unchangeable=True,
                        )
                    )
                    if j == 0:
                        safe_spawn_places.append(copy.deepcopy(agent_configs[-1]))

        return agent_configs, safe_spawn_places

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
        return super(RampSpawnManager, self).reset()

    def update_destination_for(self, vehicle_id, vehicle_config):
        current_map = self._get_current_map()
        assert current_map is not None, "Ramp map is not initialized before destination assignment"
        vehicle_config["destination"] = current_map.destination_node
        return vehicle_config


class _BaseRampPGMapManager(PGMapManager):
    MAP_CLS = None

    def reset(self):
        config = self.engine.global_config.copy()
        current_seed = self.engine.global_seed

        if self.maps[current_seed] is None:
            map_config = config["map_config"]
            map_config.update({"seed": current_seed})
            _map = self.spawn_object(self.MAP_CLS, map_config=map_config, random_seed=None)
            self.load_map(_map)
            if self.engine.global_config["store_map"]:
                self.maps[current_seed] = _map
        else:
            _map = self.maps[current_seed]
            self.load_map(_map)

        self.current_map.configure_runtime_scene()
        self.current_map.spawn_roads = list(getattr(self.current_map, "spawn_roads", []))
        self.engine.global_config["spawn_roads"] = copy.deepcopy(self.current_map.spawn_roads)


class MAInRampPGMapManager(_BaseRampPGMapManager):
    MAP_CLS = MAInRampMap


class MAOutRampPGMapManager(_BaseRampPGMapManager):
    MAP_CLS = MAOutRampMap


class MultiAgentInRampEnv(MultiAgentMetaDrive):
    @staticmethod
    def default_config() -> Config:
        return MultiAgentMetaDrive.default_config().update(copy.deepcopy(MAInRampConfig), allow_add_new_key=True)

    def setup_engine(self):
        super(MultiAgentInRampEnv, self).setup_engine()
        self.engine.update_manager("map_manager", MAInRampPGMapManager())
        self.engine.update_manager("spawn_manager", RampSpawnManager())


class MultiAgentOutRampEnv(MultiAgentMetaDrive):
    @staticmethod
    def default_config() -> Config:
        return MultiAgentMetaDrive.default_config().update(copy.deepcopy(MAOutRampConfig), allow_add_new_key=True)

    def setup_engine(self):
        super(MultiAgentOutRampEnv, self).setup_engine()
        self.engine.update_manager("map_manager", MAOutRampPGMapManager())
        self.engine.update_manager("spawn_manager", RampSpawnManager())


class MultiAgentRampEnv(MultiAgentInRampEnv):
    """Backward-compatible alias for the in-ramp scene."""
