import copy

from metadrive.component.map.pg_map import PGMap
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.std_t_intersection import StdTInterSection
from metadrive.component.road_network import Road
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.manager.base_manager import BaseManager
from metadrive.manager.pg_map_manager import PGMapManager
from metadrive.manager.spawn_manager import SpawnManager
from metadrive.utils import Config

MATIntersectionConfig = dict(
    num_agents=12,
    map_config=dict(
        exit_length=60,
        lane_num=2,
        t_type=0,
    ),
    top_down_camera_initial_x=78,
    top_down_camera_initial_y=0,
    top_down_camera_initial_z=115,
)


class MATIntersectionMap(PGMap):
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

        t_block = StdTInterSection(
            1,
            first_block.get_socket(index=0),
            self.road_network,
            random_seed=1,
            ignore_intersection_checking=False,
        )

        extra_config = {}
        for key in ("radius", "t_type", "decrease_increase"):
            if key in self.config and self.config[key] is not None:
                extra_config[key] = self.config[key]

        t_block.construct_block(parent_node_path, physics_world, extra_config=extra_config or None)
        self.blocks.append(t_block)


class TIntersectionSpawnManager(SpawnManager):
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

    def _get_spawn_roads(self):
        current_map = self.engine.current_map
        roads = list(current_map.blocks[0].get_respawn_roads())
        roads.extend(current_map.blocks[1].get_respawn_roads())
        assert len(roads) > 0, "No spawn roads found for T-intersection map"
        return roads

    def reset(self):
        spawn_roads = self._get_spawn_roads()
        agent_configs, safe_spawn_places = self._auto_fill_spawn_roads_randomly(spawn_roads)
        self.available_agent_configs = agent_configs
        self.safe_spawn_places = {place["identifier"]: place for place in safe_spawn_places}
        self.spawn_roads = spawn_roads
        self.engine.global_config["spawn_roads"] = copy.deepcopy(spawn_roads)
        self.engine.current_map.spawn_roads = spawn_roads
        return super(TIntersectionSpawnManager, self).reset()

    def update_destination_for(self, agent_id, vehicle_config):
        end_roads = copy.deepcopy(self.spawn_roads)
        start_road = Road(*vehicle_config["spawn_lane_index"][:2])
        end_roads = [road for road in end_roads if road != start_road]
        candidate_roads = end_roads if len(end_roads) > 0 else self.spawn_roads
        end_road = -self.np_random.choice(candidate_roads)
        vehicle_config["destination"] = end_road.end_node
        return vehicle_config


class MATIntersectionPGMapManager(PGMapManager):
    def reset(self):
        config = self.engine.global_config
        if len(self.spawned_objects) == 0:
            _map = self.spawn_object(MATIntersectionMap, map_config=config["map_config"], random_seed=None)
        else:
            assert len(self.spawned_objects) == 1, "It is supposed to contain one map in this manager"
            _map = self.spawned_objects.values()[0]
        self.load_map(_map)


class MultiAgentTIntersectionEnv(MultiAgentMetaDrive):
    @staticmethod
    def default_config() -> Config:
        return MultiAgentMetaDrive.default_config().update(MATIntersectionConfig, allow_add_new_key=True)

    def setup_engine(self):
        super(MultiAgentTIntersectionEnv, self).setup_engine()
        self.engine.update_manager("map_manager", MATIntersectionPGMapManager())
        self.engine.update_manager("spawn_manager", TIntersectionSpawnManager())
