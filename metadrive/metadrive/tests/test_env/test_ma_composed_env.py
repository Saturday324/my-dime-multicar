from metadrive.component.road_network import Road
from metadrive.envs.marl_envs.marl_composed import MultiAgentComposedEnv


def test_ma_composed_env_spawn_and_destinations():
    env = MultiAgentComposedEnv(
        {
            "map": "XTXTS",
            "num_agents": 20,
            "num_scenarios": 1,
            "traffic_density": 0.0,
            "allow_respawn": False,
            "use_render": False,
        }
    )
    try:
        obs, _ = env.reset(seed=0)
        assert isinstance(obs, dict)

        spawn_roads = list(getattr(env.current_map, "spawn_roads", []))
        assert len(spawn_roads) > 1
        destination_nodes = {(-road).end_node for road in spawn_roads}

        used_spawn_roads = set()
        for vehicle in env.agents.values():
            spawn_lane_index = vehicle.config["spawn_lane_index"]
            start_road = Road(*spawn_lane_index[:2])
            used_spawn_roads.add((start_road.start_node, start_road.end_node))

            destination = vehicle.config.get("destination")
            assert destination in destination_nodes
            assert destination != start_road.start_node

        assert len(used_spawn_roads) > 1

        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            assert isinstance(obs, dict)
            assert isinstance(reward, dict)
            assert isinstance(terminated, dict)
            assert isinstance(truncated, dict)
            assert isinstance(info, dict)
            if terminated["__all__"] or truncated["__all__"]:
                break
    finally:
        env.close()
