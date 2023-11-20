#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import cv2
import math

from collections import OrderedDict
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Union

import attr
from gym import Space
from gym.spaces.dict_space import Dict as SpaceDict

import habitat_sim
from habitat.config import Config
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    Config,
    Observations,
    Sensor,
    SensorSuite,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
)


RGBSENSOR_DIMENSION = 3


def overwrite_config(config_from: Config, config_to: Any) -> None:
    r"""Takes Habitat-API config and Habitat-Sim config structures. Overwrites
     Habitat-Sim config with Habitat-API values, where a field name is present
     in lowercase. Mostly used to avoid `sim_cfg.field = hapi_cfg.FIELD` code.

    Args:
        config_from: Habitat-API config node.
        config_to: Habitat-Sim config structure.
    """

    def if_config_to_lower(config):
        if isinstance(config, Config):
            return {key.lower(): val for key, val in config.items()}
        else:
            return config

    for attr, value in config_from.items():
        if hasattr(config_to, attr.lower()):
            setattr(config_to, attr.lower(), if_config_to_lower(value))


def check_sim_obs(obs, sensor):
    assert obs is not None, (
        "Observation corresponding to {} not present in "
        "simulator's observations".format(sensor.uuid)
    )


class RGBSensor(Sensor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.sim_sensor_type = habitat_sim.SensorType.COLOR
        super().__init__(*args, **kwargs)
        self.cap = cv2.VideoCapture(1)
        
    def __exit__(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "rgb"

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return spaces.Box(
            low=0,
            high=np.iinfo(np.int64).max,
            shape=(self.config.HEIGHT, self.config.WIDTH),
            dtype=np.int64,
        )

    def get_observation(self) -> Any:
        ret, obs = cap.read()
        print("RGB observed")
        
        size = self.observation_space.shape
        obs = cv2.resize(obs, size)
        
        return obs


class DepthSensor(Sensor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.sim_sensor_type = habitat_sim.SensorType.DEPTH

        if config.NORMALIZE_DEPTH:
            self.min_depth_value = 0
            self.max_depth_value = 1
        else:
            self.min_depth_value = config.MIN_DEPTH
            self.max_depth_value = config.MAX_DEPTH
            
        super().__init__(*args, **kwargs)
        self.cap = cv2.VideoCapture(1)
        
    def __exit__(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "depth"

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.DEPTH

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return spaces.Box(
            low=self.min_depth_value,
            high=self.max_depth_value,
            shape=(self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.float32,
        )

    def get_observation(self):
        # あとで変更
        import numpy as np
        obs = np.random.rand(self.observation_space.shape)
        
        return obs


@registry.register_simulator(name="Real-v0")
class RealWorld(Simulator):
    # 実世界実験用のSimulatorクラス
    
    def __init__(self, config: Config) -> None:
        self.config = config
        agent_config = self._get_agent_config()

        sim_sensors = []
        for sensor_name in agent_config.SENSORS:
            sensor_cfg = getattr(self.config, sensor_name)
            sensor_type = registry.get_sensor(sensor_cfg.TYPE)

            assert sensor_type is not None, "invalid sensor type {}".format(
                sensor_cfg.TYPE
            )
            sim_sensors.append(sensor_type(sensor_cfg))

        self._sensor_suite = SensorSuite(sim_sensors)
        self.sim_config = self.create_sim_config(self._sensor_suite)
        self._action_space = spaces.Discrete(
            len(self.sim_config.agents[0].action_space)
        )
        
        
    def create_sim_config(
        self, _sensor_suite: SensorSuite
    ) -> habitat_sim.Configuration:
        sim_config = habitat_sim.SimulatorConfiguration()
        overwrite_config(
            config_from=self.config.HABITAT_SIM_V0, config_to=sim_config
        )
        agent_config = habitat_sim.AgentConfiguration()
        overwrite_config(
            config_from=self._get_agent_config(), config_to=agent_config
        )

        sensor_specifications = []
        for sensor in _sensor_suite.sensors.values():
            sim_sensor_cfg = habitat_sim.SensorSpec()
            overwrite_config(
                config_from=sensor.config, config_to=sim_sensor_cfg
            )
            sim_sensor_cfg.uuid = sensor.uuid
            sim_sensor_cfg.resolution = list(
                sensor.observation_space.shape[:2]
            )
            sim_sensor_cfg.parameters["hfov"] = str(sensor.config.HFOV)

            sim_sensor_cfg.sensor_type = sensor.sim_sensor_type  # type: ignore
            sim_sensor_cfg.gpu2gpu_transfer = (
                self.config.HABITAT_SIM_V0.GPU_GPU
            )
            sensor_specifications.append(sim_sensor_cfg)

        agent_config.sensor_specifications = sensor_specifications
        agent_config.action_space = registry.get_action_space_configuration(
            self.config.ACTION_SPACE_CONFIG
        )(self.config).get()

        return habitat_sim.Configuration(sim_config, [agent_config])

    @property
    def sensor_suite(self) -> SensorSuite:
        return self._sensor_suite

    @property
    def action_space(self) -> Space:
        return self._action_space
    
    def _update_agents_state(self) -> bool:
        is_updated = False
        for agent_id, _ in enumerate(self.config.AGENTS):
            agent_cfg = self._get_agent_config(agent_id)
            if agent_cfg.IS_SET_START_STATE:
                self.set_agent_state(
                    agent_cfg.START_POSITION,
                    agent_cfg.START_ROTATION,
                    agent_id,
                )
                is_updated = True

        return is_updated

    def reset(self) -> Observations:
        return self._sensor_suite.get_observations()

    def seed(self, seed: int) -> None:
        return

    def geodesic_distance(
        self,
        position_a: List[float],
        position_b: Union[List[float], List[List[float]]],
    ) -> float:
        r"""Calculates geodesic distance between two points.

        :param position_a: coordinates of first point.
        :param position_b: coordinates of second point or list of goal points
        coordinates.
        :param episode: The episode with these ends points.  This is used for shortest path computation caching
        :return:
            the geodesic distance in the cartesian space between points
            :p:`position_a` and :p:`position_b`, if no path is found between
            the points then `math.inf` is returned.
        """
        x_a, y_a = position_a
        x_b, y_b = position_b
        distance = math.sqrt((x_a-x_b)*(x_a-x_b) + (y_a-y_b)*(y_a-y_b))
        return distance

    def get_agent_state(self, agent_id: int = 0) -> habitat_sim.AgentState:
        assert agent_id == 0, "No support of multi agent in {} yet.".format(
            self.__class__.__name__
        )
        return self._sim.get_agent(agent_id).get_state()

    def _get_agent_config(self, agent_id: Optional[int] = None) -> Any:
        if agent_id is None:
            agent_id = self.config.DEFAULT_AGENT_ID
        agent_name = self.config.AGENTS[agent_id]
        agent_config = getattr(self.config, agent_name)
        return agent_config
    
    def set_agent_state(
        self,
        position: List[float],
        rotation: List[float],
        agent_id: int = 0,
        reset_sensors: bool = True,
    ) -> bool:
        r"""Sets agent state similar to initialize_agent, but without agents
        creation. On failure to place the agent in the proper position, it is
        moved back to its previous pose.

        Args:
            position: list containing 3 entries for (x, y, z).
            rotation: list with 4 entries for (x, y, z, w) elements of unit
                quaternion (versor) representing agent 3D orientation,
                (https://en.wikipedia.org/wiki/Versor)
            agent_id: int identification of agent from multiagent setup.
            reset_sensors: bool for if sensor changes (e.g. tilt) should be
                reset).

        Returns:
            True if the set was successful else moves the agent back to its
            original pose and returns false.
        """
        agent = self._sim.get_agent(agent_id)
        original_state = self.get_agent_state(agent_id)
        new_state = self.get_agent_state(agent_id)
        new_state.position = position
        new_state.rotation = rotation

        # NB: The agent state also contains the sensor states in _absolute_
        # coordinates. In order to set the agent's body to a specific
        # location and have the sensors follow, we must not provide any
        # state for the sensors. This will cause them to follow the agent's
        # body
        new_state.sensor_states = dict()

        agent.set_state(new_state, reset_sensors)

        if not self._check_agent_position(position, agent_id):
            agent.set_state(original_state, reset_sensors)
            return False
        return True
    
    def get_observations_at(
        self,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        keep_agent_at_new_pose: bool = False,
    ) -> Optional[Observations]:
        current_state = self.get_agent_state()
        if position is None or rotation is None:
            success = True
        else:
            success = self.set_agent_state(
                position, rotation, reset_sensors=False
            )

        if success:
            observations = self._sensor_suite.get_observations()
            if not keep_agent_at_new_pose:
                self.set_agent_state(
                    current_state.position,
                    current_state.rotation,
                    reset_sensors=False,
                )
            return observations
        else:
            return None