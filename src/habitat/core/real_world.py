#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import cv2

from collections import OrderedDict
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Union

import attr
from gym import Space
from gym.spaces.dict_space import Dict as SpaceDict

from habitat.config import Config
from habitat.core.dataset import Episode


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
    

@attr.s(auto_attribs=True)
class ActionSpaceConfiguration:
    config: Config

    def get(self):
        raise NotImplementedError


class SensorTypes(Enum):
    r"""Enumeration of types of sensors.
    """

    NULL = 0
    COLOR = 1
    DEPTH = 2
    NORMAL = 3
    SEMANTIC = 4
    PATH = 5
    POSITION = 6
    FORCE = 7
    TENSOR = 8
    TEXT = 9
    MEASUREMENT = 10
    HEADING = 11
    TACTILE = 12
    TOKEN_IDS = 13


class Sensor:
    r"""Represents a sensor that provides data from the environment to agent.

    :data uuid: universally unique id.
    :data sensor_type: type of Sensor, use SensorTypes enum if your sensor
        comes under one of it's categories.
    :data observation_space: ``gym.Space`` object corresponding to observation
        of sensor.

    The user of this class needs to implement the get_observation method and
    the user is also required to set the below attributes:
    """

    uuid: str
    config: Config
    sensor_type: SensorTypes
    observation_space: Space

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.config = kwargs["config"] if "config" in kwargs else None
        self.uuid = self._get_uuid(*args, **kwargs)
        self.sensor_type = self._get_sensor_type(*args, **kwargs)
        self.observation_space = self._get_observation_space(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        raise NotImplementedError

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        raise NotImplementedError

    def get_observation(self) -> Any:
        r"""
        Returns:
            current observation for Sensor.
        """
        raise NotImplementedError


class Observations(dict):
    r"""Dictionary containing sensor observations
    """

    def __init__(self, sensors: Dict[str, Sensor]) -> None:
        """Constructor

        :param sensors: list of sensors whose observations are fetched and
            packaged.
        """

        data = [
            (uuid, sensor.get_observation())
            for uuid, sensor in sensors.items()
        ]
        super().__init__(data)


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

class SensorSuite:
    r"""Represents a set of sensors, with each sensor being identified
    through a unique id.
    """

    sensors: Dict[str, Sensor]
    observation_spaces: SpaceDict

    def __init__(self, sensors: Iterable[Sensor]) -> None:
        """Constructor

        :param sensors: list containing sensors for the environment, uuid of
            each sensor must be unique.
        """
        self.sensors = OrderedDict()
        spaces: OrderedDict[str, Space] = OrderedDict()
        for sensor in sensors:
            assert (
                sensor.uuid not in self.sensors
            ), "'{}' is duplicated sensor uuid".format(sensor.uuid)
            self.sensors[sensor.uuid] = sensor
            spaces[sensor.uuid] = sensor.observation_space
        self.observation_spaces = SpaceDict(spaces=spaces)

    def get(self, uuid: str) -> Sensor:
        return self.sensors[uuid]

    def get_observations(self) -> Observations:
        r"""Collects data from all sensors and returns it packaged inside
            `Observations`.
        """
        return Observations(self.sensors)


@attr.s(auto_attribs=True)
class AgentState:
    position: List[float]
    rotation: Optional[List[float]] = None


@attr.s(auto_attribs=True)
class ShortestPathPoint:
    position: List[Any]
    rotation: List[Any]
    action: Optional[int] = None


class RealWorld:
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

    def step(self, action, *args, **kwargs) -> Observations:
        # actionを受け取ってカチャカを動かす
        sim_obs = self._sim.step(action)
        observations = self._sensor_suite.get_observations()
        return observations


    def seed(self, seed: int) -> None:
        raise NotImplementedError

    def reconfigure(self, config: Config) -> None:
        raise NotImplementedError

    def geodesic_distance(
        self,
        position_a: List[float],
        position_b: Union[List[float], List[List[float]]],
        episode: Optional[Episode] = None,
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
        raise NotImplementedError

    def get_agent_state(self, agent_id: int = 0):
        r"""..

        :param agent_id: id of agent.
        :return: state of agent corresponding to :p:`agent_id`.
        """
        raise NotImplementedError

    def get_observations_at(
        self,
        position: List[float],
        rotation: List[float],
        keep_agent_at_new_pose: bool = False,
    ) -> Optional[Observations]:
        """Returns the observation.

        :param position: list containing 3 entries for :py:`(x, y, z)`.
        :param rotation: list with 4 entries for :py:`(x, y, z, w)` elements
            of unit quaternion (versor) representing agent 3D orientation,
            (https://en.wikipedia.org/wiki/Versor)
        :param keep_agent_at_new_pose: If true, the agent will stay at the
            requested location. Otherwise it will return to where it started.
        :return:
            The observations or :py:`None` if it was unable to get valid
            observations.

        """
        raise NotImplementedError

    def action_space_shortest_path(
        self, source: AgentState, targets: List[AgentState], agent_id: int = 0
    ) -> List[ShortestPathPoint]:
        r"""Calculates the shortest path between source and target agent
        states.

        :param source: source agent state for shortest path calculation.
        :param targets: target agent state(s) for shortest path calculation.
        :param agent_id: id for agent (relevant for multi-agent setup).
        :return: list of agent states and actions along the shortest path from
            source to the nearest target (both included).
        """
        raise NotImplementedError

    def get_straight_shortest_path_points(
        self, position_a: List[float], position_b: List[float]
    ) -> List[List[float]]:
        r"""Returns points along the geodesic (shortest) path between two
        points irrespective of the angles between the waypoints.

        :param position_a: the start point. This will be the first point in
            the returned list.
        :param position_b: the end point. This will be the last point in the
            returned list.
        :return: a list of waypoints :py:`(x, y, z)` on the geodesic path
            between the two points.
        """

        raise NotImplementedError

    @property
    def up_vector(self):
        r"""The vector representing the direction upward (perpendicular to the
        floor) from the global coordinate frame.
        """
        raise NotImplementedError

    @property
    def forward_vector(self):
        r"""The forward direction in the global coordinate frame i.e. the
        direction of forward movement for an agent with 0 degrees rotation in
        the ground plane.
        """
        raise NotImplementedError

    def render(self, mode: str = "rgb") -> Any:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def previous_step_collided(self) -> bool:
        r"""Whether or not the previous step resulted in a collision

        :return: :py:`True` if the previous step resulted in a collision,
            :py:`False` otherwise
        """
        raise NotImplementedError
