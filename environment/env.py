import carla
import gym
import logging
import time
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from .utils import *
from .tools.render import CarlaEnvRender
from .navigation.local_planner import RoadOption
from .navigation.global_route_planner import GlobalRoutePlanner
from .navigation.global_route_planner_dao import GlobalRoutePlannerDAO

REWARD_PER_COLLISION = -5
REWARD_PER_LANE_INVASION = -2
ROAD_OPTION_MANEUVERS = [
    RoadOption.VOID,
    RoadOption.LEFT,
    RoadOption.RIGHT,
    RoadOption.STRAIGHT,
    RoadOption.LANEFOLLOW,
    RoadOption.CHANGELANELEFT,
    RoadOption.CHANGELANERIGHT
]

logger = logging.getLogger(__name__)

class CarlaEnv(gym.Env):
    def __init__(
            self,
            client: carla.Client,
            cam_w: int=224, 
            cam_h: int=224,
            fov: int=110,
            fps: int=15,
            filterv: str="model3",
            action_eps: float=0.05,
            max_travel_distance: float=2500,
            min_speed: float=15.0,
            max_speed: float=60.0,
            target_speed: float=25.0,
            max_center_deviation: float=3.0,
            max_center_angle_deviation: float=90,
            terminal_on_collision: bool=True,
            terminal_on_lane_invasion: bool=True,
            terminal_on_max_speed: bool=True,
            terminal_on_max_angle_deviation: bool=False,
            terminal_on_max_center_deviation: bool=True,
            render_scale: Optional[float]=None
        ):
        assert max_speed > target_speed > min_speed > 0

        self._client = client
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.fov = fov
        self.fps = fps
        self.filterv = filterv
        self.action_eps = action_eps
        self.max_travel_distance = max_travel_distance # m
        self.min_speed = min_speed # km/hr
        self.max_speed = max_speed # km/hr
        self.target_speed = target_speed # km/hr
        self.max_center_deviation = max_center_deviation # m
        self.max_center_angle_deviation = max_center_angle_deviation # degrees
        self.terminal_on_collision = terminal_on_collision
        self.terminal_on_lane_invasion = terminal_on_lane_invasion
        self.terminal_on_max_speed = terminal_on_max_speed
        self.terminal_on_max_angle_deviation = terminal_on_max_angle_deviation
        self.terminal_on_max_center_deviation = terminal_on_max_center_deviation

        self._terminal_allowance_steps = 40
        self.closed = False
        self._world = self._client.get_world()
        self._make_world_synchronous()
        self._blueprint_library = self._world.get_blueprint_library()
        self.vehicle_blueprint = random.choice(self._blueprint_library.filter(self.filterv))
        self._map = self._world.get_map()
        self._actors = []
        self.observation_space = {
            "cam_obs": gym.spaces.Box(
                np.zeros([self.cam_h, self.cam_w, 3], dtype=np.float32), 
                np.ones([self.cam_h, self.cam_w, 3], dtype=np.float32), 
                dtype=np.float32
            ),
            #[linear-velocity]
            "measurements": gym.spaces.Box(
                np.asarray([-np.inf], dtype=np.float32),
                np.asarray([np.inf], dtype=np.float32), 
                dtype=np.float32
            ),
            "intention": gym.spaces.Discrete(len(ROAD_OPTION_MANEUVERS))
        }
        # [steer, throttle]
        self.action_space = gym.spaces.Box(
            np.asarray([-1, 0]),
            np.asarray([1, 1]),
            dtype=np.float32
        )
        self.renderer = CarlaEnvRender(self._world, world_scale=render_scale)
        self._spectator_cam_h = self.renderer.main_height
        self._spectator_cam_w = self.renderer.main_width + self.renderer.side_panel_width


    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        self.in_simulation_time = 0
        self.terminal_state = False
        self.num_timesteps = 0
        self.waypoint_idx = 0
        self.prev_waypoint_idx = 0
        self.distance_traveled = 0
        self.collision_reward = 0
        self.lane_invasion_reward = 0
        self.terminal_reason = "none"
        self.all_rewards = np.asarray([])

        spawnable_position = False
        npc_actors = self._world.get_actors()
        while not spawnable_position:
            # in this loop, we verify if the randomly sampled location to spawn our actor vehicle
            # does not have any other (npc) actors in it so as to avoid collision when spawning

            # generate spawn point and route waypoints
            spawn_points = self._map.get_spawn_points()
            start_point, end_point = np.random.choice(spawn_points, size=2, replace=False)
            self.start_waypoint = self._map.get_waypoint(start_point.location)
            self.end_waypoint = self._map.get_waypoint(end_point.location)
            self.route_waypoints = CarlaEnv.waypoint_planner(self._map, self.start_waypoint, self.end_waypoint)

            # create vehicle blueprint
            if (not hasattr(self, "vehicle")) or (not self.vehicle):
                self.vehicle = self._world.try_spawn_actor(self.vehicle_blueprint, start_point)
                if self.vehicle:
                    spawnable_position = True
                else:
                    continue
                self._actors.append(self.vehicle)
                # attach sensors
                self._attach_camera_to_vehicle()
                self._attach_collision_sensor_to_vehicle()
                self._attach_lane_invasion_sensor_to_vehicle()
            else:
                spawnable_position = all([
                    (npc_actors[i].get_location().distance(start_point.location) > 2.0)
                    for i in range(0, len(npc_actors))
                ])
                if spawnable_position:
                    # apply zero control and reset vehicles physics to that vehicle can be teleported
                    # to specific location without being subjected to the forces of the environment
                    self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))
                    self.vehicle.set_simulate_physics(False)
                    self.vehicle.set_transform(start_point)
                    self.vehicle.set_simulate_physics(True)
                else:
                    continue

        # make and locate spectator
        if not hasattr(self, "spectator") or self.spectator is None:
            self.spectator = self._world.get_spectator()
            self._attach_camera_to_spectator()
        self.set_spectator_transform()

        # progress the world by one timstep
        self._world.tick()

        # necessary incase camera sensor / obs has not yet been registered
        while (not hasattr(self, "cam_obs")) or (not hasattr(self, "spectator_cam_obs")):
            time.sleep(0.05)

        # init other values
        self.prev_position = to_vector(self.vehicle.get_location())
        self.prev_rotation = to_vector(self.vehicle.get_transform().rotation)
        self.prev_velocity = to_vector(self.vehicle.get_velocity())

        # call step function with no action, retrieve first observation
        obs, info, *_ = self.step(None)
        return obs, info
    

    def step(self, action: Optional[np.ndarray]=None) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, np.ndarray]]:
        route_length = len(self.route_waypoints)

        if action is not None:
            action = action.reshape(-1)
            assert action.ndim == 1 and action.shape[0] == self.action_space.shape[0]

            # check if end of waypoints sequence has been reached
            if self.waypoint_idx >= route_length-1:
                self.terminal_reason = "last waypoint reached"
                self.terminal_state = True

            # update previous readings
            self.prev_cam_obs = self.cam_obs

            # clip and smoothen action with previous action if available
            steer, throttle = action
            prev_control = self.vehicle.get_control()
            steer = smoothen_action(prev_control.steer, steer, smooth_factor=self.action_eps)
            throttle = smoothen_action(prev_control.throttle, throttle, smooth_factor=self.action_eps)
            # move vehicle with action
            # throttle values range from 0 to 1, while steer, from -1 to 1
            control = carla.VehicleControl(brake=0.0, steer=float(steer), throttle=float(throttle))
            self.vehicle.apply_control(control)

            # update number of timesteps made with action
            self.num_timesteps += 1

        self.set_spectator_transform()
        
        # progress world by one timstep
        self._world.tick()
        vtransform = self.vehicle.get_transform()
        # track closest waypoint in the route and tell us the index of the waypoint we have passed
        # when vehicle is spawned at self.route_waypoints[0], it technically implies that you're at
        # waypoint_idx=0, by performing an initial action and applying the tick, the vehicle has
        # moved from waypoint-0, hence the loop below checks if you have passed subsequent waypoints, 
        # starting from waypoint_idx + 1
        self.prev_waypoint_idx = self.waypoint_idx
        for _ in range(route_length):
            wp, _ = self.route_waypoints[(self.waypoint_idx + 1) % route_length]
            # check if we have passed the waypoint (wp)
            # dot product measures the amount at which two vectors are heading to same direction
            # here we compute the relative direction of the waypoint to the vehicle, next we compute
            # the dot product of that direction to the forward direction of the waypoint, if the vehicle
            # has passed the waypoint, the dot product will be positive non-zero value, else
            # it will be negative
            wp_direction = to_vector(wp.transform.get_forward_vector())
            vehicle_relative_direction = to_vector(vtransform.location - wp.transform.location)
            direction_dot_prod = np.dot(wp_direction[:2], vehicle_relative_direction[:2])
            if direction_dot_prod > 0:
                self.waypoint_idx += 1
            else: break

        # compute reward fn
        reward = self.reward_func()
        # collect data to be returned
        waypoint, _ = self.route_waypoints[self.waypoint_idx % route_length]
        next_waypoint, next_maneuver = self.route_waypoints[(self.waypoint_idx + 1) % route_length]

        velocity = to_vector(self.vehicle.get_velocity())
        speed = 3.6 * np.sqrt((velocity**2).sum())

        # make state terminal if vehicle has been immobile for the first 5secs of episode
        if speed < 1.0 and self.waypoint_idx >= 0 and self.in_simulation_time > 5:
            self.terminal_state = True
            self.terminal_reason = "vehicle has stopped"

        obs = {
            "cam_obs": self.cam_obs,
            "measurements": np.asarray([speed, ], dtype=np.float32) / 100, # km/hr speed scaled by 100
            "intention": CarlaEnv.label_encode_maneuver(next_maneuver),
        }
        info = {
            "vpos": to_vector(vtransform.location),
            "vrot": to_vector(vtransform.rotation),
            "vvel": to_vector(self.vehicle.get_velocity()),
            "wpos": to_vector(waypoint.transform.location),
            "wrot": to_vector(waypoint.transform.rotation),
            "next_wpos": to_vector(next_waypoint.transform.location),
            "next_wrot": to_vector(next_waypoint.transform.rotation),
            "all_rewards": self.all_rewards,
            "closed": self.closed
        }

        # accumulate distance traveled
        self.distance_traveled += carla.Location(*self.prev_position).distance(vtransform.location)
        self.prev_position = to_vector(vtransform.location)
        self.prev_rotation = to_vector(vtransform.rotation)
        self.prev_velocity = velocity

        # check distance traveled
        if self.distance_traveled >= self.max_travel_distance:
            self.terminal_reason = (
                f"vehicle has traveled a distance of {self.distance_traveled/1000 :.2f}km"
            )
            self.terminal_state = True

        # update the insimulation time (not the actual simulation time observed by the client)
        self.in_simulation_time += (1 / self.fps)
        return obs, reward, self.terminal_state, info


    def set_spectator_transform(self):
        self.spectator.set_transform(self.vehicle.get_transform())


    def _make_world_synchronous(self):
        settings = self._world.get_settings()
        # a thing to note is that the FPS parameter does not determine the actual FPS of the simulation, rather
        # is an 'in-simulation' parameter, used to ensure that the time between each timestep is recorded as the
        # inverse of the FPS, rather than the actual time spent
        settings.fixed_delta_seconds = 1 / self.fps
        settings.synchronous_mode = True
        self._world.apply_settings(settings)


    def _attach_camera_to_vehicle(self):
        camera_bp = self._blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x", f"{self.cam_w}")
        camera_bp.set_attribute("image_size_y", f"{self.cam_h}")
        camera_bp.set_attribute("fov", f"{self.fov}")
        transform = carla.Transform(carla.Location(x=1.6, z=1.7))
        self.camera = self._world.spawn_actor(camera_bp, transform, attach_to=self.vehicle) 
        self._actors.append(self.camera)
        self.camera.listen(self._handle_camera_data)

    
    def _attach_camera_to_spectator(self):
        camera_bp = self._blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x", f"{self._spectator_cam_w}")
        camera_bp.set_attribute("image_size_y", f"{self._spectator_cam_h}")
        camera_bp.set_attribute("fov", "110")
        transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        self.spectator_camera = self._world.spawn_actor(
            camera_bp, transform, attach_to=self.spectator
        )
        self._actors.append(self.spectator_camera)
        self.spectator_camera.listen(self._handle_spectator_camera_data)


    def _attach_collision_sensor_to_vehicle(self):
        collision_bp = self._blueprint_library.find("sensor.other.collision")
        spawn_point = self.vehicle.get_transform()
        self.collision_sensor = self._world.spawn_actor(collision_bp, spawn_point, attach_to=self.vehicle)
        self._actors.append(self.collision_sensor)
        self.collision_sensor.listen(self._handle_collision_data)


    def _attach_lane_invasion_sensor_to_vehicle(self):
        lane_invasion_bp = self._blueprint_library.find("sensor.other.lane_invasion")
        spawn_point = self.vehicle.get_transform()
        self.lane_invasion_sensor = self._world.spawn_actor(lane_invasion_bp, spawn_point, attach_to=self.vehicle)
        self._actors.append(self.lane_invasion_sensor)
        self.lane_invasion_sensor.listen(self._handle_lane_invasion_data)


    def _handle_camera_data(self, data: carla.libcarla.Image):
        img = np.asarray(data.raw_data)
        img = img.reshape(self.cam_h, self.cam_w, 4)
        self.cam_obs = img[..., :-1]
        
    
    def _handle_spectator_camera_data(self, data: carla.libcarla.Image):
        img = np.asarray(data.raw_data)
        img = img.reshape(self._spectator_cam_h, self._spectator_cam_w, 4)
        self.spectator_cam_obs = img[..., :-1]


    def _handle_collision_data(self, data: carla.CollisionEvent):
        if get_actor_display_name(data.other_actor) != "Road":
            self.collision_reward += REWARD_PER_COLLISION
            if self.terminal_on_collision:
                self.terminal_reason = "collision"
                self.terminal_state = True


    def _handle_lane_invasion_data(self, data: carla.LaneInvasionEvent):
        invalid_lanes = [
            carla.LaneMarkingType.Solid, 
            carla.LaneMarkingType.SolidSolid,
            carla.LaneMarkingType.Grass,
            carla.LaneMarkingType.Curb
        ]
        for marking in data.crossed_lane_markings:
            if marking.type in invalid_lanes:
                self.lane_invasion_reward += REWARD_PER_LANE_INVASION
                if self.terminal_on_lane_invasion:
                    self.terminal_reason = "lane invasion"
                    self.terminal_state = True


    def _speed_reward_func(self) -> float:
        route_length = len(self.route_waypoints)
        vvelocity = self.vehicle.get_velocity()
        target_speed = self.vehicle.get_speed_limit() or self.target_speed
        # speed reward will be linearly interpolated between 0 and 1
        speed = 3.6 * np.sqrt((to_vector(vvelocity)**2).sum())
        if speed < self.min_speed:
            reward = speed / self.min_speed
        elif speed > target_speed:
            if speed > self.max_speed:
                if self.terminal_on_max_speed and self.num_timesteps > self._terminal_allowance_steps:
                    self.terminal_reason = "overspeeding (max speed exceeded)"
                    self.terminal_state = True
            reward = 1.0 - ((speed - target_speed) / (self.max_speed - target_speed))
        else:
            reward = 1.0
        current_waypoint, _ = self.route_waypoints[self.waypoint_idx % route_length]
        next_waypoint, _ = self.route_waypoints[(self.waypoint_idx + 1) % route_length]
        center_deviation = distance_to_line(
            to_vector(current_waypoint.transform.location),
            to_vector(next_waypoint.transform.location),
            to_vector(self.vehicle.get_location())
        )
        if center_deviation > self.max_center_deviation:
            if self.terminal_on_max_center_deviation and self.num_timesteps > self._terminal_allowance_steps:
                self.terminal_reason = "over-deviation of vehicle from center of lane"
                self.terminal_state = True
        deviation_factor = max(1.0 - center_deviation / self.max_center_deviation, 0.0)
        reward = reward * deviation_factor
        return reward


    def _deviation_reward_func(self) -> float:
        route_length = len(self.route_waypoints)
        current_waypoint, _ = self.route_waypoints[self.waypoint_idx % route_length]
        wdirection = current_waypoint.transform.get_forward_vector()
        vvelocity = self.vehicle.get_velocity()
        deviation_angle = radian_angle_diff(to_vector(wdirection), to_vector(vvelocity))
        deviation_angle = abs(np.rad2deg(deviation_angle))
        
        if deviation_angle > self.max_center_angle_deviation:
            if self.terminal_on_max_angle_deviation and self.num_timesteps > self._terminal_allowance_steps:
                self.terminal_reason = "over-deviation of vehicle angle from waypoint"
                self.terminal_state = True

        angle_factor = max(1.0 - deviation_angle / self.max_center_angle_deviation, 0.0)
        reward = (self.waypoint_idx - self.prev_waypoint_idx) * angle_factor
        return reward


    def reward_func(self) -> float:
        speed_reward = self._speed_reward_func()
        deviation_reward = self._deviation_reward_func()
        main_reward = (speed_reward + deviation_reward) / 2
        self.all_rewards = np.asarray([
            speed_reward,
            deviation_reward,
            self.collision_reward,
            self.lane_invasion_reward,
        ])
        reward = (
            main_reward + 
            self.collision_reward + 
            self.lane_invasion_reward
        )
        # reset collision and lane invasion rewards because they typically only apply per episode
        self.collision_reward, self.lane_invasion_reward = 0, 0
        return reward


    def render(self):
        if not self.terminal_state:
            return self.renderer.render(
                self.vehicle, 
                self.spectator_cam_obs, 
                self.cam_obs , 
                self.terminal_reason
            )


    def close_render(self):
        return self.renderer.close()


    def close(self):
        self.renderer.close()
        settings = self._world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self._world.apply_settings(settings)

        for actor in self._actors:
            actor.destroy()
        self.closed = True
        

    @staticmethod
    def waypoint_planner(
        map, 
        start_waypoint: carla.Waypoint, 
        end_waypoint: carla.Waypoint, 
        sampling_resolution: int=1.0
    ) -> List[Tuple[carla.Waypoint, RoadOption]]:
        dao = GlobalRoutePlannerDAO(map, sampling_resolution=sampling_resolution)
        grp = GlobalRoutePlanner(dao)
        grp.setup()
        route = grp.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location
        )
        return route


    @staticmethod
    def label_encode_maneuver(maneuver: RoadOption) -> np.int64:
        return np.int64(ROAD_OPTION_MANEUVERS.index(maneuver))
    

    @classmethod
    def make_env_with_client(cls, uri: str, port: int, **kwargs) -> "CarlaEnv":
        client = carla.Client(uri, port)
        client.set_timeout(10)
        return cls(client, **kwargs)
