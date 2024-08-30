import time
import carla
import logging
import random
from multiprocessing.synchronize import Event
from typing import Optional


def spawn_npcs(
        client: Optional[carla.Client]=None,
        host: Optional[str]="localhost",
        port: Optional[int]=2000,
        num_vehicles: int=10, 
        num_walkers: int=50, 
        safe: bool=True, 
        filterv: str="vehicle.*", 
        filterw: str="walker.pedestrian.*",
        tm_port: int=8000,
        sync: bool=True,
        rl_training: bool=False,
        spawn_event: Optional[Event]=None
    ):

    if not client:
        if (not host) or (not port):
            raise ValueError("host and port is required if client is NoneType")
        client = carla.Client(host, port)

    vehicles_list = []
    walkers_list = []
    all_id = []
    try:
        traffic_manager = client.get_trafficmanager(tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        world = client.get_world()
        synchronous_master = False

        if sync or rl_training:
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                world.apply_settings(settings)
            else:
                synchronous_master = False

        blueprints = world.get_blueprint_library().filter(filterv)
        blueprintsWalkers = world.get_blueprint_library().filter(filterw)

        if safe:
            blueprints = [x for x in blueprints if int(x.get_attribute("number_of_wheels")) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith("isetta")]
            blueprints = [x for x in blueprints if not x.id.endswith("carlacola")]
            blueprints = [x for x in blueprints if not x.id.endswith("cybertruck")]
            blueprints = [x for x in blueprints if not x.id.endswith("t2")]

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if num_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif num_vehicles > number_of_spawn_points:
            msg = "requested %d vehicles, but could only find %d spawn points"
            logging.warning(msg, num_vehicles, number_of_spawn_points)
            num_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= num_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute("color"):
                color = random.choice(blueprint.get_attribute("color").recommended_values)
                blueprint.set_attribute("color", color)
            if blueprint.has_attribute("driver_id"):
                driver_id = random.choice(blueprint.get_attribute("driver_id").recommended_values)
                blueprint.set_attribute("driver_id", driver_id)
            blueprint.set_attribute("role_name", "autopilot")
            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(num_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute("is_invincible"):
                walker_bp.set_attribute("is_invincible", "false")
            # set the max speed
            if walker_bp.has_attribute("speed"):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute("speed").recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute("speed").recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find("controller.ai.walker")
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if not rl_training:
            if not sync or not synchronous_master:
                world.wait_for_tick()
            else:
                world.tick()
        else:
            world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        print("spawned %d vehicles and %d walkers, press Ctrl+C to exit." % (len(vehicles_list), len(walkers_list)))

        # example of how to use parameters
        traffic_manager.global_percentage_speed_difference(30.0)

        while True:
            if spawn_event and (not spawn_event.is_set()):
                spawn_event.set()

            if not rl_training:
                if sync and synchronous_master:
                    world.tick()
                else:
                    world.wait_for_tick()
            else:
                time.sleep(0.2)
                continue

    finally:
        if sync and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        print("\ndestroying %d vehicles" % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        print("\ndestroying %d walkers" % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

        time.sleep(0.5)


if __name__ == "__main__":
    client = carla.Client("localhost", 2000)
    spawn_npcs(client)
