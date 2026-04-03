import glob
import os
import sys
import time
import random
import queue
import numpy as np
import cv2
import argparse
import math

# Try to find the CARLA library
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# Image parameters
WIDTH = 800
HEIGHT = 600
BASE_SAVE_DIR = "./datasets/carla_data"

# Queues for storing images
rgb_queue = queue.Queue()
seg_queue = queue.Queue()

def process_rgb_image(image):
    rgb_queue.put(image)

def process_seg_image(image):
    seg_queue.put(image)

def get_weather_presets():
    # Return a list of weather presets for data diversity
    return [
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.CloudySunset,
        carla.WeatherParameters.HardRainNoon,
        carla.WeatherParameters.MidRainyNoon,
        carla.WeatherParameters.ClearSunset
    ]

def main():
    parser = argparse.ArgumentParser(description="CARLA Data Collection Script (1 Moving NPC + 5 Walkers)")
    parser.add_argument('--host', default='127.0.0.1', help='Server IP')
    parser.add_argument('-p', '--port', default=2000, type=int, help='TCP Port')
    parser.add_argument('-n', '--nb_images', default=1000, type=int, help='Total images to save for this map')
    parser.add_argument('--map', default='Town03', type=str, help='Target map to collect data from (e.g., Town03)')
    # Parameters for dynamic actors
    parser.add_argument('--npc', default=1, type=int, help='Number of moving NPC vehicles')
    parser.add_argument('--pedestrians', default=5, type=int, help='Number of walking pedestrians')
    args = parser.parse_args()

    # Create directories based on the map name
    map_save_dir = os.path.join(BASE_SAVE_DIR, args.map)
    rgb_dir = os.path.join(map_save_dir, "rgb")
    mask_dir = os.path.join(map_save_dir, "mask")
    
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    print(f"=== Starting data collection for {args.map} ===")
    print(f"Data will be saved to: {map_save_dir}")
    print(f"Configuration: {args.npc} Moving Vehicle(s), {args.pedestrians} Walking Pedestrian(s)")

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(300.0) # original 60
        
        # Check if requested map exists
        available_maps = client.get_available_maps()
        if not any(args.map in m for m in available_maps):
            print(f"Error: Map '{args.map}' is not available on the server.")
            return

        print(f"Loading map: {args.map}...")
        world = client.load_world(args.map)
        
        # Setup synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # Setup Traffic Manager
        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)
        tm.set_global_distance_to_leading_vehicle(2.0)
        
        blueprint_library = world.get_blueprint_library()
        ego_vehicle_bp = blueprint_library.filter('model3')[0]
        
        # Blueprints for NPCs
        npc_vehicle_bps = [bp for bp in blueprint_library.filter('vehicle.*') if int(bp.get_attribute('number_of_wheels')) == 4]
        walker_bps = blueprint_library.filter('walker.pedestrian.*')
        walker_controller_bp = blueprint_library.find('controller.ai.walker')
        
        weathers = get_weather_presets()
        images_per_weather = math.ceil(args.nb_images / len(weathers))
        
        print(f"Target: {args.nb_images} images.")
        print(f"Distribution: {len(weathers)} weathers, ~{images_per_weather} images per weather.")

        total_saved_images = 0
        global_frame_index = 0 

        for weather_idx, weather_params in enumerate(weathers):
            if total_saved_images >= args.nb_images:
                break 
                
            print(f"\n[{weather_idx+1}/{len(weathers)}] Applying weather: {weather_params}")
            world.set_weather(weather_params)
            
            images_current_weather = 0
            
            # Keep trying until we gather enough images for this weather
            while images_current_weather < images_per_weather and total_saved_images < args.nb_images:
                
                spawn_points = world.get_map().get_spawn_points()
                if not spawn_points:
                    print("Error: No spawn points found.")
                    break
                    
                # 1. Spawn Ego Vehicle
                ego_spawn_point = random.choice(spawn_points)
                vehicle = world.spawn_actor(ego_vehicle_bp, ego_spawn_point)
                
                if vehicle is None:
                    continue 
                    
                vehicle.set_autopilot(True, tm.get_port())
                actor_list = [vehicle]
                
                # 2. Spawn Moving NPC Vehicle(s)
                spawn_points.remove(ego_spawn_point) 
                random.shuffle(spawn_points)
                
                spawned_npcs = 0
                for npc_spawn_point in spawn_points:
                    if spawned_npcs >= args.npc:
                        break
                    npc_bp = random.choice(npc_vehicle_bps)
                    npc_vehicle = world.try_spawn_actor(npc_bp, npc_spawn_point)
                    
                    if npc_vehicle is not None:
                        # ENABLE AUTOPILOT for the NPC so it drives around
                        npc_vehicle.set_autopilot(True, tm.get_port())
                        actor_list.append(npc_vehicle)
                        spawned_npcs += 1
                        
                if spawned_npcs > 0:
                    print(f"   -> Spawned {spawned_npcs} moving NPC vehicle(s).")

                # 3. Spawn Walking Pedestrians
                spawned_walkers = 0
                walker_controllers = []
                
                # Wake up the Navigation Mesh
                for _ in range(3):
                    world.tick()
                
                max_spawn_attempts = args.pedestrians * 4
                attempts = 0
                
                while spawned_walkers < args.pedestrians and attempts < max_spawn_attempts:
                    attempts += 1
                    spawn_location = world.get_random_location_from_navigation()
                    if spawn_location is None:
                        continue
                        
                    spawn_point = carla.Transform(spawn_location)
                    walker_bp = random.choice(walker_bps)
                    
                    walker = world.try_spawn_actor(walker_bp, spawn_point)
                    if walker:
                        controller = world.try_spawn_actor(walker_controller_bp, carla.Transform(), attach_to=walker)
                        if controller:
                            actor_list.append(walker)
                            actor_list.append(controller)
                            walker_controllers.append(controller)
                            spawned_walkers += 1
                        else:
                            walker.destroy()

                if spawned_walkers > 0:
                    print(f"   -> Spawned {spawned_walkers} walking pedestrians (took {attempts} attempts).")
                
                # Start pedestrian controllers
                world.tick()
                for controller in walker_controllers:
                    controller.start()
                    destination = world.get_random_location_from_navigation()
                    if destination:
                        controller.go_to_location(destination)
                    controller.set_max_speed(1.2 + random.random())
                
                # Clear residual queues
                while not rgb_queue.empty(): rgb_queue.get()
                while not seg_queue.empty(): seg_queue.get()
                
                # 4. Spawn Sensors
                cam_bp = blueprint_library.find('sensor.camera.rgb')
                cam_bp.set_attribute('image_size_x', str(WIDTH))
                cam_bp.set_attribute('image_size_y', str(HEIGHT))
                cam_bp.set_attribute('fov', '90')
                cam_bp.set_attribute('sensor_tick', '0.0') 
                
                transform = carla.Transform(carla.Location(x=1.5, z=2.4))
                rgb_cam = world.spawn_actor(cam_bp, transform, attach_to=vehicle)
                actor_list.append(rgb_cam)
                rgb_cam.listen(process_rgb_image)

                seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
                seg_bp.set_attribute('image_size_x', str(WIDTH))
                seg_bp.set_attribute('image_size_y', str(HEIGHT))
                seg_bp.set_attribute('fov', '90')
                seg_bp.set_attribute('sensor_tick', '0.0')

                seg_cam = world.spawn_actor(seg_bp, transform, attach_to=vehicle)
                actor_list.append(seg_cam)
                seg_cam.listen(process_seg_image)
                
                col_bp = blueprint_library.find('sensor.other.collision')
                col_cam = world.spawn_actor(col_bp, carla.Transform(), attach_to=vehicle)
                actor_list.append(col_cam)
                
                has_collided = [False] 
                
                def on_collision(event):
                    has_collided[0] = True
                    print(f"\n[!] Collision detected with {event.other_actor.type_id}. Respawning...")
                
                col_cam.listen(on_collision)
                
                # Warm-up phase
                bad_spawn = False
                for _ in range(40):
                    world.tick()
                    if has_collided[0]:
                        bad_spawn = True
                        break
                        
                if bad_spawn:
                    print("--> Invalid spawn point or immediate collision. Discarding and respawning...")
                    for actor in reversed(actor_list):
                        if actor.is_alive:
                            if hasattr(actor, 'stop'): actor.stop()
                            actor.destroy()
                    continue 
                    
                # Collection phase
                frames_skipped = 0
                
                while images_current_weather < images_per_weather and total_saved_images < args.nb_images:
                    world.tick()
                    
                    if has_collided[0]:
                        break # Exit inner loop to respawn everything
                    
                    try:
                        rgb_img = rgb_queue.get(timeout=2.0)
                        seg_img = seg_queue.get(timeout=2.0)
                    except queue.Empty:
                        continue
                        
                    # Save 1 image every 20 frames (1 FPS)
                    frames_skipped += 1
                    if frames_skipped >= 20:
                        frames_skipped = 0
                        
                        # Process RGB
                        array_rgb = np.frombuffer(rgb_img.raw_data, dtype=np.dtype("uint8"))
                        array_rgb = np.reshape(array_rgb, (rgb_img.height, rgb_img.width, 4))
                        array_rgb = array_rgb[:, :, :3]
                        rgb_path = os.path.join(rgb_dir, f"{global_frame_index:06d}.png")
                        cv2.imwrite(rgb_path, array_rgb)

                        # Process Segmentation
                        array_seg = np.frombuffer(seg_img.raw_data, dtype=np.dtype("uint8"))
                        array_seg = np.reshape(array_seg, (seg_img.height, seg_img.width, 4))
                        labels = array_seg[:, :, 2]
                        seg_path = os.path.join(mask_dir, f"{global_frame_index:06d}.png")
                        cv2.imwrite(seg_path, labels)

                        images_current_weather += 1
                        total_saved_images += 1
                        global_frame_index += 1
                        
                        if total_saved_images % 20 == 0:
                            print(f"   Progress: {total_saved_images}/{args.nb_images}")

                # Clean up everything (Ego, NPC, Walkers, Sensors) before next spawn/weather
                for actor in reversed(actor_list):
                    if actor.is_alive:
                        if hasattr(actor, 'stop'):
                            actor.stop() 
                        actor.destroy()

        print("\n=== Data collection finished successfully! ===")
        print(f"Total images saved: {total_saved_images}")

    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        print("Script terminated.")

if __name__ == '__main__':
    main()