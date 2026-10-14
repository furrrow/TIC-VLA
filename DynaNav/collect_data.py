"""
Standalone script to run a dynamic navigation data collection job in a local env.

Usage:
    python.sh collect_data.py -c default_config.yaml 

Optional params:
    --sensor_placment_file sensor_placements.json
    --crash_report_path /home/xxx
    --debug_print
    --save_usd
"""

import os
import sys
import yaml
import argparse
import asyncio
import numpy as np
from isaacsim import SimulationApp
from generate_commands import generate_character_commands, generate_robot_commands


APP_CONFIG = {"renderer": "RayTracedLighting", "headless": False, "width": 640, "height": 480}


class DataCollector:
    def __init__(
        self, sim_app, config_file_path, camera_file_path, crash_report_path, teleop, debug_print, save_usd
    ):
        self._sim_app = sim_app
        # Inputs
        self.config_file_path = config_file_path
        self.camera_file_path = camera_file_path
        self.crash_report_path = crash_report_path
        self.teleop = teleop
        self.debug_print = debug_print
        self.save_usd = save_usd
        self.robot_type = "Nova_Carter"  # "Spot" or "Nova_Carter"
        # self.robot_type = "Spot"  # "Spot" or "Nova_Carter"
        self.robot_behavior_script_path = {
            "Spot": "./spot_behavior.py",
            "Nova_Carter": "./robot_behavior.py"
        }

        self.output_path = None
        self.camera_placements_json = None
        self._sim_manager = None
        self._setup_sim_sub = None
        self._setup_sim_succeed = False
        self._dg_sub = None
        self._settings = None
        self._config = read_config(self.config_file_path)

    async def run(self):
        # Enable all required extensions
        self._enable_extensions()
        await self._sim_app.app.next_update_async()
        
        # Set up global settings
        self._set_simulation_settings()
        await self._sim_app.app.next_update_async()

        # Init SimulatonManager
        from isaacsim.replicator.agent.core.simulation import SimulationManager

        self._sim_manager = SimulationManager()

        try:
            can_load_config = self._sim_manager.load_config_file(self.config_file_path)
            if not can_load_config:
                return False

            writer_selection = self._sim_manager.get_config_file_property_group("replicator", "writer_selection")
            params = writer_selection.content_prop.get_value()
            self.output_path = params["output_dir"]

            # Set up sim
            await self._setup_sim()
            print("Setup sim succeed: {}".format(self._setup_sim_succeed))

            # [Optional] Camera placement
            if self.camera_file_path:
                self._do_camera_placement()

            # manually spawn robot if needed
            self._spawned_robot_prim_path = await self._spawn_robot(self.robot_type)

            # set up robot after spawning
            if self.robot_type == "Spot":
                await self._set_quadruped_robot(self._spawned_robot_prim_path)
            elif self.robot_type == "Nova_Carter":
                await self._set_carter_robot(self._spawned_robot_prim_path)
            else:
                print(f"Unknown robot type '{self.robot_type}'. No additional setup done.")

            # Generate random commands
            await self._gen_random_commands()
            print("Random commands generated.")

            # Warmup simulation
            await self._warmup(frames=300)

            # Wait for data generation callback
            await self._sim_manager.run_data_generation_async(will_wait_until_complete=True)
            return True
        except Exception as e:
            import carb

            carb.log_error(f"Failed to run simulation: {e}")
            return False

    def _enable_extensions(self):
        import omni.kit.app

        ext_manager = omni.kit.app.get_app().get_extension_manager()

        ext_manager.set_extension_enabled_immediate("omni.kit.viewport.window", True)
        ext_manager.set_extension_enabled_immediate("omni.kit.manipulator.prim", True)
        ext_manager.set_extension_enabled_immediate("omni.kit.property.usd", True)
        ext_manager.set_extension_enabled_immediate("omni.kit.scripting", True)
        ext_manager.set_extension_enabled_immediate("omni.anim.timeline", True)
        ext_manager.set_extension_enabled_immediate("omni.anim.graph.core", True)
        ext_manager.set_extension_enabled_immediate("omni.anim.retarget.core", True)
        ext_manager.set_extension_enabled_immediate("omni.anim.navigation.core", True)
        ext_manager.set_extension_enabled_immediate("omni.anim.navigation.meshtools", True)
        ext_manager.set_extension_enabled_immediate("omni.anim.people", True)
        ext_manager.set_extension_enabled_immediate("omni.isaac.sensor", True)
        ext_manager.set_extension_enabled_immediate("isaacsim.replicator.agent.core", True)
        ext_manager.set_extension_enabled_immediate("omni.kit.mesh.raycast", True)

    def _set_simulation_settings(self):
        import carb
        import omni.replicator.core as rep

        rep.settings.carb_settings("/omni/replicator/backend/writeThreads", 16)
        self._settings = carb.settings.get_settings()
        self._settings.set("/rtx/rtxsensor/coordinateFrameQuaternion", "0.5,-0.5,-0.5,-0.5")
        self._settings.set("/app/scripting/ignoreWarningDialog", True)
        self._settings.set("/persistent/exts/omni.anim.navigation.core/navMesh/viewNavMesh", True)
        self._settings.set("/exts/omni.anim.people/navigation_settings/navmesh_enabled", True)
        self._settings.set("/persistent/exts/isaacsim.replicator.agent/aim_camera_to_character", True)
        self._settings.set("/persistent/exts/isaacsim.replicator.agent/min_camera_distance", 6.5)
        self._settings.set("/persistent/exts/isaacsim.replicator.agent/max_camera_distance", 14.5)
        self._settings.set("/persistent/exts/isaacsim.replicator.agent/max_camera_look_down_angle", 60)
        self._settings.set("/persistent/exts/isaacsim.replicator.agent/min_camera_look_down_angle", 0)
        self._settings.set("/persistent/exts/isaacsim.replicator.agent/min_camera_height", 2)
        self._settings.set("/persistent/exts/isaacsim.replicator.agent/max_camera_height", 3)
        self._settings.set("/persistent/exts/isaacsim.replicator.agent/character_focus_height", 0.7)
        self._settings.set("/persistent/exts/isaacsim.replicator.agent/frame_write_interval", 3) # 30 fps -> 10 fps
        self._settings.set("/persistent/exts/isaacsim.replicator.agent/skip_starting_frames", 30)
        self._settings.set("/exts/isaacsim.anim.robot/simulation_settings/dynamic_avoidance", True)
        self._settings.set("/app/omni.graph.scriptnode/enable_opt_in", False)  # To bypass action graph scriptnode check
        self._settings.set("/rtx/raytracing/fractionalCutoutOpacity", True)  # Needed for the DH characters

        # NavMesh settings
        self._settings.set("/exts/omni.anim.navigation.core/navMesh/config/agentHeight", 180) # cm
        self._settings.set("/exts/omni.anim.navigation.core/navMesh/config/agentRadius", 40) 
        self._settings.set("/exts/omni.anim.navigation.core/navMesh/config/agentMaxStepHeight", 5) 
        self._settings.set("/exts/omni.anim.navigation.core/navMesh/config/agentMaxFloorSlope", 45.0)

        # Logging and debug print
        self._settings.set("/log/level", "info")
        self._settings.set("/log/channels/omni.replicator.core", "info")
        self._settings.set("/log/channels/isaacsim.replicator.character.core", "info")
        self._settings.set("/log/channels/omni.usd", "error")
        self._settings.set("/log/channels/omni.hydra", "error")
        self._settings.set("/log/channels/omni.kit.menu.*", "error")
        self._settings.set("/log/channels/omni.kit.property.*", "error")
        self._settings.set("/log/channels/omni.anim.graph.*", "error")
        self._settings.set("/exts/isaacsim.replicator.agent/debug_print", self.debug_print)

        # Crash reporter
        self._settings.set("/crashreporter/enabled", True)
        if self.crash_report_path:
            self._settings.set("/crashreporter/dumpDir", self.crash_report_path)
            path = self._settings.get("/crashreporter/dumpDir")
            print(f"Using carsh reporter path: {path}")

    async def _setup_sim(self):
        def done_callback(e):
            self._setup_sim_succeed = True
            self._setup_sim_sub = None

        # Set up simulation and start data generation
        self._setup_sim_sub = self._sim_manager.register_set_up_simulation_done_callback(done_callback)
        self._sim_manager.set_up_simulation_from_config_file()

        while self._setup_sim_sub and not self._sim_app.is_exiting():
            await self._sim_app.app.next_update_async()

    async def _warmup(self, frames=200):
        for _ in range(frames):
            await self._sim_app.app.next_update_async()

    # ---------- Robot spawning ----------
    async def _spawn_robot(self, robot_type: str) -> str:
        """
        Spawns a robot prim and returns its prim path.
        """
        import omni.usd
        from pxr import Gf, UsdGeom
        from isaacsim.core.utils.stage import add_reference_to_stage

        stage = omni.usd.get_context().get_stage()
        robots_root = "/World/Robots"

        if robot_type == "Spot":
            asset_url = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Robots/BostonDynamics/spot/spot.usd"
            name = "Spot"
        elif robot_type == "Nova_Carter":
            asset_url = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Carter/nova_carter_sensors.usd"
            name = "Nova_Carter"
        else:
            raise ValueError(f"Unknown robot type '{robot_type}'")

        prim_path = f"{robots_root}/{name}"
        add_reference_to_stage(usd_path=asset_url, prim_path=prim_path)
        await self._sim_app.app.next_update_async()

        # random spawn
        if robot_type == "Nova_Carter":
            spawn_location = self._sim_manager._nova_carter_randomizer.get_random_position(0)
        elif robot_type == "Spot":
            spawn_location = self._sim_manager._iw_hub_randomizer.get_random_position(0)
        else:
            spawn_location = (0.0, 0.0, 0.01)

        # Sample spawn location from predefined targets
        # if "warehouse" in self._config['scene']['asset_path']:

        # Sample spawn location
        # if "warehouse" in self._config['scene']['asset_path']:            

        #     loc = random.choice(list(WAREHOUSE_TARGETS.keys()))
        #     spawn_location = WAREHOUSE_TARGETS[loc]
        # elif "hospital" in self._config['scene']['asset_path']:
        #     loc = random.choice(list(HOSPITAL_TARGETS.keys()))
        #     spawn_location = HOSPITAL_TARGETS[loc]
        # elif "office" in self._config['scene']['asset_path']:
        #     loc = random.choice(list(OFFICE_TARGETS.keys()))
        #     spawn_location = OFFICE_TARGETS[loc]
        # elif "outdoor" in self._config['scene']['asset_path']:
        #     loc = random.choice(list(OUTDOOR_TARGETS.keys()))
        #     spawn_location = OUTDOOR_TARGETS[loc]
        # else:
        #     spawn_location = (0.0, 0.0, 0.01)

        if "outdoor" in self._config['scene']['asset_path']:
            spawn_location = random.choice(list(OUTDOOR_TARGETS.values()))

        # Set pose
        print(f"[spawn] Spawning {robot_type} at {spawn_location}")
        robot_prim = stage.GetPrimAtPath(prim_path)

        if self.robot_type == "Spot":
            robot_prim.GetAttribute("xformOp:translate").Set(
                Gf.Vec3d(float(spawn_location[0]), float(spawn_location[1]), float(spawn_location[2] + 0.8))
            )
        else:
            robot_prim.GetAttribute("xformOp:translate").Set(
                Gf.Vec3d(float(spawn_location[0]), float(spawn_location[1]), float(spawn_location[2] + 0.1))
            )

        # let stage settle
        await self._sim_app.app.next_update_async()
        print(f"[spawn] Spawned {robot_type} at {prim_path}")

        return prim_path
    
    async def _set_carter_robot(self, prim_path: str):
        """
        Mount a forward-looking camera to Carter chassis (fallback to root if link differs).
        """
        import omni.usd
        import carb
        from pxr import UsdGeom, Gf, Sdf

        stage = omni.usd.get_context().get_stage()

        robot_prim = stage.GetPrimAtPath(prim_path)
        if not robot_prim or not robot_prim.IsValid():
            carb.log_warn(f"[robot_setup] Prim not found: {prim_path}")
            return
        
        # Add behavioral script
        script_path = os.path.abspath(self.robot_behavior_script_path[self.robot_type])
        omni.kit.commands.execute("ApplyScriptingAPICommand", paths=[Sdf.Path(robot_prim.GetPrimPath())])
        attr = robot_prim.GetAttribute("omni:scripting:scripts")
        # Get the corresponding robot script
        attr.Set([Sdf.AssetPath(script_path)])
        print(f"Set up python script for robot, prim = {robot_prim.GetPrimPath()}.")
        
    async def _set_quadruped_robot(self, prim_path: str):
        """
        Lift the robot and mount a Camera sensor on its body (near head, looking forward).
        Also adds a third-person follow camera. Uses omni.isaac.sensor.Camera.
        """
        import math
        import numpy as np
        import omni.usd
        import carb
        import omni.replicator.core as rep
        from omni.isaac.sensor import Camera
        from pxr import UsdGeom, Gf, Sdf
        from isaacsim.core.utils.rotations import euler_to_rot_matrix  # keep same math style you already use
        import os  # <-- add (used for script_path)

        stage = omni.usd.get_context().get_stage()

        # --- Find robot prim -----------------------------------------------------
        robot_prim = stage.GetPrimAtPath(prim_path)
        if not (robot_prim and robot_prim.IsValid()):
            carb.log_error(f"[robot_setup] Robot prim not found at {prim_path}. Cannot set up robot.")
            return

        # attach behavior script
        script_path = os.path.abspath(self.robot_behavior_script_path[self.robot_type])
        omni.kit.commands.execute("ApplyScriptingAPICommand", paths=[Sdf.Path(robot_prim.GetPrimPath())])
        attr = robot_prim.GetAttribute("omni:scripting:scripts")
        attr.Set([Sdf.AssetPath(script_path)])
        print(f"Set up python script for robot, prim = {robot_prim.GetPrimPath()}.")

        # --- Resolve body link ---------------------------------------------------
        body_path = f"{prim_path}/body"
        body_prim = stage.GetPrimAtPath(body_path)
        if not (body_prim and body_prim.IsValid()):
            carb.log_warn(f"[robot_setup] Body link not found at {body_path}; mounting on robot root.")
            body_path = prim_path
            body_prim = robot_prim

        try:
            if body_prim.IsInstanceable():
                body_prim.SetInstanceable(False)
        except Exception:
            pass

        # --- Create head Camera sensor under /body -----------------------------------
        cam_name = "Head_Camera"
        cam_prim_path = f"{body_path}/{cam_name}"

        cam_translation = np.array([0.44, 0.075, 0.01], dtype=float)  # near head
        cam_yaw_deg   = 90.0
        cam_pitch_deg = -2.0
        roll_fix_deg  = -90.0

        np_mat_yaw     = euler_to_rot_matrix(np.array([0, cam_yaw_deg, 0]), degrees=True, extrinsic=False)
        np_mat_pitch   = euler_to_rot_matrix(np.array([-cam_pitch_deg, 0, 0]), degrees=True, extrinsic=False)
        np_mat_default = euler_to_rot_matrix(np.array([90, -90, 0]), degrees=True, extrinsic=False)
        np_mat_rollfix = euler_to_rot_matrix(np.array([roll_fix_deg, 0, 0]), degrees=True, extrinsic=False)

        rot_matrix = (
            Gf.Matrix3d(np_mat_rollfix.T.tolist())
            * Gf.Matrix3d(np_mat_pitch.T.tolist())
            * Gf.Matrix3d(np_mat_yaw.T.tolist())
            * Gf.Matrix3d(np_mat_default.T.tolist())
        )
        quat = rot_matrix.ExtractRotation().GetQuat()
        cam_orientation = np.array([quat.GetReal(), *list(quat.GetImaginary())], dtype=float)

        cam = Camera(
            prim_path=cam_prim_path,
            name=cam_name,
            frequency=30,
            resolution=(1280, 720),
            translation=cam_translation,
            orientation=cam_orientation,
        )
        cam.initialize()
        carb.log_info(f"[robot_setup] Camera sensor created at {cam_prim_path}")

        # Optional intrinsics
        try:
            width_px, height_px = 1280, 720
            film_w_mm = 20.955 * (width_px / 1920.0)
            film_h_mm = film_w_mm * (height_px / float(width_px))
            focal_mm  = (0.5 * film_w_mm) / math.tan(math.radians(90.0) * 0.5)

            usd_cam = UsdGeom.Camera.Get(stage, Sdf.Path(cam_prim_path))
            usd_cam.CreateHorizontalApertureAttr(film_w_mm)
            usd_cam.CreateVerticalApertureAttr(film_h_mm)
            usd_cam.CreateFocalLengthAttr(float(focal_mm))
            usd_cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
        except Exception as e:
            carb.log_warn(f"[robot_setup] Could not set camera intrinsics: {e}")

        await self._sim_app.app.next_update_async()

        rp = rep.create.render_product(cam_prim_path, resolution=(1280, 720))
        carb.log_info(f"[robot_setup] Replicator render product created for {cam_prim_path}: {rp}")

        await self._sim_app.app.next_update_async()

        # ===================== Third-person camera ============================
        # Mount behind & above the robot, looking at the body origin.
        try:
            # --- Resolve body link ---------------------------------------------------
            body_path = f"{prim_path}/body"
            body_prim = stage.GetPrimAtPath(body_path)
            if not (body_prim and body_prim.IsValid()):
                carb.log_warn(f"Body link not found at {body_path}; mounting on robot root.")
                body_path = str(self.prim_path)
                body_prim = robot_prim

            # --- Create Third Person Camera sensor under /body ----------------------
            tp_cam_name = "ThirdPerson_Camera"
            tp_cam_prim_path = f"{body_path}/{tp_cam_name}"

            # Position camera behind and above robot
            tp_cam_translation = np.array([-2.5, 0.0, 1.0], dtype=float)  # Behind and above

            # Orientation: look at robot with slight downward angle
            tp_cam_yaw_deg = 90.0      # look straight forward (+X)
            tp_cam_pitch_deg = 0.0  # downward angle
            tp_roll_fix_deg = -90.0   # rotate image back by +90°. If wrong way, use -90.0
            tp_np_mat_yaw     = euler_to_rot_matrix(np.array([0, tp_cam_yaw_deg, 0]), degrees=True, extrinsic=False)
            tp_np_mat_pitch   = euler_to_rot_matrix(np.array([-tp_cam_pitch_deg, 0, 0]), degrees=True, extrinsic=False)
            tp_np_mat_default = euler_to_rot_matrix(np.array([90, -90, 0]), degrees=True, extrinsic=False)
            tp_np_mat_rollfix = euler_to_rot_matrix(np.array([tp_roll_fix_deg, 0, 0]), degrees=True, extrinsic=False)

            tp_rot_matrix = (
                Gf.Matrix3d(tp_np_mat_rollfix.T.tolist())
                * Gf.Matrix3d(tp_np_mat_pitch.T.tolist())
                * Gf.Matrix3d(tp_np_mat_yaw.T.tolist())
                * Gf.Matrix3d(tp_np_mat_default.T.tolist())
            )
            tp_quat = tp_rot_matrix.ExtractRotation().GetQuat()  # Gf.Quatd
            tp_cam_orientation = np.array([tp_quat.GetReal(), *list(tp_quat.GetImaginary())], dtype=float)  # [w, x, y, z]

            # Create the sensor (owns a render product internally)
            self._third_person_camera = Camera(
                prim_path=tp_cam_prim_path,
                name=tp_cam_name,
                frequency=30,
                resolution=(1280, 720),
                translation=tp_cam_translation,
                orientation=tp_cam_orientation,
            )
            self._third_person_camera.initialize()
            self._third_person_camera_prim = stage.GetPrimAtPath(tp_cam_prim_path)
            carb.log_info(f"Third person camera sensor created at {tp_cam_prim_path}")

            # Optional: set intrinsics to match 90° horizontal FOV style
            # match "scale from 1920px → 20.955 mm" convention
            width_px, height_px = 1280, 720
            film_w_mm = 20.955 * (width_px / 1920.0)
            film_h_mm = film_w_mm * (height_px / float(width_px))
            focal_mm = (0.5 * film_w_mm) / math.tan(math.radians(90.0) * 0.5)

            usd_cam = UsdGeom.Camera.Get(stage, Sdf.Path(tp_cam_prim_path))
            usd_cam.CreateHorizontalApertureAttr(film_w_mm)
            usd_cam.CreateVerticalApertureAttr(film_h_mm)
            usd_cam.CreateFocalLengthAttr(float(focal_mm))
            usd_cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))

            # Create Replicator render product for logging
            rp = rep.create.render_product(tp_cam_prim_path, resolution=(1280, 720))
            carb.log_info(f"Replicator render product created for {tp_cam_prim_path}: {rp}")

            await self._sim_app.app.next_update_async()

        except Exception as e:
            carb.log_error(f"[robot_setup] Failed to create third-person camera: {e}")
        # ==========================================================================

    async def _gen_random_commands(self):
        if self._sim_manager.get_config_file_valid_value("character", "command_file"):
            if "default" in self._sim_manager.get_config_file_property("character", "command_file").get_value():
                task = asyncio.create_task(self._sim_manager.generate_random_commands())
                await task
                commands = task.result()
                self._sim_manager.save_commands(commands)
                print("Use built-in character command generator")
            else:
                if "hospital" in self._config['scene']['asset_path']:
                    generate_character_commands(
                        "hospital",
                        num_characters=self._config['character']['num'],
                        seed=self._config['global']['seed'],
                        filename=self._config['character']['command_file']
                    )
                elif "office" in self._config['scene']['asset_path']:
                    generate_character_commands(
                        "office",
                        num_characters=self._config['character']['num'],
                        seed=self._config['global']['seed'],
                        filename=self._config['character']['command_file']
                    )
                elif "outdoor" in self._config['scene']['asset_path']:
                    generate_character_commands(
                        "outdoor",
                        num_characters=self._config['character']['num'],
                        seed=self._config['global']['seed'],
                        filename=self._config['character']['command_file']
                    )
                elif "warehouse" in self._config['scene']['asset_path']:
                    generate_character_commands(
                        "warehouse",
                        num_characters=self._config['character']['num'],
                        seed=self._config['global']['seed'],
                        filename=self._config['character']['command_file']
                    )
                else:
                    print("No character command generated. Unknown scene.")
        
        print("Generate people commands completed")

        if self._sim_manager.get_config_file_valid_value("robot", "command_file"):
            if "default" in self._sim_manager.get_config_file_property("robot", "command_file").get_value():
                task = asyncio.create_task(self._sim_manager.generate_random_robot_commands())
                await task
                commands = task.result()
                self._sim_manager.save_robot_commands(commands)
                print("Use built-in robot command generator")
            else:
                if "hospital" in self._config['scene']['asset_path']:
                    generate_robot_commands("hospital", seed=self._config['global']['seed'], filename=self._config['robot']['command_file'])
                elif "office" in self._config['scene']['asset_path']:
                    generate_robot_commands("office", seed=self._config['global']['seed'], filename=self._config['robot']['command_file'])
                elif "outdoor" in self._config['scene']['asset_path']:
                    generate_robot_commands("outdoor", seed=self._config['global']['seed'], filename=self._config['robot']['command_file'])
                elif "warehouse" in self._config['scene']['asset_path']:
                    generate_robot_commands("warehouse", seed=self._config['global']['seed'], filename=self._config['robot']['command_file'])
                else:
                    print("No robot command generated. Unknown scene.")

        print("Generate robot commands completed")
        
    # ===== Camera placement related =====

    def _do_camera_placement(self):
        self._read_camera_json()
        if not self.camera_placements_json:
            return
        print(f"camera_placements_json = {self.camera_placements_json}")
        prop = self._sim_manager.get_config_file_property("sensor", "camera_num")
        prop.set_value(len(self.camera_placements_json))
        # Load camera
        self._sim_manager.load_camera_from_config_file()
        self._place_cameras()

    def _read_camera_json(self):
        import json

        import carb
        import omni.client

        # Read json file
        result, version, context = omni.client.read_file(self.camera_file_path)
        if result != omni.client.Result.OK:
            carb.log_error(f"Cannot load camera file path: {self.camera_file_path}. Skip camera placement.")
            return
        json_str = memoryview(context).tobytes().decode("utf-8")
        self.camera_placements_json = json.loads(json_str)

    def _place_cameras(self):
        import carb

        # Perform placement
        from isaacsim.replicator.agent.core.stage_util import CameraUtil

        camera_prims = CameraUtil.get_cameras_in_stage()
        count = 0
        for camera_dict in self.camera_placements_json:
            if count >= len(camera_prims):
                carb.log_warn("No enough cameras in the scene to set placement. Will skip the rest placement data.")
                break
            self._place_one_camera(camera_dict, camera_prims[count])
            count += 1

        print(f"Place total {count} cameras.")

    def _place_one_camera(self, camera_dict, camera_prim):
        from isaacsim.core.utils.rotations import euler_to_rot_matrix
        from isaacsim.replicator.agent.core.stage_util import CameraUtil
        from pxr import Gf

        # Extract focal length
        # - In OV, the default pixel size is 20.955/1920=0.0109140625mm
        ov_focal_length = camera_dict["focal_length"] * 0.0109140625
        # Extract transformation
        ov_pos = Gf.Vec3d(camera_dict["x"], camera_dict["y"], camera_dict["height"])
        yaw = camera_dict["yaw"]
        pitch = camera_dict["pitch"]
        np_mat_yaw = euler_to_rot_matrix(np.array([0, yaw, 0]), degrees=True, extrinsic=False)
        np_mat_pitch = euler_to_rot_matrix(np.array([-pitch, 0, 0]), degrees=True, extrinsic=False)
        np_mat_default = euler_to_rot_matrix(
            np.array([90, -90, 0]), degrees=True, extrinsic=False
        )  # To make sure when yaw=0, the camera in IsaacSim points to X positive
        rot_matrix = (
            Gf.Matrix3d(np_mat_pitch.T.tolist())
            * Gf.Matrix3d(np_mat_yaw.T.tolist())
            * Gf.Matrix3d(np_mat_default.T.tolist())
        )
        # ov_rot_euler = rot_matrix.ExtractRotation().Decompose(Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis())
        ov_rot = rot_matrix.ExtractRotation().GetQuat()
        CameraUtil.set_camera(camera_prim, ov_pos, ov_rot, ov_focal_length)


def read_config(config_file_path):
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
        config = config['isaacsim.replicator.agent']
    return config

def get_args():
    parser = argparse.ArgumentParser("DynaNav Data Collection")
    parser.add_argument("-c", "--config_file", required=True, help="Path to a IRA config file")
    # Optional config
    parser.add_argument(
        "--sensor_placment_file", required=False, help="Path to camera placement json file. Default is none."
    )
    parser.add_argument(
        "--crash_report_path",
        required=False,
        default=None,
        help="Path to store the crash report. Default is the current working directory.",
    )
    parser.add_argument(
        "--teleoperation",
        required=False,
        default=False,
        action="store_true",
        help="Enable teleoperation mode.",
    )
    parser.add_argument(
        "--debug_print", required=False, default=False, action="store_true", help="Enable IRA debug print."
    )
    parser.add_argument(
        "--save_usd",
        action="store_true",
        default=False,
        help="Save the simulated scene when data generation finishes.",
    )
    args, _ = parser.parse_known_args()
    return args


# ===== Save USD =====
async def _save_usd(save_as_path):
    print("Saving USD...")
    try:
        import omni.usd

        await omni.usd.get_context().save_as_stage_async(save_as_path)
        print("Save scene to: " + str(save_as_path))
        await omni.usd.get_context().close_stage_async()
    except Exception as e:
        print("Caught exception. Unable to save USD. " + str(e), file=sys.stderr)


def main():
    # Read command line arguments
    args = get_args()
    config_file_path = args.config_file
    camera_file_path = args.sensor_placment_file
    crash_report_path = args.crash_report_path
    teleoperation = args.teleoperation
    debug_print = args.debug_print
    save_usd = args.save_usd

    print("Config file path: {}".format(config_file_path))
    print("Sensor placement file path: {}".format(camera_file_path))
    print("Crash Report Path: {}".format(crash_report_path))
    print("Teleoperation: {}".format(teleoperation))
    print("Debug Print: {}".format(debug_print))
    print("Save USD: {}".format(save_usd))

    # Check files exist
    if not os.path.isfile(config_file_path):
        print("Invalid config file path. Exit.", file=sys.stderr)
        return
    if camera_file_path and not os.path.isfile(camera_file_path):
        print("Invalid camera placement path. Exit.", file=sys.stderr)
        return

    # Start SimApp
    sim_app = SimulationApp(launch_config=APP_CONFIG)

    # Start Data Collection
    data_collector = DataCollector(
        sim_app,
        os.path.abspath(config_file_path),
        camera_file_path,
        crash_report_path,
        teleoperation,
        debug_print,
        save_usd,
    )
    
    from omni.kit.async_engine import run_coroutine
    task = run_coroutine(data_collector.run())

    # Main update loop
    try:
        # Also stop updating if app is exiting (e.g., via ESC)
        while not task.done() and not sim_app.is_exiting():
            sim_app.update()

        # Handle result (including user cancellation) without throwing
        try:
            ok = bool(task.result())
        except asyncio.CancelledError:
            print("Task was cancelled by user (ESC).")
            ok = False
        except Exception as e:
            print(f"Task error: {e}")
            ok = False

        if not ok:
            print("Failed to run data collection. Exit.")

        # [Optional] Save USD only if the run completed successfully
        if save_usd and ok:
            import omni.client
            save_as_path = omni.client.combine_urls("{}/".format(data_collector.output_path), "scene.usd")
            save_usd_task = asyncio.ensure_future(_save_usd(save_as_path))
            while not save_usd_task.done() and not sim_app.is_exiting():
                sim_app.update()

    finally:
        print("TASK DONE. EXIT.")
        sim_app.close()


if __name__ == "__main__":
    main()
