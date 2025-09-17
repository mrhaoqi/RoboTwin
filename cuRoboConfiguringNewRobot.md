Title: Configuring a New Robot - cuRobo

URL Source: https://curobo.org/tutorials/1_robot_configuration.html

Markdown Content:
Prequisite[¶](https://curobo.org/tutorials/1_robot_configuration.html#prequisite "Link to this heading")
--------------------------------------------------------------------------------------------------------

*   Install [NVIDIA Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/install_workstation.html)

*   URDF with meshes for your robot.

Robot Configuration File[¶](https://curobo.org/tutorials/1_robot_configuration.html#robot-configuration-file "Link to this heading")
------------------------------------------------------------------------------------------------------------------------------------

To configure a new robot in cuRobo, you need to create a yaml file that can be loaded into [`curobo.cuda_robot_model.cuda_robot_generator.CudaRobotGeneratorConfig`](https://curobo.org/_api/curobo.cuda_robot_model.cuda_robot_generator.html#curobo.cuda_robot_model.cuda_robot_generator.CudaRobotGeneratorConfig "curobo.cuda_robot_model.cuda_robot_generator.CudaRobotGeneratorConfig").

We provide a template at `curobo/src/curobo/content/configs/robot/template.yml` that you can duplicate to build your robot configuration file. The template is displayed below:

robot_cfg:
 kinematics:
 usd_path: "FILL_THIS"
 usd_robot_root: "/robot"
 isaac_usd_path: ""
 usd_flip_joints: {}
 usd_flip_joint_limits: []

 urdf_path: "FILL_THIS"
 asset_root_path: ""

 base_link: "FILL_THIS"
 ee_link: "FILL_THIS"
 link_names: null
 lock_joints: null
 extra_links: null

 collision_link_names: null # List[str]
 collision_spheres: null #
 collision_sphere_buffer: 0.005
 extra_collision_spheres: {}
 self_collision_ignore: null # Dict[str, List[str]]
 self_collision_buffer: null # Dict[str, float]

 use_global_cumul: True
 mesh_link_names: null # List[str]

 cspace:
 joint_names: [] # List[str]
 retract_config: null # List[float]
 null_space_weight: null # List[str]
 cspace_distance_weight: null # List[str]
 max_jerk: 500.0
 max_acceleration: 15.0

An example configuration for the Franka Panda looks like,

robot_cfg:
 kinematics:
 use_usd_kinematics: False
 isaac_usd_path: "/Isaac/Robots/Franka/franka.usd"
 usd_path: "robot/franka_description/franka_panda_meters.usda"
 usd_robot_root: "/panda"
 usd_flip_joints: ["panda_joint1","panda_joint2","panda_joint3","panda_joint4", "panda_joint5",
 "panda_joint6","panda_joint7","panda_finger_joint1", "panda_finger_joint2"]
 usd_flip_joints: {
 "panda_joint1": "Z",
 "panda_joint2": "Z",
 "panda_joint3": "Z",
 "panda_joint4": "Z",
 "panda_joint5": "Z",
 "panda_joint6": "Z",
 "panda_joint7": "Z",
 "panda_finger_joint1": "Y",
 "panda_finger_joint2": "Y",
 }
 usd_flip_joint_limits: ["panda_finger_joint2"]
 urdf_path: "robot/franka_description/franka_panda.urdf"
 asset_root_path: "robot/franka_description"
 base_link: "base_link"
 ee_link: "panda_hand"
 # link_names: null
 collision_link_names:
 [
 "panda_link0",
 "panda_link1",
 "panda_link2",
 "panda_link3",
 "panda_link4",
 "panda_link5",
 "panda_link6",
 "panda_link7",
 "panda_hand",
 "panda_leftfinger",
 "panda_rightfinger",
 "attached_object",
 ]
 collision_spheres: "spheres/franka_mesh.yml"
 collision_sphere_buffer: 0.003 #0.01
 extra_collision_spheres: {"attached_object": 4}
 use_global_cumul: True
 self_collision_ignore:
 {
 "panda_link0": ["panda_link1", "panda_link2"],
 "panda_link1": ["panda_link2", "panda_link3", "panda_link4"],
 "panda_link2": ["panda_link3", "panda_link4"],
 "panda_link3": ["panda_link4", "panda_link6"],
 "panda_link4":
 ["panda_link5", "panda_link6", "panda_link7", "panda_link8"],
 "panda_link5": ["panda_link6", "panda_link7", "panda_hand","panda_leftfinger", "panda_rightfinger"],
 "panda_link6": ["panda_link7", "panda_hand", "attached_object", "panda_leftfinger", "panda_rightfinger"],
 "panda_link7": ["panda_hand", "attached_object", "panda_leftfinger", "panda_rightfinger"],
 "panda_hand": ["panda_leftfinger", "panda_rightfinger","attached_object"],
 "panda_leftfinger": ["panda_rightfinger", "attached_object"],
 "panda_rightfinger": ["attached_object"],

 }

 self_collision_buffer:
 {
 "panda_link0": 0.1,
 "panda_link1": 0.05,
 "panda_link2": 0.0,
 "panda_link3": 0.0,
 "panda_link4": 0.0,
 "panda_link5": 0.0,
 "panda_link6": 0.0,
 "panda_link7": 0.0,
 "panda_hand": 0.0,
 "panda_leftfinger": 0.01,
 "panda_rightfinger": 0.01,
 "attached_object": 0.0,
 }

 mesh_link_names:
 [
 "panda_link0",
 "panda_link1",
 "panda_link2",
 "panda_link3",
 "panda_link4",
 "panda_link5",
 "panda_link6",
 "panda_link7",
 "panda_hand",
 "panda_leftfinger",
 "panda_rightfinger",
 ]
 lock_joints: {"panda_finger_joint1": 0.04, "panda_finger_joint2": -0.04}
 extra_links: {"attached_object":{"parent_link_name": "panda_hand" ,
 "link_name": "attached_object", "fixed_transform": [0,0,0,1,0,0,0], "joint_type":"FIXED",
 "joint_name": "attach_joint" }}
 cspace:
 joint_names: ["panda_joint1","panda_joint2","panda_joint3","panda_joint4", "panda_joint5",
 "panda_joint6","panda_joint7","panda_finger_joint1", "panda_finger_joint2"]
 retract_config: [0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0., 0.04, -0.04]
 null_space_weight: [1,1,1,1,1,1,1,1,1]
 cspace_distance_weight: [1,1,1,1,1,1,1,1,1]
 max_acceleration: 15.0
 max_jerk: 500.0

Tutorial with a UR5e robot[¶](https://curobo.org/tutorials/1_robot_configuration.html#tutorial-with-a-ur5e-robot "Link to this heading")
----------------------------------------------------------------------------------------------------------------------------------------

We provide prepared assets for the ur5e in `curobo/src/curobo/content/assets/ur_description`. We prepared the assets using the below instructions. You can skip the below step and directly use the provided assets.

### Prepare assets[¶](https://curobo.org/tutorials/1_robot_configuration.html#prepare-assets "Link to this heading")

1.   Create a new folder to keep the urdf and meshes. `mkdir robot`.

2.   Clone [https://github.com/ros-industrial/universal_robot](https://github.com/ros-industrial/universal_robot) into your catkin workspace and convert the xacro file for ur5e to urdf using `rosrun xacro xacro -o robot/ur5e.urdf universal_robot/ur_description/urdf/ur5e.xacro`, more details are at [http://wiki.ros.org/xacro](http://wiki.ros.org/xacro)

3.   Copy meshes to the robot folder, `cp universal_robot/ur_description/meshes/ur5e robot/meshes/ur5e/`

4.   Replace all instances of `package://ur_description/meshes` with `meshes/` in `ur5e.urdf`

### Create Configuration File[¶](https://curobo.org/tutorials/1_robot_configuration.html#create-configuration-file "Link to this heading")

1.   Copy the `template.yml` file and name it `ur5e.yml`

2.   Update `urdf_path` and `asset_root_path` to the path of your urdf file and the location of the meshes file.

3.   Update `base_link` and `ee_link` to the name of the base link and end-effector link you want to use for Cartesian Pose planning.

4.   Update `joint_names` in `cspace` to match the articulated joint names in your urdf.

5.   Set a default collision-free configuration for the robot with `retract_config`.

6.   Initialize `null_space_weight` and `cspace_distance_weight` to a vector of length of `joint_names` and value 1. If you want to weigh the joints differently use values less than 1.

The configuration file should now look like below:

robot_cfg:
 kinematics:
 usd_path: "robot/ur_description/ur5e.usd"
 usd_robot_root: "/robot"
 isaac_usd_path: ""
 usd_flip_joints: {}
 usd_flip_joint_limits: []

 urdf_path: "robot/ur_description/ur5e.urdf"
 asset_root_path: "robot/ur_description"

 base_link: "base_link"
 ee_link: "tool0"
 link_names: null
 lock_joints: null
 extra_links: null

 collision_link_names: null # List[str]
 collision_spheres: null #
 collision_sphere_buffer: 0.005
 extra_collision_spheres: {}
 self_collision_ignore: null # Dict[str, List[str]]
 self_collision_buffer: null # Dict[str, float]

 use_global_cumul: True
 mesh_link_names: null # List[str]

 cspace:
 joint_names: ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
 retract_config: [-1.57, -2.2, 1.9, -1.383, -1.57, 0.00]
 null_space_weight: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
 cspace_distance_weight: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
 max_jerk: 500.0
 max_acceleration: 15.0

To enable collision checking, we need to build a sphere model of the robot. We will use NVIDIA Isaac Sim to build the sphere model.

### Robot Collision Representation[¶](https://curobo.org/tutorials/1_robot_configuration.html#robot-collision-representation "Link to this heading")

1.   Install cuRobo in Isaac sim’s python environment by running `omni_python -m pip install -e . --no-build-isolation` where `omni_python` is the python script from isaac sim (usually found in `ISAAC_SIM_PATH/python.sh`).

2.   Convert the urdf of the robot to usd using: `omni_python curobo/examples/isaac_sim/utils/convert_urdf_to_usd.py --robot /{PATH}/ur5e.yml --save_usd`. Here `{PATH}` is the path to your robot file. If you want to use the existing one, just use `ur5e.yml`, which will load the configuraiton cuRobo’s package.

3.   Launch NVIDIA Isaac Sim and open the converted usd file.

4.   Launch Isaac Utils -> Lula Robot Description Editor and follow the video below:

1.   Once you have the yaml file, copy the `collision_spheres:` section into your `ur5e.yml` file. Remove `-` before each link name as the generated configuration file treats it as a list while cuRobo requires it to be a dictionary.

2.   Add the name of all links that you want to perform collision checking to `collision_link_names`.

Note

Reduce the radius of the link of your robot that will attach to the world (e.g., `base_link`) so that when the robot is standing in the ground, it does not trigger a world collision. You can then add an offset for this link in the `self_collision_buffer` so that the robot will avoid the link during self collision checking.

Your configuration file should look like below,

robot_cfg:
 kinematics:
 usd_path: "robot/ur_description/ur5e.usd"
 usd_robot_root: "/robot"
 isaac_usd_path: ""
 usd_flip_joints: {}
 usd_flip_joint_limits: []

 urdf_path: "robot/ur_description/ur5e.urdf"
 asset_root_path: "robot/ur_description"

 base_link: "base_link"
 ee_link: "tool0"
 link_names: null
 lock_joints: null
 extra_links: null

 collision_link_names: ["shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"]
 collision_spheres:
 shoulder_link:
 - "center": [0.0, 0.0, 0.0]
 "radius": 0.1
 upper_arm_link:
 - "center": [-0.416, -0.0, 0.143]
 "radius": 0.078
 - "center": [-0.015, 0.0, 0.134]
 "radius": 0.077
 - "center": [-0.14, 0.0, 0.138]
 "radius": 0.062
 - "center": [-0.285, -0.001, 0.139]
 "radius": 0.061
 - "center": [-0.376, 0.001, 0.138]
 "radius": 0.077
 - "center": [-0.222, 0.001, 0.139]
 "radius": 0.061
 - "center": [-0.055, 0.008, 0.14]
 "radius": 0.07
 - "center": [-0.001, -0.002, 0.143]
 "radius": 0.076
 forearm_link:
 - "center": [-0.01, 0.002, 0.031]
 "radius": 0.072
 - "center": [-0.387, 0.0, 0.014]
 "radius": 0.057
 - "center": [-0.121, -0.0, 0.006]
 "radius": 0.057
 - "center": [-0.206, 0.001, 0.007]
 "radius": 0.057
 - "center": [-0.312, -0.001, 0.006]
 "radius": 0.056
 - "center": [-0.057, 0.003, 0.008]
 "radius": 0.065
 - "center": [-0.266, 0.0, 0.006]
 "radius": 0.057
 - "center": [-0.397, -0.001, -0.018]
 "radius": 0.052
 - "center": [-0.164, -0.0, 0.007]
 "radius": 0.057
 wrist_1_link:
 - "center": [-0.0, 0.0, -0.009]
 "radius": 0.047
 - "center": [-0.0, 0.0, -0.052]
 "radius": 0.047
 - "center": [-0.002, 0.027, -0.001]
 "radius": 0.045
 - "center": [0.001, -0.01, 0.0]
 "radius": 0.046
 wrist_2_link:
 - "center": [0.0, -0.01, -0.001]
 "radius": 0.047
 - "center": [0.0, 0.008, -0.001]
 "radius": 0.047
 - "center": [0.001, -0.001, -0.036]
 "radius": 0.047
 - "center": [0.001, -0.03, -0.0]
 "radius": 0.047
 wrist_3_link:
 - "center": [0.001, 0.001, -0.029]
 "radius": 0.043

 collision_sphere_buffer: 0.005
 extra_collision_spheres: {}
 self_collision_ignore: {} # Dict[str, List[str]]
 self_collision_buffer: null # Dict[str, float]

 use_global_cumul: True
 mesh_link_names: null # List[str]

 cspace:
 joint_names: ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
 retract_config: [-1.57, -2.2, 1.9, -1.383, -1.57, 0.00]
 null_space_weight: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
 cspace_distance_weight: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
 max_jerk: 500.0
 max_acceleration: 15.0

Next, we will configure the self-collision representation for the robot.

### Self Collision Configuration[¶](https://curobo.org/tutorials/1_robot_configuration.html#self-collision-configuration "Link to this heading")

To handle self-collisions, we use two keys in the file, specifically a `self_collision_ignore` dictionary that lists which links to ignore self-collision checks for each link. We can fill this with consecutive links in the robot. We map the pairs in both directions, so it’s sufficient to list the link-link pair once (e.g., if you list link-1: [link-2], then it’s not neccessary to add link-1 to link-2 list). In addition some robots might have three links very close together and are kinematically limited to avoid collision, you can add these as well to the dictionary as it will speed-up self collision checks.

We also use a `self_collision_buffer` to add safety buffers to the spheres. We use this for some robots which use a more conservative geometry for self-collision checks such as the Franka Panda: [https://github.com/frankaemika/franka_ros/issues/39](https://github.com/frankaemika/franka_ros/issues/39)

Your configuration file should now look as below,

robot_cfg:
 kinematics:
 usd_path: "robot/ur_description/ur5e.usd"
 usd_robot_root: "/robot"
 isaac_usd_path: ""
 usd_flip_joints: {}
 usd_flip_joint_limits: []

 urdf_path: "robot/ur_description/ur5e.urdf"
 asset_root_path: "robot/ur_description"

 base_link: "base_link"
 ee_link: "tool0"
 link_names: null
 lock_joints: null
 extra_links: null

 collision_link_names: ["shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"]

 collision_spheres:
 shoulder_link:
 - "center": [0.0, 0.0, 0.0]
 "radius": 0.1
 upper_arm_link:
 - "center": [-0.416, -0.0, 0.143]
 "radius": 0.078
 - "center": [-0.015, 0.0, 0.134]
 "radius": 0.077
 - "center": [-0.14, 0.0, 0.138]
 "radius": 0.062
 - "center": [-0.285, -0.001, 0.139]
 "radius": 0.061
 - "center": [-0.376, 0.001, 0.138]
 "radius": 0.077
 - "center": [-0.222, 0.001, 0.139]
 "radius": 0.061
 - "center": [-0.055, 0.008, 0.14]
 "radius": 0.07
 - "center": [-0.001, -0.002, 0.143]
 "radius": 0.076
 forearm_link:
 - "center": [-0.01, 0.002, 0.031]
 "radius": 0.072
 - "center": [-0.387, 0.0, 0.014]
 "radius": 0.057
 - "center": [-0.121, -0.0, 0.006]
 "radius": 0.057
 - "center": [-0.206, 0.001, 0.007]
 "radius": 0.057
 - "center": [-0.312, -0.001, 0.006]
 "radius": 0.056
 - "center": [-0.057, 0.003, 0.008]
 "radius": 0.065
 - "center": [-0.266, 0.0, 0.006]
 "radius": 0.057
 - "center": [-0.397, -0.001, -0.018]
 "radius": 0.052
 - "center": [-0.164, -0.0, 0.007]
 "radius": 0.057
 wrist_1_link:
 - "center": [-0.0, 0.0, -0.009]
 "radius": 0.047
 - "center": [-0.0, 0.0, -0.052]
 "radius": 0.047
 - "center": [-0.002, 0.027, -0.001]
 "radius": 0.045
 - "center": [0.001, -0.01, 0.0]
 "radius": 0.046
 wrist_2_link:
 - "center": [0.0, -0.01, -0.001]
 "radius": 0.047
 - "center": [0.0, 0.008, -0.001]
 "radius": 0.047
 - "center": [0.001, -0.001, -0.036]
 "radius": 0.047
 - "center": [0.001, -0.03, -0.0]
 "radius": 0.047
 wrist_3_link:
 - "center": [0.001, 0.001, -0.029]
 "radius": 0.043

 collision_sphere_buffer: 0.005
 extra_collision_spheres: {}
 self_collision_ignore: {
 "upper_arm_link": ["forearm_link", "shoulder_link"],
 "forearm_link": ["wrist_1_link", "wrist_2_link", "wrist_3_link"],
 "wrist_1_link": ["wrist_2_link","wrist_3_link"],
 "wrist_2_link": ["wrist_3_link"],
 }
 self_collision_buffer: {'upper_arm_link': 0,
 'forearm_link': 0,
 'wrist_1_link': 0,
 'wrist_2_link': 0,
 'wrist_3_link' : 0,
 }

 use_global_cumul: True
 mesh_link_names: null # List[str]

 cspace:
 joint_names: ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
 retract_config: [-1.57, -2.2, 1.9, -1.383, -1.57, 0.00]
 null_space_weight: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
 cspace_distance_weight: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
 max_jerk: 500.0
 max_acceleration: 15.0

### Additional Configurations[¶](https://curobo.org/tutorials/1_robot_configuration.html#additional-configurations "Link to this heading")

1. If your robot has a gripper attached and you want to fix them at a specific configuration (e.g., do not want to plan for the grippers joints), you can add them to `lock_joints` tag. As an example, for the Franka Panda we set the fingers to fully open with `lock_joints: {"panda_finger_joint1": 0.04, "panda_finger_joint2": -0.04}`

### Test Robot Configuration[¶](https://curobo.org/tutorials/1_robot_configuration.html#test-robot-configuration "Link to this heading")

1.   Run `omni_python curobo/examples/isaac_sim/motion_gen_reacher.py --robot /{PATH}/ur5e.yml --visualize_spheres` to start isaac sim with the configured robot. Replace `PATH` with the path to your file.

2.   Click Play and move the red cube to enable motion generation for the robot. You should see behavior similar to [Motion Generation](https://curobo.org/get_started/2b_isaacsim_examples.html#isaac-motion-generation).
