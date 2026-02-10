import viser 
import yourdfpy
import numpy as np
import time
from viser.extras import ViserUrdf
import vamp

PROBLEM_SPHERES = [
    [0.55, 0, 0.25],
    [0.35, 0.35, 0.25],
    [0, 0.55, 0.25],
    [-0.55, 0, 0.25],
    [-0.35, -0.35, 0.25],
    [0, -0.55, 0.25],
    [0.35, -0.35, 0.25],
    [-0.35, 0.35, 0.25],
    
    [0.35, 0.35, 0.8],
    [0, 0.55, 0.8],
    [-0.35, 0.35, 0.8],
    [-0.55, 0, 0.8],
    [-0.35, -0.35, 0.8],
    [0, -0.55, 0.8],
    [0.35, -0.35, 0.8],
    [0.55, 0, 0.8],
]
SPHERE_RADIUS = 0.2
robot_urdf_path="resources/panda/panda_spherized.urdf"
robot_mesh_dir="resources/panda/meshes"

lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

sdf_cli_path = "/Users/weihang/Documents/research/vamp/build/vamp_sdf_cli"

def create_sphere_vertices(radius, rings=10, sectors=10):
    vertices = []
    for r in range(rings + 1):
        theta = r * np.pi / rings
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        for s in range(sectors + 1):
            phi = s * 2 * np.pi / sectors
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)
            x = radius * cos_phi * sin_theta
            y = radius * sin_phi * sin_theta
            z = radius * cos_theta
            vertices.append([x, y, z])
    return np.array(vertices)

def create_sphere_faces(rings=10, sectors=10):
    faces = []
    for r in range(rings):
        for s in range(sectors):
            first = r * (sectors + 1) + s
            second = first + sectors + 1
            faces.append([first, second, first + 1])
            faces.append([second, second + 1, first + 1])
    return np.array(faces)

def main():
    server = viser.ViserServer()
    
    # Setup robot
    robot_urdf = yourdfpy.URDF.load(robot_urdf_path, mesh_dir=robot_mesh_dir)
    # Ghost robot for initial colliding state (Red)
    viser_urdf_init = ViserUrdf(server, robot_urdf, root_node_name="/panda_init")

    viser_urdf_proj = ViserUrdf(server, robot_urdf, root_node_name="/panda_proj", mesh_color_override=(0, 255, 0, 0.5))

    for i, pos in enumerate(PROBLEM_SPHERES):
        server.scene.add_mesh_simple(
            name=f"/spheres/sphere_{i}",
            position=np.array(pos),
            vertices=create_sphere_vertices(SPHERE_RADIUS),
            faces=create_sphere_faces(),
            color=(255, 100, 100),
            opacity=0.8
        )

  
    # Create environment
    env = vamp.Environment()
    for pos in PROBLEM_SPHERES:
        env.add_sphere(vamp.Sphere(pos, SPHERE_RADIUS))

    def sample_and_project():
        print("Sampling colliding config...")
        q_rand = np.random.uniform(lower, upper)
        viser_urdf_init.update_cfg(q_rand)        
        # Show initial
        viser_urdf_proj.update_cfg(q_rand)
        
        # Optimize
        print("Optimizing...")
        start_t = time.time()
        
        q_proj = vamp.panda.project_to_valid(q_rand, env)
        print(f"Projected config shape: {q_proj}")

        q_proj = np.array(q_proj).transpose().flatten()[:len(lower)]
        print(f"Projected config raw: {q_proj}")
        end_t = time.time()
        
        viser_urdf_proj.update_cfg(q_proj)
        print(f"Projected config: {q_proj}")
        print(f"Optimization took {end_t - start_t:.4f}s")
        
        
    btn = server.gui.add_button("Sample & Project")
    # Text input for manual config
    config_input = server.gui.add_text("Config Input", initial_value="-1.280461, -0.539198, -1.745114, -3.076952, -1.076599, 1.627267, -0.418525")
    
    def visualize_from_input():
        try:
            txt = config_input.value
            # Handle newlines and commas
            vals = [float(x.strip()) for x in txt.replace('\n', ',').split(',') if x.strip()]
            q_manual = np.array(vals)
            
            if len(q_manual) != 7:
                print(f"Expected 7 DOFs, got {len(q_manual)}")
                return

            print(f"Visualizing manual config: {q_manual}")
            viser_urdf_init.update_cfg(q_manual)
            
            # Project logic (reusing environment and helper)
            start_t = time.time()
            q_proj = vamp.panda.project_to_valid(q_manual, env)
            q_proj = np.array(q_proj).transpose().flatten()[:len(lower)]
            end_t = time.time()
            
            viser_urdf_proj.update_cfg(q_proj)
            print(f"Projected config: {q_proj}")
            print(f"Optimization took {end_t - start_t:.4f}s")
            
        except Exception as e:
            print(f"Error parsing input: {e}")

    btn_visual_config = server.gui.add_button("Visualize Input")
    btn_visual_config.on_click(lambda _: visualize_from_input())
    btn.on_click(lambda _: sample_and_project())
    
    print("Ready. Connect to Viser at http://localhost:8080")
    
    # Run loop
    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    main()