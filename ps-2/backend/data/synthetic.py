"""
Synthetic LiDAR Generator and Loader.
Generates realistic point cloud frames with ground surfaces and obstacle geometries for development and testing.
"""

from typing import Optional, List
import numpy as np
from backend.data.frame import LidarFrame
from backend.data.base import BaseLidarLoader


class SyntheticLidarGenerator:
    """
    Generates synthetic LiDAR point cloud frames with ground and obstacles.
    """

    def __init__(self, seed: Optional[int] = 42):
        self.rng = np.random.default_rng(seed)

    def generate_ground(
        self,
        num_points: int = 5000,
        radius_max: float = 30.0,
        ground_z: float = 0.0,
        noise_std: float = 0.03,
    ) -> np.ndarray:
        """
        Generates ground plane point cloud with realistic radial distribution.
        
        Returns:
            np.ndarray: Shape (N, 4) with [x, y, z, intensity].
        """
        # Radial distribution mimicking LiDAR beam spreads
        r = np.sqrt(self.rng.uniform(1.0, radius_max**2, size=num_points))
        theta = self.rng.uniform(-np.pi, np.pi, size=num_points)
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = ground_z + self.rng.normal(0, noise_std, size=num_points)
        
        # Ground reflectivity is typically lower (e.g. asphalt/soil: 0.10 - 0.35)
        intensity = self.rng.uniform(0.10, 0.35, size=num_points)
        
        return np.column_stack((x, y, z, intensity)).astype(np.float32)

    def generate_box_obstacle(
        self,
        center_x: float,
        center_y: float,
        size_x: float = 4.0,
        size_y: float = 2.0,
        height: float = 1.5,
        num_points: int = 600,
        intensity_range: tuple = (0.6, 0.9),
    ) -> np.ndarray:
        """
        Generates points on the outer surface of a box (e.g., car, container).
        """
        # Surface points along faces
        half_x, half_y = size_x / 2.0, size_y / 2.0
        
        # Distribute points on sides and top
        pts_per_side = num_points // 5
        sides = []
        
        # Front / Back faces
        y_side = self.rng.uniform(-half_y, half_y, size=pts_per_side)
        z_side = self.rng.uniform(0.0, height, size=pts_per_side)
        sides.append(np.column_stack((np.full(pts_per_side, half_x), y_side, z_side)))
        sides.append(np.column_stack((np.full(pts_per_side, -half_x), y_side, z_side)))
        
        # Left / Right faces
        x_side = self.rng.uniform(-half_x, half_x, size=pts_per_side)
        z_side2 = self.rng.uniform(0.0, height, size=pts_per_side)
        sides.append(np.column_stack((x_side, np.full(pts_per_side, half_y), z_side2)))
        sides.append(np.column_stack((x_side, np.full(pts_per_side, -half_y), z_side2)))
        
        # Roof face
        x_top = self.rng.uniform(-half_x, half_x, size=pts_per_side)
        y_top = self.rng.uniform(-half_y, half_y, size=pts_per_side)
        sides.append(np.column_stack((x_top, y_top, np.full(pts_per_side, height))))
        
        points_xyz = np.vstack(sides)
        points_xyz[:, 0] += center_x
        points_xyz[:, 1] += center_y
        
        intensity = self.rng.uniform(intensity_range[0], intensity_range[1], size=len(points_xyz))
        return np.column_stack((points_xyz, intensity)).astype(np.float32)

    def generate_cylinder_obstacle(
        self,
        center_x: float,
        center_y: float,
        radius: float = 0.4,
        height: float = 3.0,
        num_points: int = 250,
        intensity_range: tuple = (0.5, 0.85),
    ) -> np.ndarray:
        """
        Generates points on a vertical cylinder (e.g., pole, tree trunk).
        """
        theta = self.rng.uniform(0, 2 * np.pi, size=num_points)
        z = self.rng.uniform(0.0, height, size=num_points)
        x = center_x + radius * np.cos(theta)
        y = center_y + radius * np.sin(theta)
        intensity = self.rng.uniform(intensity_range[0], intensity_range[1], size=num_points)
        return np.column_stack((x, y, z, intensity)).astype(np.float32)

    def generate_frame(
        self,
        frame_id: int = 0,
        timestamp: float = 0.0,
        num_ground_points: int = 6000,
    ) -> LidarFrame:
        """
        Synthesizes a complete LiDAR frame with ground and multiple distinct obstacles.
        """
        ground = self.generate_ground(num_points=num_ground_points)
        
        # Obstacle 1: Vehicle ahead
        car_1 = self.generate_box_obstacle(
            center_x=12.0, center_y=2.5, size_x=4.2, size_y=1.8, height=1.5, num_points=700
        )
        
        # Obstacle 2: Small vehicle / object to the left
        car_2 = self.generate_box_obstacle(
            center_x=8.0, center_y=-5.0, size_x=3.0, size_y=1.6, height=1.4, num_points=500
        )
        
        # Obstacle 3: Pole / Streetlight to the right
        pole = self.generate_cylinder_obstacle(
            center_x=6.0, center_y=4.0, radius=0.25, height=3.5, num_points=250
        )
        
        # Obstacle 4: Tree trunk
        tree = self.generate_cylinder_obstacle(
            center_x=15.0, center_y=-7.0, radius=0.45, height=4.0, num_points=350
        )

        all_points = np.vstack([ground, car_1, car_2, pole, tree]).astype(np.float32)
        
        metadata = {
            "source": "synthetic",
            "ground_points_count": len(ground),
            "obstacle_points_count": len(car_1) + len(car_2) + len(pole) + len(tree),
            "num_obstacles": 4,
        }

        return LidarFrame(
            points=all_points,
            frame_id=frame_id,
            timestamp=timestamp,
            metadata=metadata,
        )

    def generate_temporal_sequence(
        self,
        scenario: str = "dynamic",
        num_ground_points: int = 4000,
        frame_rate_hz: float = 10.0,
    ) -> List[LidarFrame]:
        """
        Generates deterministic multi-frame sequences for temporal change detection testing.

        Scenarios:
          - 'static': Consecutive frames with identical terrain and obstacles.
          - 'new_obstacle': Frame 0 baseline, Frame 1 adds a new box obstacle.
          - 'removed_obstacle': Frame 0 has obstacle, Frame 1 removes it.
          - 'elevation_change': Frame 0 baseline, Frame 1 raises the elevation of an obstacle region.
          - 'dynamic': 4-frame sequence (Init -> Static -> Multi-Change -> Stabilized).
        """
        dt = 1.0 / max(0.001, frame_rate_hz)
        ground = self.generate_ground(num_points=num_ground_points)

        car_1 = self.generate_box_obstacle(center_x=12.0, center_y=2.5, size_x=4.2, size_y=1.8, height=1.5, num_points=600)
        pole = self.generate_cylinder_obstacle(center_x=6.0, center_y=4.0, radius=0.25, height=3.5, num_points=200)

        # Baseline Frame 0
        f0_pts = np.vstack([ground, car_1, pole]).astype(np.float32)
        f0 = LidarFrame(points=f0_pts, frame_id=0, timestamp=0.0, metadata={"scenario": scenario, "stage": "initial"})

        if scenario == "static":
            f1 = LidarFrame(points=f0_pts.copy(), frame_id=1, timestamp=dt, metadata={"scenario": scenario, "stage": "static_repeat"})
            return [f0, f1]

        elif scenario == "new_obstacle":
            new_box = self.generate_box_obstacle(center_x=-8.0, center_y=-5.0, size_x=3.0, size_y=2.0, height=1.6, num_points=400)
            f1_pts = np.vstack([ground, car_1, pole, new_box]).astype(np.float32)
            f1 = LidarFrame(points=f1_pts, frame_id=1, timestamp=dt, metadata={"scenario": scenario, "stage": "new_obstacle"})
            return [f0, f1]

        elif scenario == "removed_obstacle":
            # Frame 1 removes car_1
            f1_pts = np.vstack([ground, pole]).astype(np.float32)
            f1 = LidarFrame(points=f1_pts, frame_id=1, timestamp=dt, metadata={"scenario": scenario, "stage": "removed_obstacle"})
            return [f0, f1]

        elif scenario == "elevation_change":
            # Frame 1 elevates car_1 (+1.5m taller)
            elevated_car = self.generate_box_obstacle(center_x=12.0, center_y=2.5, size_x=4.2, size_y=1.8, height=3.0, num_points=600)
            f1_pts = np.vstack([ground, elevated_car, pole]).astype(np.float32)
            f1 = LidarFrame(points=f1_pts, frame_id=1, timestamp=dt, metadata={"scenario": scenario, "stage": "elevated_obstacle"})
            return [f0, f1]

        elif scenario == "dynamic":
            # Frame 1: Identical / static repeat
            f1 = LidarFrame(points=f0_pts.copy(), frame_id=1, timestamp=dt, metadata={"scenario": "dynamic", "stage": "static"})
            
            # Frame 2: New obstacle at (-8, -5), removed pole, elevated car
            new_box = self.generate_box_obstacle(center_x=-8.0, center_y=-5.0, size_x=3.0, size_y=2.0, height=1.6, num_points=400)
            elevated_car = self.generate_box_obstacle(center_x=12.0, center_y=2.5, size_x=4.2, size_y=1.8, height=3.0, num_points=600)
            f2_pts = np.vstack([ground, elevated_car, new_box]).astype(np.float32)
            f2 = LidarFrame(points=f2_pts, frame_id=2, timestamp=2 * dt, metadata={"scenario": "dynamic", "stage": "multi_change"})

            # Frame 3: Stabilized scene identical to Frame 2
            f3 = LidarFrame(points=f2_pts.copy(), frame_id=3, timestamp=3 * dt, metadata={"scenario": "dynamic", "stage": "stabilized"})
            return [f0, f1, f2, f3]

        else:
            raise ValueError(f"Unknown temporal scenario '{scenario}'. Available: 'static', 'new_obstacle', 'removed_obstacle', 'elevation_change', 'dynamic'")


class SyntheticLidarLoader(BaseLidarLoader):
    """
    Synthetic dataset loader providing a sequence of simulated LiDAR frames.
    """

    def __init__(self, num_frames: int = 5, frame_rate_hz: float = 10.0, seed: Optional[int] = 42):
        self.num_frames = max(1, num_frames)
        self.dt = 1.0 / frame_rate_hz
        self.generator = SyntheticLidarGenerator(seed=seed)
        self._cached_frames: List[Optional[LidarFrame]] = [None] * self.num_frames

    def __len__(self) -> int:
        return self.num_frames

    def get_frame(self, index: int) -> LidarFrame:
        if index < 0 or index >= self.num_frames:
            raise IndexError(f"Index {index} out of bounds for {self.num_frames} frames.")
        
        if self._cached_frames[index] is None:
            frame = self.generator.generate_frame(
                frame_id=index,
                timestamp=index * self.dt,
            )
            self._cached_frames[index] = frame
            
        return self._cached_frames[index]
