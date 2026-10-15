import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple

class ComplexContour(ABC):
    """
    Abstract base class for parameterized complex contours z(t) for t in [0, 1].
    """
    
    @abstractmethod
    def evaluate(self, t: float) -> complex:
        """Evaluate z(t) for t in [0, 1]."""
        pass

    @abstractmethod
    def derivative(self, t: float) -> complex:
        """Evaluate the derivative z'(t) for t in [0, 1]."""
        pass

    def sample(self, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearly sample points along the contour.
        Returns (t_values, z_values).
        """
        t_vals = np.linspace(0.0, 1.0, num_points)
        z_vals = np.array([self.evaluate(t) for t in t_vals], dtype=complex)
        return t_vals, z_vals

    def sample_adaptive(self, initial_points: int = 50, tolerance: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adaptively sample the contour based on curvature / step size to capture features accurately.
        """
        t_vals = [0.0, 1.0]
        z_vals = [self.evaluate(0.0), self.evaluate(1.0)]
        
        iterations = 0
        max_iterations = 6
        while iterations < max_iterations:
            refined = False
            new_t_vals = []
            for i in range(len(t_vals) - 1):
                t1, t2 = t_vals[i], t_vals[i+1]
                t_mid = 0.5 * (t1 + t2)
                z1, z2 = z_vals[i], z_vals[i+1]
                z_mid = self.evaluate(t_mid)
                
                chord_mid = 0.5 * (z1 + z2)
                dist = abs(z_mid - chord_mid)
                
                new_t_vals.append(t1)
                if dist > tolerance:
                    new_t_vals.append(t_mid)
                    refined = True
            new_t_vals.append(t_vals[-1])
            
            t_vals = sorted(list(set(new_t_vals)))
            z_vals = [self.evaluate(t) for t in t_vals]
            
            if not refined or len(t_vals) > 500:
                break
            iterations += 1
            
        return np.array(t_vals), np.array(z_vals)

class Circle(ComplexContour):
    """
    A circular contour defined by center, radius, and orientation (counter-clockwise or clockwise).
    """
    def __init__(self, center: complex, radius: float, ccw: bool = True):
        self.center = center
        self.radius = radius
        self.ccw = ccw
        
    def evaluate(self, t: float) -> complex:
        angle = 2.0 * np.pi * t
        if not self.ccw:
            angle = -angle
        return self.center + self.radius * np.exp(1j * angle)
        
    def derivative(self, t: float) -> complex:
        angle = 2.0 * np.pi * t
        direction = 1.0 if self.ccw else -1.0
        return 2.0j * np.pi * direction * self.radius * np.exp(1j * (direction * angle))

class Line(ComplexContour):
    """
    A straight line segment from start to end complex points.
    """
    def __init__(self, start: complex, end: complex):
        self.start = start
        self.end = end
        
    def evaluate(self, t: float) -> complex:
        return self.start + t * (self.end - self.start)
        
    def derivative(self, t: float) -> complex:
        return self.end - self.start

class Polygonal(ComplexContour):
    """
    A contour built from an ordered list of vertices, connecting them with line segments.
    """
    def __init__(self, vertices: List[complex]):
        if len(vertices) < 2:
            raise ValueError("Polygonal contour requires at least 2 vertices.")
        self.vertices = vertices
        self.num_segments = len(vertices) - 1
        
    def _get_segment(self, t: float) -> Tuple[int, float]:
        if t >= 1.0:
            return self.num_segments - 1, 1.0
        if t <= 0.0:
            return 0, 0.0
        scaled_t = t * self.num_segments
        idx = int(np.floor(scaled_t))
        local_t = scaled_t - idx
        return idx, local_t
        
    def evaluate(self, t: float) -> complex:
        idx, local_t = self._get_segment(t)
        z_start = self.vertices[idx]
        z_end = self.vertices[idx+1]
        return z_start + local_t * (z_end - z_start)
        
    def derivative(self, t: float) -> complex:
        idx, _ = self._get_segment(t)
        z_start = self.vertices[idx]
        z_end = self.vertices[idx+1]
        return (z_end - z_start) * self.num_segments
