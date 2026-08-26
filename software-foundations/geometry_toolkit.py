'''
Project: Spatial Geometry Engine: 2D & 3D Object-Oriented Modeling
Description: An OOP-based spatial engine that models points, rectangles, 
and 3D prisms. Includes methods for distance, vectors, area, volume, 
and automated 2D/3D visualizations using matplotlib.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import product
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 1. 2D Geometry Classes
class Point2D:
    """Represents a coordinate point in a 2D Cartesian plane."""
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        
    def __str__(self):
        return f"({self.x}, {self.y})"
    
    def get_quadrant(self):
        """Identifies the spatial quadrant of the point."""
        if self.x > 0 and self.y > 0: return 'First Quadrant'
        elif self.x < 0 and self.y > 0: return 'Second Quadrant'
        elif self.x < 0 and self.y < 0: return 'Third Quadrant'
        elif self.x > 0 and self.y < 0: return 'Fourth Quadrant'
        elif self.x == 0 and self.y == 0: return 'Origin'
        else: return 'On an Axis'
            
    def get_vector_to(self, target_point):
        """Calculates the directional vector to another point."""
        return (target_point.x - self.x, target_point.y - self.y)
        
    def get_distance_to(self, target_point):
        """Calculates the Euclidean distance to another point."""
        return np.sqrt((target_point.x - self.x)**2 + (target_point.y - self.y)**2)

class Rectangle:
    """Represents a 2D rectangle defined by two opposite corner points."""
    def __init__(self, p1=Point2D(), p2=Point2D()):
        self.p1 = p1
        self.p2 = p2
        
    def width(self):
        return abs(self.p1.x - self.p2.x)
        
    def height(self):
        return abs(self.p1.y - self.p2.y)
        
    def area(self):
        return self.width() * self.height()
    
    @property
    def center(self):
        """Calculates the geometric centroid of the rectangle."""
        cx = (self.p1.x + self.p2.x) / 2.0
        cy = (self.p1.y + self.p2.y) / 2.0
        return Point2D(cx, cy)

# 2. 2D Visualization Engine
def plot_rectangle(rect):
    """Renders a 2D Rectangle on a Cartesian plane."""
    fig, ax = plt.subplots(figsize=(8, 6)) 
    
    # Identify bottom-left origin for the patch
    min_x = min(rect.p1.x, rect.p2.x)
    min_y = min(rect.p1.y, rect.p2.y)
    max_x = max(rect.p1.x, rect.p2.x)
    max_y = max(rect.p1.y, rect.p2.y)
    
    # Create and add the rectangle patch
    label_text = f'Area: {rect.area()} sq units\nWidth: {rect.width()}\nHeight: {rect.height()}'
    rect_patch = patches.Rectangle(
        (min_x, min_y), rect.width(), rect.height(), 
        facecolor='lightblue', edgecolor='blue', alpha=0.6, label=label_text
    )
    ax.add_patch(rect_patch)
    
    # Plot corners and center (The "Plus" Feature)
    ax.plot(rect.p1.x, rect.p1.y, 'go', markersize=8, label=f'P1 {rect.p1}')
    ax.plot(rect.p2.x, rect.p2.y, 'ro', markersize=8, label=f'P2 {rect.p2}')
    ax.plot(rect.center.x, rect.center.y, 'k+', markersize=12, label=f'Center {rect.center}')
    
    # Formatting
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    
    margin = 3
    plt.xlim(min_x - margin, max_x + margin)
    plt.ylim(min_y - margin, max_y + margin)
    plt.title("2D Rectangle Visualization")
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

# 3. 3D Geometry Classes (Inheritance)
class Point3D(Point2D):
    """Extends Point2D into a 3D spatial coordinate."""
    def __init__(self, x=0, y=0, z=0):
        super().__init__(x, y)
        self.z = z
    
    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"
    
    def get_distance_to(self, target_point):
        d_2d = super().get_distance_to(target_point)
        dz = target_point.z - self.z
        return np.sqrt(d_2d**2 + dz**2)
    
    def get_vector_to(self, target_point):
        v_2d = super().get_vector_to(target_point)
        dz = target_point.z - self.z
        return (*v_2d, dz)

class Prism(Rectangle):
    """Extends Rectangle into a 3D Rectangular Prism."""
    def __init__(self, p1=Point3D(), p2=Point3D()):
        super().__init__(p1, p2)
        
    def depth(self):
        return abs(self.p1.z - self.p2.z)
        
    def volume(self):
        return self.area() * self.depth()
    
    @property
    def center(self):
        """Calculates the 3D centroid of the prism."""
        cx = (self.p1.x + self.p2.x) / 2.0
        cy = (self.p1.y + self.p2.y) / 2.0
        cz = (self.p1.z + self.p2.z) / 2.0
        return Point3D(cx, cy, cz)

# 4. 3D Visualization Engine
def plot_prism(prism):
    """Renders a 3D Prism on a spatial axis."""
    x0, x1 = sorted([prism.p1.x, prism.p2.x])
    y0, y1 = sorted([prism.p1.y, prism.p2.y])
    z0, z1 = sorted([prism.p1.z, prism.p2.z])
    
    vertices = list(product([x0, x1], [y0, y1], [z0, z1]))
    
    faces = [
        [vertices[i] for i in [0, 4, 6, 2]],  # Bottom (z = z0)
        [vertices[i] for i in [1, 5, 7, 3]],  # Top (z = z1)
        [vertices[i] for i in [0, 1, 3, 2]],  # Back (x = x0)
        [vertices[i] for i in [4, 5, 7, 6]],  # Front (x = x1)
        [vertices[i] for i in [0, 1, 5, 4]],  # Left (y = y0)
        [vertices[i] for i in [2, 3, 7, 6]],  # Right (y = y1)
    ]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    # Draw Faces
    ax.add_collection3d(Poly3DCollection(faces, facecolor='lightpink', edgecolors='blue', alpha=0.3))
    
    # Plot Corner Points and Center
    ax.scatter([prism.p1.x, prism.p2.x], [prism.p1.y, prism.p2.y], [prism.p1.z, prism.p2.z], color='red', s=50)
    ax.scatter(*[getattr(prism.center, axis) for axis in ['x', 'y', 'z']], color='black', marker='+', s=100, label=f'Center {prism.center}')
    
    # Annotations
    ax.text(prism.p1.x, prism.p1.y, prism.p1.z, f' P1 {prism.p1}', color='darkred')
    ax.text(prism.p2.x, prism.p2.y, prism.p2.z, f' P2 {prism.p2}', color='darkred')
    ax.set_title(f'3D Prism Visualization\nVolume: {prism.volume()} cubic units')
    
    margin = 2
    ax.set_xlim([x0 - margin, x1 + margin])
    ax.set_ylim([y0 - margin, y1 + margin])
    ax.set_zlim([z0 - margin, z1 + margin])
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    
    plt.legend()
    plt.show()

# 5. Execution Block
if __name__ == '__main__':
    print("="*50)
    print("TESTING 2D GEOMETRY")
    print("="*50)
    
    A = Point2D(2, 3)
    B = Point2D(5, -5)
    
    print(f"Point A {A} is in the {A.get_quadrant()}.")
    print(f"The resulting vector from B {B} to A {A} is {B.get_vector_to(A)}.")
    
    rect_AB = Rectangle(A, B)
    plot_rectangle(rect_AB)
    
    print("\n" + "="*50)
    print("TESTING 3D GEOMETRY")
    print("="*50)
    
    E = Point3D(0, 0, 0)
    F = Point3D(1, 2, 3)
    
    prism_EF = Prism(E, F)
    print(f"Prism created from {E} to {F}.")
    print(f"Volume of Prism: {prism_EF.volume()} cubic units.")
    
    plot_prism(prism_EF)
