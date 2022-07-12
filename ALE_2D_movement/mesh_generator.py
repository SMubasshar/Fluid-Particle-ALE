import sys
from termios import FF1
from numpy.lib.function_base import iterable
import admesh4py.admesh4py as admesh
from dolfin import *
import mshr
import numpy as np 
import itertools


def py_snap_boundary(mesh, sub_domain):
    boundary = BoundaryMesh(mesh, "exterior")
    dim = mesh.geometry().dim()
    x = boundary.coordinates()
    for i in range(0, boundary.num_vertices()):
        sub_domain.snap(x[i, :])
    ALE.move(mesh, boundary)


class Ball(SubDomain):
    radius = 1
    dim = 2

    def snap(self, x):
        r = sqrt(self.sqr_dist(Point(x)))
        if r < self.radius:
            for i in range(self.dim):
                x[i] = self.center[i] + (self.radius / r) * (x[i] -
                                                             self.center[i])

    def inside(self, x, on_boundary):
        return ((self.sqr_dist(x)<= 1.01 * self.radius**2) and on_boundary)

    def __init__(self, center: tuple, radius: float,rho=0):
        self.center = Point(center)
        self.radius = radius
        self.rho = rho
        self.dim = len(center)
        super().__init__()

    def sqr_dist(self, point):
        if not isinstance(point, Point):
            point = Point(point)
        a = self.center - point
        return a[0]**2+a[1]**2

    def dist(self, point):
        return sqrt(self.sqr_dist(point))

    def is_touching(self, other):
        if self.sqr_dist(other.center) < (self.radius + other.radius)**2:
            return True
        else:
            return False

    def ball_mesh(self):
        if self.dim == 2:
            return mshr.Circle(self.center, self.radius)
        if self.dim == 3:
            return mshr.Sphere(self.center, self.radius,64)
        else:
            return None
        
    def get_radius(self):
        return self.radius
    
    def dist_string(self):
        return f"((x[0]-({self.center[0]}))*(x[0]-({self.center[0]}))) + ((x[1]-({self.center[1]}))*(x[1]-({self.center[1]})))-({self.radius}*{self.radius})"

    def is_boundary(self):
        return f"((x[0]-({self.center[0]}))*(x[0]-({self.center[0]}))) + ((x[1]-({self.center[1]}))*(x[1]-({self.center[1]})))-({self.radius}*{self.radius})"

    def update(self, dt, v_x, v_y ,omg):
        self.center = Point((self.center[0] + v_x*dt,self.center[1] + v_y*dt))

class Ellipse(Ball):
    radius = 1
    dim = 2

    def snap(self, x):
        y = Point(x)
        dist = sqrt((y[0]-self.center[0])**2 + (y[1]-self.center[1])**2)
        d = y - self.center
        d = d/(sqrt(d[0]**2 + d[1]**2))
        cos = d[0]*self.dir[0] + d[1]*self.dir[1]
        sin = d[1]*self.dir[0] - d[0]*self.dir[1]
        
        r = (self.radius**2)/(np.sqrt((sin*self.el*self.radius)**2 + (cos*self.radius/self.el)**2))
        if dist < r:
            for i in [0,1]:
                x[i] = self.center[i] + (r/dist) * (x[i] - self.center[i])

    def inside(self, x, on_boundary):
        a,b = self.get_axes()
        return ((self.dist(x) <= 1.005*a) and on_boundary)

    def __init__(self, center: tuple,dir:tuple,el:float, radius: float,rho=0):
        dir=np.array(dir)
        self.dir = dir/np.linalg.norm(dir)
        self.el = el
        self.dim = len(center)
        super().__init__(center,radius,rho)

    def get_axes(self):
        if self.el >= 1:
            a = self.radius*self.el
            b = self.radius/self.el
        else:
            a = self.radius/self.el
            b = self.radius*self.el
        return a,b

    def get_focals(self):
        a,b = self.get_axes()
        
        c = np.sqrt(np.abs(a**2 - b**2))
        
        dir = self.dir
        if self.el < 1:
            u = dir[0]
            dir[0] = -dir[1]
            dir[1] = u
        
        f1 = Point([self.center[0] + (c*dir[0]),self.center[1] + (c*dir[1])])  
        f2 = Point([self.center[0] - (c*dir[0]),self.center[1] - (c*dir[1])])
        
        return f1, f2

    def dist(self, point):   
        f1, f2 =self.get_focals()
        
        if not isinstance(point, Point):
            point = Point(point)
        
        d1 = point - f1
        d2 = point - f2
        
        return (np.sqrt(d1[0]**2+d1[1]**2)+np.sqrt(d2[0]**2+d2[1]**2))/2

    def is_touching(self, other):
        if self.dist(other.center) < (self.radius + other.radius):
            return True
        else:
            return False

    def ball_mesh(self):
        a = self.radius*self.el
        b = self.radius/self.el
        
        x=self.dir
        y=np.ndarray([2])
        y[0]=-x[1]
        y[1]=x[0]
        
        points=[x*a*np.cos(theta) + y*b*np.sin(theta) for theta in np.linspace(0,2*np.pi,10)]
        points = [Point(self.center[0]+p[0],self.center[1]+p[1]) for p in points]
        if self.dim == 2:
            return mshr.Polygon(points)
        else:
            return None
        
    def get_radius(self):
        a,b=self.get_axes()
        return a

    def dist_string(self):
        f1,f2 = self.get_focals()
        a,b = self.get_axes()
        focal_dist=f"( sqrt( ((x[0]-{f1[0]})*(x[0]-{f1[0]})) + ((x[1]-{f1[1]})*(x[1]-{f1[1]})) ) + sqrt( ((x[0]-{f2[0]})*(x[0]-{f2[0]})) + ((x[1]-{f2[1]})*(x[1]-{f2[1]})) ) )"
        return focal_dist + "*" + focal_dist + f"/4 -({a*a})"

    def update(self, dt, v_x, v_y ,omg):
        self.center = Point((self.center[0] + v_x*dt,self.center[1] + v_y*dt))
        R = np.array([[np.cos(dt*omg),-np.sin(dt*omg)],[np.sin(dt*omg),np.cos(dt*omg)]]) 
        self.dir=R@self.dir


comm = MPI.comm_world
rank = comm.Get_rank()
remesh_params = {"num_iter":5,
                 "Marge_cells":True,
	         "Add_Vertex_On_Edge":True,
	         "Flip_Edges":True,
                 "Remove_Bndry_Edge":True,
	         "Remove_Vertices":True,
                 "Move_Vertices":False,
	         "Report":True}


def create_mesh(dimensions,
                  balls: iterable,
                  refinement_perimeters=[0.4, 0.2, 0.05],
                  leftbottom=(0,0),
                  unique_outer2D=False,
                  mesh_density=32,
                  mark_bndry=True,
                  circular=False,
                  **kwargs
                  ):
    """Createsh mesh with holes of specified shape.

    Args:
        dimensions (tuple): Two or three dimensional tuple specifying size of a bounding box.
        balls (iterable): Instance of Ball object or list of Ball objects.
        refinement_perimeters (list, optional): Distances from a ball surface, where the mesh will be refined. Defaults to [0.4, 0.2, 0.05].

    Returns:
        (Mesh,MeshFunction): Output MeshFunction has order of ball in balls list as a value on a surface of ball. The value on the outer boundary is len(balls)+1.
    """
    dim = len(dimensions)

    if isinstance(balls, Ball):
        balls = [balls]

    righttop=[x+y for x,y in zip(leftbottom,dimensions)]
    if circular:
        geometry = mshr.Circle(Point((0,0)),dimensions[1],segments=256) - mshr.Circle(Point((0,0)),dimensions[0],segments=180)
    else:
        if dim == 2:
            geometry = mshr.Rectangle(Point(leftbottom), Point(righttop))
        elif dim == 3:
            geometry = mshr.Box(Point(leftbottom), Point(righttop))
        else:
            raise Exception(f"Incorrect dimension: {dim}")

    for ball1, ball2 in itertools.combinations(balls, 2):
        if not (ball1.dim == dim):
            raise Exception("Ball dimension does not match domain dimension",
                            ball1)
        #if not (ball1 == ball2) and ball1.is_touching(ball2):
            #raise Exception("Circle overlap detected!", ball1, ball2)

    for ball in balls:
        geometry = geometry - ball.ball_mesh()

    mesh = mshr.generate_mesh(geometry, mesh_density)
    
    def on_outer_boundary(point):
        if not unique_outer2D:
            for bound in (leftbottom,righttop):
                for coord, bound_coord in zip(point, bound):
                    if near(coord, bound_coord) or near(coord, bound_coord):
                        return 1
        elif unique_outer2D:
            if near(leftbottom[0], point[0]):
                return 1
            if near(righttop[0], point[0]):
                return 2
            if near(leftbottom[1], point[1]):
                return 3
            if near(righttop[1], point[1]):
                return 4
        return 0

    def mark_borders(mesh):
        bndry = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

        for i,b in enumerate(balls):
            b.mark(bndry, i + 1)

        for f in facets(mesh):
            if f.exterior():
                mp = f.midpoint()
                bd=on_outer_boundary(mp)
                if bd > 0:
                    bndry[f] = len(balls) + bd
        return (bndry)

    for r in refinement_perimeters:
        info("refining distance {}".format(r))
        cf = MeshFunction('bool', mesh, mesh.topology().dim(), False)
        for c in cells(mesh):
            for ball in balls:
                if ball.dist(c.midpoint()) - ball.get_radius() < r:
                    cf[c] = True

        mesh = refine(mesh, cf)
        for ball in balls:
            py_snap_boundary(mesh, ball)
    
    
    if(mark_bndry):
        bndry = mark_borders(mesh)
    else:
        bndry=None
    return mesh, bndry


if __name__ == "__main__":

    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = "mesh"

    _mesh, bndry = create_mesh((5,3), Ball((2.5, 0),1),refinement_perimeters=[])
    mesh_file = XDMFFile(_mesh.mpi_comm(), "test_mesh.xdmf")
    mesh_file.write(_mesh)  
