from typing import Tuple
from dolfin import *
from ALE_solver import FluidSystem
from mesh_generator import Ball, Ellipse
import numpy as np
import random
import sys

from datetime import date

V_boundary=float(sys.argv[1])
distr=sys.argv[2]

#type="init"
type="started"

# Adjustments of parent class to modify the boundary conditions and add force computation 
class ShearTest(FluidSystem):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
    
    def get_bcs(self):
        ball_count = len(self.balls)

        if type=="init":
            self.V_current=Expression(("U*fmin(0.2*t/0.04e-4,1)","0.0"),t=self.t,U=V_boundary,degree=2)
        if type=="started":
            self.V_current=Expression(("U*fmin(0.2*1/0.04e-4,1)","0.0"),t=self.t,U=V_boundary,degree=2)
        
        bc_b = DirichletBC(self.W.sub(0), Constant((0.0,0.0)), self.bndry,
                          ball_count + 3)
        bc_t = DirichletBC(self.W.sub(0), self.V_current, self.bndry,
                           ball_count + 4)
        bc_l = DirichletBC(self.W.sub(0).sub(1), Constant(0.0), self.bndry,
                          ball_count + 1)
        bc_r = DirichletBC(self.W.sub(0).sub(1), Constant(0.0), self.bndry,
                          ball_count + 2)
        fluid_bcs = [bc_l,bc_r,bc_b,bc_t]

        bc_balls=[]
        for i,ball in enumerate(self.balls):
            bc_balls.append( DirichletBC(self.MV.sub(0), self.w.sub(2).sub(2*i), self.bndry, i+1) )
            bc_balls.append( DirichletBC(self.MV.sub(1), self.w.sub(2).sub(2*i+1), self.bndry, i+1) )
        bc_l = DirichletBC(self.MV, Constant((0, 0)), self.bndry,
                           ball_count + 1)
        bc_r = DirichletBC(self.MV, Constant((0, 0)), self.bndry,
                           ball_count + 2)
        bc_b = DirichletBC(self.MV.sub(1),Constant( 0), self.bndry,
                           ball_count + 3)
        bc_t = DirichletBC(self.MV.sub(1), Constant( 0), self.bndry,
                           ball_count + 4)
        mesh_bcs = bc_balls+[bc_l, bc_r, bc_b, bc_t]

        return fluid_bcs, mesh_bcs
    
    def solve_step(self):
        self.V_current.t = self.t
        return super().solve_step()
    
    def end_step_callback(self):
        D = sym(grad(self.v))
        ball_count = len(self.balls)
        self.shear_stress=assemble(2*self.nu*D[0,1]*self.dx)/self.vol_f
        self.force_t=assemble(2*self.nu*D[0,1]*self.ds(ball_count+4))
        self.force_b=assemble(2*self.nu*D[0,1]*self.ds(ball_count+3))
        #omg=assemble((self.omg[0]*self.dx))/self.vol_f
        #self.print(f"{omg}")
        return 

    
    def set_IC(self):
        if type=="init":
            self.w0.interpolate(Expression(( "0","0", "0")+tuple((a for b in [(f"0","0") for b in self.balls] for a in b ))+(len(self.balls)*tuple(["0"])),h=self.meshsizes[1],V=V_boundary,degree=2))
        if type=="started":
            self.w0.interpolate(Expression(( "(x[1]/h)*V","0", "0")+tuple((a for b in [(f"({b.center[1]}/h)*V","0") for b in self.balls] for a in b ))+(len(self.balls)*tuple(["0"])),h=self.meshsizes[1],V=V_boundary,degree=2))
        self.w.assign(self.w0)
    
def generate_rdist(xlim,ylim,r,count):
    points=[]
    for i in range(count):
        for j in range(100):
            x=random.uniform(xlim[0],xlim[1])
            y=random.uniform(ylim[0],ylim[1])
            valid=True
            for p in points:
                if np.sqrt((x-p[0])**2 + (y-p[1])**2) < 2.1*r:
                    valid=False
                    break
            if valid:
                points.append((x,y))
                break
    return points

# rho = 1050 kg/m^3
# mu = 3.896 * 10e^-3
# v = 0.65 m/s
# r = 4 1e-6
# h = 100 1e-6
# Re = 0.3     

t_end=20e-4
print(V_boundary)
w=500e-6
h=100e-6
#w=1000e-6
h=60e-6

t_end=150e-4
#w=1000e-6

random.seed(4)


d1 = date.today().strftime("%m%d")

if distr == "unif":
    #balls=[Ball((p[0],p[1]), 4e-6,1050)  for p in generate_rdist((10e-6,60e-6),(25e-6,85e-6),4e-6,20)]
    #balls=[Ball((p[0],p[1]), 4e-6,1050)  for p in generate_rdist((10e-6,110e-6),(12e-6,h-10e-6),4e-6,40)]
    balls=[Ball((p[0],p[1]), 4e-6,1050)  for p in generate_rdist((10e-6,110e-6),(10e-6,h-15e-6),4e-6,20)]
    #balls=[Ball((p[0],p[1]), 4e-6,1050)  for p in generate_rdist((10e-6,210e-6),(10e-6,h-15e-6),4e-6,40)]
if distr == "grid":
    #balls=[Ball((10e-6+(i*13e-6),20e-6+(j*13e-6)), 4e-6,1050) for i in range(4) for j in range(5)]
    balls=[Ball((10e-6+(i*13e-6),10e-6+(j*11e-6)), 4e-6,1050) for i in range(5) for j in range(4)]
    #balls=[Ball((10e-6+(i*13e-6),10e-6+(j*11e-6)), 4e-6,1050) for i in range(10) for j in range(4)]
#balls=[Ellipse((2,3),(0,1),1.3,0.5,rho=1)]
#balls=[Ball((2,3),0.5,rho=1)]

problem_name=f"{d1}_shear_{len(balls)}_thin_{type}_{distr}_{V_boundary}"

problem=ShearTest(problem_name, 
                   balls, 
                   (w, h),
                   mesh_perimeters=[10e-6,5e-6,2e-6], 
                   dt=0.03e-4 if V_boundary <1.1 else ( 0.06e-5 if V_boundary <3.1 else 0.03e-5),
                   nu=3.896e-3,
                   rho_fluid=1050,
                   starting_density=0.45,
                   density_falloff=4e9,
                   density_cap=(22e-6)**2,
                   remeshing_period=int(round(9-(4*min(V_boundary,1.5)))),
                   linear_density_falloff=False,
                   linear=False,
                   allow_rotation=True,
                   mesh_density=160,
                   beta_h=10e8, 
                   tolerance=5e-9
                   ) 

init=True

while t < t_end:
    t = problem.solve_step()

    problem.print(problem.force_t)
    problem.print(problem.force_b)
    
    with open(f"results/result_{problem_name}/forces.csv","w" if init else "a") as f:
        print(f"{t},{problem.force_t},{problem.force_b}",file=f)
    
    init=False
