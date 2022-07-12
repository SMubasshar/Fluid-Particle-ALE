from ast import Expression
from typing import Tuple
from dolfin import *
from ALE_solver import FluidSystem
from mesh_generator import Ball, Ellipse
import numpy as np
import random
import sys
import pandas as pd

from datetime import date

V_boundary=float(sys.argv[1])
distr=sys.argv[2]

#type="init"
type="started"

# rho = 1050 kg/m^3
# mu = 3.896 * 10e^-3
# v = 0.65 m/s
# r = 4 1e-6
# h = 100 1e-6
# Re = 0.3     

t_end=20e-4
print(V_boundary)
r1=30e-6
r2=60e-6

t_end=500e-4

# Adjustments of parent class to modify the boundary conditions and add force computation 
class ShearTest(FluidSystem):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
    
    def get_bcs(self):
        ball_count = len(self.balls)

        if type=="init":
            self.V_current=Expression(("-(x[1]/h)*U*fmin(0.2*t/0.04e-4,1)","(x[0]/h)*U*fmin(0.2*t/0.04e-4,1)"),t=self.t,U=V_boundary,h=self.meshsizes[1],degree=2)
        if type=="started":
            self.V_current=Expression(("-(x[1]/h)*U*fmin(0.2*1/0.04e-4,1)","(x[0]/h)*U*fmin(0.2*1/0.04e-4,1)"),t=self.t,U=V_boundary,h=self.meshsizes[1],degree=2)
        
        It_facet = SubsetIterator(self.bndry,ball_count + 1)
        argmin=None
        mindist=1
        x=Point((self.meshsizes[0],0))
        for c in It_facet:
            d=(c.midpoint()-x)
            dist=d[0]**2+d[1]**2
            if dist<mindist:
                mindist=dist
                argmin=c

        self.bndry_p=MeshFunction('size_t', self.mesh, 1,0)  
        self.bndry_p[argmin]=2

        with XDMFFile(self.comm,"results/results_%s/boundary_p.xdmf" % self.filename) as xdmf:
            xdmf.write(self.bndry_p)

        bc_p_leftbot = DirichletBC(self.W.sub(1), Constant(0.0),self.bndry_p,2)
        
        bc_i = DirichletBC(self.W.sub(0), Constant((0.0,0.0)), self.bndry,
                          ball_count + 1)
        bc_o = DirichletBC(self.W.sub(0), self.V_current, self.bndry,
                           ball_count + 2)

        fluid_bcs = [bc_i,bc_o,bc_p_leftbot]

        bc_balls=[]

        self.v_bound_o=Expression((f"(1.0/{self.dt})*(( ( cos({self.dt/r2}*v ) -1.0)*x[0])-(sin({self.dt/r2}*v)*x[1]))",f"(1.0/{self.dt})*((sin({self.dt/r2}*v)*x[0])+((cos({self.dt/r2}*v)-1.0)*x[1]))"),v=0,degree=2)
        self.v_bound_i=Expression((f"(1.0/{self.dt})*(( ( cos({self.dt/r1}*v ) -1.0)*x[0])-(sin({self.dt/r1}*v)*x[1]))",f"(1.0/{self.dt})*((sin({self.dt/r1}*v)*x[0])+((cos({self.dt/r1}*v)-1.0)*x[1]))"),v=0,degree=2)
        
        v_o= np.sqrt(max([self.ballvel[2*i]**2+self.ballvel[2*i+1]**2 for i in range(len(self.balls))]))
        v_i= np.sqrt(min([self.ballvel[2*i]**2+self.ballvel[2*i+1]**2 for i in range(len(self.balls))]))
       
        self.v_bound_o.v=np.min([v_o,0.9*V_boundary])
        self.v_bound_i.v=np.min([v_i,0.9*V_boundary])
        
        for i,ball in enumerate(self.balls):
            bc_balls.append( DirichletBC(self.MV.sub(0), self.w.sub(2).sub(2*i), self.bndry, i+1) )
            bc_balls.append( DirichletBC(self.MV.sub(1), self.w.sub(2).sub(2*i+1), self.bndry, i+1) )
        m_bc_i = DirichletBC(self.MV, self.v_bound_i, self.bndry,
                          ball_count + 1)
        m_bc_o = DirichletBC(self.MV, self.v_bound_o, self.bndry,
                           ball_count + 2)
        mesh_bcs = bc_balls+[m_bc_i,m_bc_o]

        return fluid_bcs, mesh_bcs
    
    def solve_step(self):
        self.V_current.t = self.t
        return super().solve_step()
    
    def end_step_callback(self):

        v_o= np.sqrt(max([self.ballvel[2*i]**2+self.ballvel[2*i+1]**2 for i in range(len(self.balls))]))
        v_i= np.sqrt(min([self.ballvel[2*i]**2+self.ballvel[2*i+1]**2 for i in range(len(self.balls))]))

        self.v_bound_o.v=np.min([v_o,0.9*V_boundary])
        self.v_bound_i.v=np.min([v_i,0.9*V_boundary])

        D = sym(grad(self.v))
        ball_count = len(self.balls)
        r=Expression(("x[0]","x[1]"),degree=2)
        self.force_t=assemble( ((2*self.nu)*(D[0,1]*sqrt(r[0]**2+r[1]**2)+(r[0]*r[1]/sqrt(r[0]**2+r[1]**2))*(D[1,1]-D[0,0])))*self.ds(ball_count+2) )
        self.force_b=-assemble( ((2*self.nu)*(D[0,1]*sqrt(r[0]**2+r[1]**2)+(r[0]*r[1]/sqrt(r[0]**2+r[1]**2))*(D[1,1]-D[0,0])))*self.ds(ball_count+1) )
        return 

    
    def set_IC(self):
        if type=="init":
            self.w0.interpolate(Expression(( "0","0", "0")+tuple((a for b in [(f"0","0") for b in self.balls] for a in b ))+(len(self.balls)*tuple(["0"])),h=self.meshsizes[1],V=V_boundary,degree=2))
        if type=="started":
            self.w0.interpolate(Expression(( f"-x[1]*(sqrt(x[0]*x[0]+x[1]*x[1])+({r1**2}/sqrt(x[0]*x[0]+x[1]*x[1])))*({r2}*V)/({r1**2}+{r2**2})",f"x[0]*(sqrt(x[0]*x[0]+x[1]*x[1])+({r1**2}/sqrt(x[0]*x[0]+x[1]*x[1])))*({r2}*V)/({r1**2}+{r2**2})", "0")+
            tuple((a for b in [(f"-({b.center[1]})*({b.center[0]**2+b.center[1]**2}+({r1**2}/{b.center[0]**2+b.center[1]**2}))*({r2}*V)/({r1**2}+{r2**2})",f"{b.center[0]}*({b.center[0]**2+b.center[1]**2}+({r1**2}/{b.center[0]**2+b.center[1]**2}))*({r2}*V)/({r1**2}+{r2**2})") for b in self.balls] for a in b ))+(len(self.balls)*tuple(["0"])),h=self.meshsizes[1],V=V_boundary,degree=2))
        self.w.assign(self.w0)
    
def generate_rdist(r1,r2,r,count):
    points=[]
    for i in range(count):
        for j in range(400):
            rho=np.sqrt(random.uniform(r1**2,r2**2))
            theta=random.uniform(0,2*np.pi)
            x=rho*np.cos(theta)
            y=rho*np.sin(theta)
            valid=True
            for p in points:
                if np.sqrt((x-p[0])**2 + (y-p[1])**2) < 2.1*r:
                    valid=False
                    break
            if valid:
                points.append((x,y))
                break
    return points


random.seed(4)


d1 = date.today().strftime("%m%d")

if distr == "unif":
    balls=[Ball((p[0],p[1]), 4e-6,1010)  for p in generate_rdist(r1+5e-6,r2-10e-6,4e-6,int(sys.argv[3]))]
    #balls=[Ball((p[0],p[1]), 4e-6,1050)  for p in generate_rdist(r1+5e-6,r2-5e-6,4e-6,60)]

if distr == "grid":

    balls=[Ball((10e-6+(i*13e-6),10e-6+(j*11e-6)), 4e-6,1050) for i in range(10) for j in range(4)]

problem_name=f"{d1}_circ_shear_fast2_{len(balls)}_{type}_{distr}_{V_boundary}"

problem=ShearTest(problem_name, 
                   balls, 
                   (r1, r2),
                   mesh_perimeters=[8e-6,3e-6], 
                   dt=0.05e-4 if V_boundary <1.1 else ( 0.01e-4 if V_boundary <3.1 else 0.05e-5), #0.03
                   #dt = 0.03e-4 if V_boundary <1.1 else ( 0.02e-4 if V_boundary <2.1 else 0.01e-4),
                   nu=3.896e-3,
                   rho_fluid=1050,
                   starting_density=0.35,
                   density_falloff=30e9,
                   density_cap=(8e-6)**2,
                   remeshing_period=int(round(12-(4*min(V_boundary,1)))),
                   linear_density_falloff=False,
                   linear=False,
                   allow_rotation=True,
                   mesh_density=64,
                   beta_h=10e8, 
                   tolerance=5e-8,
                   circular=True
                   ) 
init=True

while t < t_end:
    t = problem.solve_step()

    problem.print(problem.force_t)
    problem.print(problem.force_b)
    
    with open(f"results/result_{problem_name}/forces.csv","w" if init else "a") as f:
        print(f"{t},{problem.force_t},{problem.force_b}",file=f)
    
    init=False
