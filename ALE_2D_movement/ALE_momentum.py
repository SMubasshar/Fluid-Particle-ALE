from ast import Expression
from dolfin import *
from ALE_solver import FluidSystem
from utils.mesh_generator import Ball
import numpy as np
from datetime import date
import sys 

class Kulecnik(FluidSystem):
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        
    def set_IC(self):
        self.w0.interpolate(Expression(("0", "0", "0")+("5","0","0","0","0","0"),degree=2))
        self.w.assign(self.w0)
    
    def get_bcs(self):
        ball_count = len(self.balls)

        bc_l = DirichletBC(self.W.sub(0).sub(0), Constant(0.0), self.bndry,
                           ball_count + 1)
        bc_r = DirichletBC(self.W.sub(0).sub(0), Constant(0.0), self.bndry,
                           ball_count + 2)
        bc_b = DirichletBC(self.W.sub(0).sub(1), Constant(0.0), self.bndry,
                          ball_count + 3)
        bc_t = DirichletBC(self.W.sub(0).sub(1), Constant(0.0), self.bndry,
                           ball_count + 4)
        bc_p_leftbot = DirichletBC(self.W.sub(1), 0.0, "near(x[0],0.0) && near(x[1],0.0)", method="pointwise")
        fluid_bcs=[]

        bc_balls = []
        

        for i,ball in enumerate(self.balls):
            bc_balls.append( DirichletBC(self.MV.sub(0), self.w.sub(2).sub(2*i), self.bndry, i+1) )
            bc_balls.append( DirichletBC(self.MV.sub(1), self.w.sub(2).sub(2*i+1), self.bndry, i+1) )
             
        mbc_l = DirichletBC(self.MV, Constant((0, 0)), self.bndry,
                           ball_count + 1)
        mbc_r = DirichletBC(self.MV, Constant((0, 0)), self.bndry,
                           ball_count + 2)
        mbc_b = DirichletBC(self.MV, Constant((0, 0)), self.bndry,
                           ball_count + 3)
        mbc_t = DirichletBC(self.MV, Constant((0, 0)), self.bndry,
                           ball_count + 4)
        mesh_bcs = [mbc_l, mbc_r, mbc_b, mbc_t]+bc_balls

        return fluid_bcs, mesh_bcs
    
    def before_move_callback(self):
        ball_count=len(self.balls)
    
        self.fluid_x = assemble(self.rho_f*self.v[0]*self.dx)
        self.fluid_y = assemble(self.rho_f*self.v[1]*self.dx)

        self.balls_x=[self.ballvel[2*i]*self.M[i] for i in range(len(self.balls))]
        self.balls_y=[self.ballvel[2*i+1]*self.M[i] for i in range(len(self.balls))]

        self.inflow_x=np.sum([assemble(self.rho_f*dot(self.v,self.n)*self.v[0]*self.ds(ball_count + 1 + i)) for i in range(4)])
        self.inflow_y=np.sum([assemble(self.rho_f*dot(self.v,self.n)*self.v[1]*self.ds(ball_count + 1 + i)) for i in range(4)])

        self.print(f"Total momentum: ( {self.fluid_x+sum(self.balls_x)}, {self.fluid_y+sum(self.balls_y)})")
        self.omg_val=[assemble(self.omg[i]*self.dx)/self.vol_f for i in range(len(self.balls))]

        self.print(f"rho_f={assemble(self.rho_f*self.dx)/assemble(Constant(1)*self.dx)}, m_1={self.M[0]}, m_2={self.M[1]}, v_1x={self.ballvel[0]}, v_2x={self.ballvel[2]}")

        x=Expression(("x[0]","x[1]"),degree=2)
        self.ang_origin_f = assemble(self.rho_f*(x[0]*self.v[1]-x[1]*self.v[0])*self.dx)
        self.ang_origin_b=sum([self.M[i]*(self.omg_val[i]*0.5*(self.balls[i].radius**2) + (self.balls[i].center[0]*self.ballvel[2*i+1]-self.balls[i].center[1]*self.ballvel[2*i])) for i in range(len(self.balls))])
        self.ang_origin_i=np.sum([assemble(self.rho_f*dot(self.v,self.n)*(x[0]*self.v[1]-x[1]*self.v[0])*self.ds(ball_count + 1 + i)) for i in range(4)])
        
        #self.print(f"Ang origin: {self.ang_origin}")
        r=Expression(("x[0]-5","x[1]-5"),degree=2)
        self.ang_center_f = assemble(self.rho_f*(r[0]*self.v[1]-r[1]*self.v[0])*self.dx)
        self.ang_center_b = sum([self.M[i]*(self.omg_val[i]*0.5*(self.balls[i].radius**2) + ((self.balls[i].center[0]-5-(self.dt*self.ballvel[2*i]))*self.ballvel[2*i+1]-(self.balls[i].center [1]-5-(self.dt*self.ballvel[2*i+1]))*self.ballvel[2*i])) for i in range(len(self.balls))])
        self.ang_center_i=np.sum([assemble(self.rho_f*dot(self.v,self.n)*(r[0]*self.v[1]-r[1]*self.v[0])*self.ds(ball_count + 1 + i)) for i in range(4)])
        #self.print(f"Ang center: {self.ang_center}")
        return
        

t_end=5

start=(5,4)

dt=float(sys.argv[1])
density=float(sys.argv[2])

d1 = date.today().strftime("%m%d")

name=f"{d1}_ALE_d{density}_2balls_5_0_dt{dt}"

problem=Kulecnik(name, 
                  [Ball((3,5.35), 0.4,1),Ball((4.5,4.65), 0.4,1)],#[Ball((3,6), 0.4,1),Ball((7,4), 0.4,1)],,#
                  (10, 10),
                  t=0,
                  mesh_perimeters=[0.5,0.2,0.1], 
                  dt=dt,
                  nu=1e-1,
                  rho_fluid=1,
                  g=0,
                  useADmesh=True,
                  starting_density=0.14,
                  density_falloff=0.3,
                  remeshing_period=5,
                  density_cap=density,
                  linear=False,
                  allow_rotation=True, 
                  )

t=0

init=True

while t < t_end:
    
    t = problem.solve_step()
    
    with open(f"ALE_2D_movement/benchmark_data/{name}_momentum.csv","w" if init else "a") as f:
        print(f"{t-dt},{problem.fluid_x},{problem.fluid_y},{problem.inflow_x},{problem.inflow_y},{problem.balls_x[0]},{problem.balls_x[1]},{problem.balls_y[0]},{problem.balls_y[1]},{problem.ang_origin_f},{problem.ang_origin_b},{problem.ang_origin_i},{problem.ang_center_f},{problem.ang_center_b},{problem.ang_center_i}",file=f)

    init=False