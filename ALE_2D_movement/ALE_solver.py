from dolfin import *
import numpy as np
from mesh_generator import Ball, create_mesh
from admesh_ext import admesh_ext as admesh
import os
import sys
import time

parameters["std_out_all_processes"] = False
parameters["form_compiler"]["quadrature_degree"] = 8
parameters["refinement_algorithm"] = "plaza_with_parent_facets"
parameters["ghost_mode"] = "shared_facet"


# admesh parameters for initial mesh
init_remesh_params = {"num_iter":20,
                 "Marge_cells":True,
	         "Add_Vertex_On_Edge":True,
	         "Flip_Edges":True,
                 "Remove_Bndry_Edge":True,
	         "Remove_Vertices":False,
                 "Move_Vertices":True,
	         "Report":True,
              "auto_refine": False,
            "converge_to_one": False
}

# admesh parameters for remeshing
remesh_params = {"num_iter":6,
                 "Marge_cells":True,
	         "Add_Vertex_On_Edge":True,
	         "Flip_Edges":True,
                 "Remove_Bndry_Edge":False,
	         "Remove_Vertices":False,
                 "Move_Vertices":True,
	         "Report":False,
             "auto_refine": False,
            "converge_to_one": False}

def gen_DistFun(balls,squared=True,inverted=False,offset=0.1,scale=1,cap=0):
    """
    Creates a distance function as a FEniCS expression.
    """
    start=""
    if len(balls)==0:
        statement=f"{cap}"
    else:
        statement=balls[0].dist_string()
    for b in balls[1:]:
        start+="fmin("
        statement+=","+b.dist_string()+")"
    statement=start+statement
    
    statement="fmax("+statement+",0)"
    if not squared:
        statement = "sqrt("+statement+")"
    
    if cap>0:
        statement = "fmin("+statement+f",{cap})"
    
    statement = str(offset)+"+("+str(scale)+"*" + statement +")"
    
    if inverted:
        statement = "1/("+statement+")"
   
    return Expression(statement,degree=1)


comm = MPI.comm_world
rank = MPI.rank(comm)

class FluidSystem(object):
    def __init__(self,
                 filename,
                 balls,
                 sizes,
                 t=0.0,
                 dt=0.5,
                 theta=1.0,
                 nu=0.01,
                 rho_fluid=1,
                 rho_ball=2,
                 mesh_perimeters=[0.4, 0.1],
                 g=0,
                 beta_h=1000,
                 useADmesh=True,
                 remeshing_period = 5,
                 starting_density=0.1,
                 density_falloff=0.25,
                 density_cap=20,
                 linear_density_falloff=False,
                 linear=True,
                 allow_rotation=False,
                 print_ball_info=False,
                 tolerance=1e-10,
                 circular=False,
                 *args,
                 **kwargs):
        """Creates FluidSystem object, genrates mesh with gven parameter and formulates the problem.

        Args:
            filename (string): Name used in the saved file.
            balls (List(Ball)): List of ball objects.
            sizes (tuple): Two dimensional tuple containing sizes of the outer box.
            t (float, optional): Starting time. Defaults to 0.0.
            dt (float, optional): Time step. Defaults to 0.5.
            theta (float, optional): Theta parameter of the theta difference scheme. Defaults to 1.0.
            nu (float, optional): Kinematic viscosity. Defaults to 0.01.
            rho_f (float, optional): Density of the fluid. Defaults to 1.0e3.
            rho_ball (float, optional): Density of the balls, density can be specified separately if set to 0. Defaults to 1.0e3.
            mesh_perimeters (list, optional): List of distances around the balls where the mesh will be finer. Defaults to [0.4, 0.1].
            g (float, optional): Gravitational acceleration.
            beta_h (float, optional): Parameter in the Nitsche method. Defaults to 1000.
            useADmesh (bool, optional): If true, uses ADmesh library to generate better mesh and allow remeshing. Defaults to True.
            remeshing_period (int, optional): = Creates new mesh after supplied number of steps. Defaults to 5.
            starting_density (float, optional): Size of cells closest to ball boundary. Defaults to 0.1.
            density_falloff (float, optional): Rate at which the cell size gets larger with distance from closest ball. Defaults to 0.25.
            density_cap (float, optional): Upper boundary to cell size measured in distance from particle. Defaults to 20.
            linear_density_falloff (float, optional): If true, the cell size increases linearly with distance from particle, otherwise quadratically. Defaults to False. 
            linear (bool, optional): If true, Stokes equation is used, otherwise uses Navier-Stokes. Defaults to True.
            allow_rotation (bool, optional): If true, ball rotation is computed. Defaults to False.
            print_ball_info (bool, optional): If true, prints information about balls in each iteration.
            tolerance (float, optional): Absolute and relative tolerance for the Newton solver. Defaults to 1e-10. 
        """


        self.useADmesh=useADmesh
        
        if isinstance(balls, Ball):
            balls = [balls]
        self.balls = balls

        self.leftbottom=kwargs.get("leftbottom",(0,0))
        self.starting_density = starting_density
        self.density_falloff = density_falloff
        self.linear = linear
        self.allow_rotation=allow_rotation
        self.linear_density_falloff = linear_density_falloff
        self.density_cap = density_cap
        self.print_ball_info=print_ball_info
        self.tolerance = tolerance
        self.circular=circular
        self.comm=comm

        try:
            if not len(sizes) == 2:
                self.print("Sizes has to be touple of lenth 2")
                return
        except:
            self.print("Sizes has to be touple of lenth 2")
            return
        self.meshsizes = sizes
        self.perimeters = mesh_perimeters

        self.filename=filename
        self.output_file = "logs/"+filename+"_log.txt"
        
        if not os.path.exists("logs"):
            os.makedirs("logs")
            
        if ("--mesh" in sys.argv):
            self.print("Creating initial mesh")
            self.create_init_mesh(**kwargs)
            exit()
        
        if not os.path.exists("results/results_%s/mesh_init.xdmf" % self.filename):
            self.print("================================")
            self.print("ERROR: Mesh file does not exist.")
            self.print("================================")
            return
        
        if not os.path.exists("results"):
            os.makedirs("results")  
        
        self.print(f"Creating problem with th following parameters:\nball_count= {len(balls)}\nsizes= {sizes}\ndt= {dt}\ntheta= {theta}\n")
        
        # fluid parameters
        self.nu = Constant(nu)
        self.rho_f = Constant(rho_fluid)
        self.rho_fluid=rho_fluid

        # time stepping parameters
        self.t = t
        self.dt = dt
        self.q0 = Constant(1.0 - theta)
        self.q1 = Constant(theta)
        self.k = Constant(1.0 / dt)

        self.ballvel = [0 for i in range(2*len(self.balls))]

        self.create_first_mesh(**kwargs)
        self.create_boundaries_and_spaces()
        self.set_IC()

        # Scaling for ODE:
        self.M = []
        for ball in balls:
            r = ball.radius
            self.M.append((rho_ball if ball.rho==0 else ball.rho) * pi * r**2)

        self.g = g
        self.ey = Constant((0, 1)*len(self.balls))
        
        self.beta_h=beta_h

        self.drag_x = np.zeros(len(self.balls)) 
        self.drag_y = np.zeros(len(self.balls)) 

        self.formulate_problem()
        
        self.remeshing_period = remeshing_period
        self.steps_to_remesh = remeshing_period

        # Create files for storing solution
        self.vfile = XDMFFile("results/results_%s/fluid_v.xdmf" % filename)
        self.mfile = XDMFFile("results/results_%s/fluid_mesh_v.xdmf" % filename)
        self.pfile = XDMFFile("results/results_%s/fluid_p.xdmf" % filename)
        self.print(f"Save location: {'results/results_%s/' % filename}")
        self.vfile.parameters["flush_output"] = True
        self.pfile.parameters["flush_output"] = True
        self.mfile.parameters["flush_output"] = True

        self.save(self.w,self.mesh_v,t)
        self.t+=dt

    def set_IC(self):
        if self.allow_rotation:
            self.w0.interpolate(Expression(("0", "0", "0")+(len(self.balls)*("0", "0"))+(len(self.balls)*tuple(["0"])),degree=2))
            self.w.assign(self.w0)
        else:
            self.w0.interpolate(Expression(("0", "0", "0")+(len(self.balls)*("0", "0")),degree=2))
            self.w.assign(self.w0)

    def create_init_mesh(self,**kwargs):
        mesh, bndry = create_mesh(
            self.meshsizes,
            self.balls,
            refinement_perimeters=self.perimeters,
            unique_outer2D=True,
            mark_bndry=False,
            circular=self.circular,
            **kwargs)

        self.mesh_filename="results/results_%s/mesh_init.xdmf" % self.filename
        #save mesh and boundary function
        with XDMFFile(comm,self.mesh_filename) as xdmf:
            xdmf.write(mesh)   

    def create_first_mesh(self,**kwargs):     
        
        self.mesh_filename="results/results_%s/mesh.xdmf" % self.filename
        mesh = Mesh(comm)
        with XDMFFile(comm,"results/results_%s/mesh_init.xdmf" % self.filename) as xdmf:
            xdmf.read(mesh)   
        with XDMFFile(comm,self.mesh_filename) as xdmf:
            xdmf.write(mesh)   

        if self.useADmesh:
            # using admesh
            self.m = admesh(self.mesh_filename,comm)

            self.m.set_remesh_params(init_remesh_params)

            mesh_distfun = gen_DistFun(self.balls,not self.linear_density_falloff,False,offset=self.starting_density ,scale=self.density_falloff,cap=self.density_cap)
            
            #m.set_bndry(self.bndry)
            self.m.set_remesh_function(mesh_distfun)
            self.r_min = self.m.rmin()
            self.r_max = self.m.rmax()
            self.m.remesh((self.r_min+self.r_max)*0.3,(self.r_min+self.r_max)*0.6)
            self.mesh = self.m.get_mesh()
            self.m.set_remesh_params(remesh_params)


            self.mesh_file = XDMFFile(comm, "results/results_%s/mesh.xdmf" % self.filename)
            self.mesh_file.write(self.mesh)
        else:
            with XDMFFile(comm,self.mesh_filename) as xdmf:
                xdmf.read(self.mesh)
        
        if self.circular:
            self.i_bndry = CompiledSubDomain(f"(x[0]*x[0])+(x[1]*x[1])<{(self.meshsizes[0]+1e-7)**2} && on_boundary")
            self.o_bndry = CompiledSubDomain(f"(x[0]*x[0])+(x[1]*x[1])>{(self.meshsizes[1]-1e-7)**2} && on_boundary")
        else:
            self.l_bndry = CompiledSubDomain(f"near(x[0],{self.leftbottom[0]},1e-9) && on_boundary")
            self.r_bndry = CompiledSubDomain(f"near(x[0],{self.leftbottom[0]+self.meshsizes[0]},1e-9) && on_boundary")
            self.b_bndry = CompiledSubDomain(f"near(x[1],{self.leftbottom[1]},1e-9) && on_boundary")
            self.t_bndry = CompiledSubDomain(f"near(x[1],{self.leftbottom[0]+self.meshsizes[1]},1e-9) && on_boundary")

    def create_boundaries_and_spaces(self,w0_new=None):
        self.print("Marking boundaries...")
        start=time.time()
        #self.ball_bndries=[CompiledSubDomain(ball.dist_string()+"<0.01*r*r && on_boundary",r=ball.get_radius()) for ball in self.balls]
        self.bndry=MeshFunction('size_t', self.mesh, self.mesh.topology().dim() - 1)  
        for i in range(len(self.balls)):
            self.balls[i].mark(self.bndry,i+1)
        
        if self.circular:
            self.i_bndry.mark(self.bndry,len(self.balls)+1)
            self.o_bndry.mark(self.bndry,len(self.balls)+2)
        else:
            self.l_bndry.mark(self.bndry,len(self.balls)+1)
            self.r_bndry.mark(self.bndry,len(self.balls)+2)
            self.b_bndry.mark(self.bndry,len(self.balls)+3)
            self.t_bndry.mark(self.bndry,len(self.balls)+4)
            
        end = time.time()
        self.print(f"Projection marking time: {end - start}")

        with XDMFFile(comm,"results/results_%s/boundary.xdmf" % self.filename) as xdmf:
            xdmf.write(self.bndry)
        
        # Taylor-Hood spaces
        self.print("Creating spaces...")
        VP = VectorElement("CG", self.mesh.ufl_cell(), 2)
        VP1 = VectorElement("CG", self.mesh.ufl_cell(), 1)
        P1 = FiniteElement("CG", self.mesh.ufl_cell(), 1)
        R  = VectorElement("R", self.mesh.ufl_cell(), 0,dim = 2*len(self.balls) )  # space for boundary velocity
        if self.allow_rotation:
            Rt  = VectorElement("R", self.mesh.ufl_cell(), 0,dim = len(self.balls) )
        self.MV = FunctionSpace(self.mesh, VP1)  #space for mesh velocity
        self.V =  FunctionSpace(self.mesh, VP)
        self.VR =  FunctionSpace(self.mesh, R)
        if self.allow_rotation:
            self.VRt =  FunctionSpace(self.mesh, Rt)
            self.W = FunctionSpace(self.mesh, MixedElement([VP, P1, R, Rt]))
        else:
            self.W = FunctionSpace(self.mesh, MixedElement([VP, P1, R]))

        self.print(f"Created new mesh and spaces, DOF count: {self.W.dim()}")

        # Facet normal, identity tensor and boundary measure
        #n = FacetNormal(mesh)
        #I = Identity(mesh.geometry().dim())
        self.print("Initialising measures...")
        self.ds = Measure("ds",self.mesh, subdomain_data=self.bndry)

        self.dx = Measure("dx", self.mesh)
        
        self.n=FacetNormal(self.mesh)

        self.print("Initialising fluid functions...")
        if self.allow_rotation:
            (self.v_, self.p_, self.vp_, self.omg_) = TestFunctions(self.W)
        else:
            (self.v_, self.p_, self.vp_) = TestFunctions(self.W)

        # current unknown at time step t
        self.w = Function(self.W)

        # previous known time step solution
        if w0_new == None:
            self.w0 = Function(self.W)
        else:
            self.w0 = project(w0_new,self.W)


        if self.allow_rotation:
            (self.v, self.p, self.vp, self.omg) = split(self.w)
            (self.v0, self.p0, self.vp0, self.omg0) = split(self.w0)
        else:
            (self.v, self.p, self.vp) = split(self.w)
            (self.v0, self.p0, self.vp0) = split(self.w0)

        # Mesh velocity
        self.print("Initialising mesh functions...")
        #self.mesh_u = Function(self.MV)
        self.mesh_v = Function(self.MV)
        self.dh = Function(self.MV)
        self.u1 = TrialFunction(self.MV)
        self.u_ = TestFunction(self.MV)
    
    

    def F_fluid(self):
        def a(v, v_):
            D = sym(grad(v))
            return (inner(
                2.0 * self.nu * D,
                grad(v_))) * self.dx  + (0 if self.linear else self.rho_f * inner(grad(v) *(v -self.mesh_v ), v_)*self.dx)

        def b(q, v):
            return inner(div(v), q) * self.dx

        # Define variational forms without time derivative in previous time
        F0_eq1 = a(self.v0, self.v_) - b(self.p, self.v_)
        F0_eq2 = b(self.p_, self.v)
        F0 = F0_eq1 + F0_eq2

        # variational form without time derivative in current time
        F1_eq1 = a(self.v, self.v_) - b(self.p, self.v_)
        F1_eq2 = b(self.p_, self.v)
        F1 = F1_eq1 + F1_eq2

        #combine variational forms with time derivative
        #
        #  dw/dt + F(t) = 0 is approximated as
        #  (w-w0)/dt + (1-theta)*F(t0) + theta*F(t) = 0
        #
        ey=Constant((0,1))
        
        F = self.k * self.rho_f * inner(
            (self.v - self.v0),
            self.v_) * self.dx + self.q0 * F0 + self.q1 * F1 +self.rho_f* inner(self.g*ey,self.v_)*self.dx# + rho*inner(f,v_)*dx

        return F

    def get_bcs(self):
        ball_count = len(self.balls)

        profile=Expression(("4.0*um*x[1]*(H-x[1])/(H*H)","0"),H=0.41,um=0.3,degree=2)
        bc_l = DirichletBC(self.W.sub(0), Constant((0, 0)), self.bndry,
                           ball_count + 1)
        bc_r = DirichletBC(self.W.sub(0),Constant((0, 0)), self.bndry,
                           ball_count + 2)
        bc_b = DirichletBC(self.W.sub(0), Constant((0, 0)), self.bndry,
                          ball_count + 3)
        bc_t = DirichletBC(self.W.sub(0), Constant((0.0, 0.0)), self.bndry,
                           ball_count + 4)
        bc_p_leftbot = DirichletBC(self.W.sub(1), 0.0, "near(x[0],0.0) && near(x[1],0.0)", method="pointwise")
        fluid_bcs = [bc_l,bc_r,bc_b,bc_t,bc_p_leftbot] 

        bc_balls = []
        
        for i,ball in enumerate(self.balls):
            bc_balls.append( DirichletBC(self.MV.sub(0), self.w.sub(2).sub(2*i), self.bndry, i+1) )
            bc_balls.append( DirichletBC(self.MV.sub(1), self.w.sub(2).sub(2*i+1), self.bndry, i+1) )
             
        bc_l = DirichletBC(self.MV, Constant((0, 0)), self.bndry,
                           ball_count + 1)
        bc_r = DirichletBC(self.MV, Constant((0, 0)), self.bndry,
                           ball_count + 2)
        bc_b = DirichletBC(self.MV, Constant((0, 0)), self.bndry,
                           ball_count + 3)
        bc_t = DirichletBC(self.MV, Constant((0, 0)), self.bndry,
                           ball_count + 4)
        mesh_bcs = [bc_l, bc_r, bc_b, bc_t]+bc_balls

        return fluid_bcs, mesh_bcs

    def formulate_problem(self):
        self.print("Formulating problem...")
        F = self.F_fluid()
        
        # Cauchy tensor
        def sigma(q,v):
            # Identity a and velocity gradient:
            I = Identity(self.mesh.geometry().dim())
            D = sym(grad(v))

            return -q*I + 2*self.nu*D

        self.vol_f = Constant(assemble(Constant(1.0)*self.dx))
        self.M_vector=[]
        for i in range(len(self.balls)):
            self.M_vector += [self.M[i],self.M[i]]

        #linear momentum of ball
        F +=(1/self.vol_f)* sum([
            self.M_vector[i] * ((self.vp[i] - self.vp0[i]) / self.dt - self.g *
                         (-self.ey[i]))* self.vp_[i] for i in range(2*len(self.balls))])* self.dx
        #angular mometnum of ball
        if self.allow_rotation:
            self.I_vector = []
            for i in range(len(self.balls)):
                self.I_vector += [ 0.5 * self.M[i] * (self.balls[i].radius**2) ]
            
            F += (1/self.vol_f)* sum([self.I_vector[i] * ((self.omg[i] - self.omg0[i]) / self.dt) * self.omg_[i] for i in range(len(self.balls))] )* self.dx

        self.R_vector = []
        if self.allow_rotation:
            self.R_vector += [ 0.5 * self.M[i] * (self.balls[i].radius**2) ]
        
        proj=[ np.zeros((2,2*len(self.balls))) for i in range(len(self.balls))]
        
        if self.allow_rotation:
            self.rot_vec = [Expression(("c_y-x[1]","x[0]-c_x"),c_x=b.center[0],c_y=b.center[1],degree=1) for b in self.balls ]
            
       
 
        for i in range(len(self.balls)):           
            
            # Nitshcem si vynutim okrajovky
            # Nitsche method boundary condition:
            if self.allow_rotation:
                # velocity on i_th boundary (proj[i]* self.vp + ( omg_proj*self.omg* (x-centers[i]) ) ) 
                F += -(inner(dot(sigma(self.p,self.v),self.n),self.v_-(as_vector((self.vp_[2*i],self.vp_[2*i+1])) + ( self.omg_[i] *self.rot_vec[i]) ) ) + \
                inner(dot(sigma(self.p_,self.v_),self.n),self.v- (as_vector((self.vp[2*i],self.vp[2*i+1])) + ( self.omg[i] * self.rot_vec[i] ) ) ) - \
                (2*self.beta_h)*inner(self.v- (as_vector((self.vp[2*i],self.vp[2*i+1])) + ( self.omg[i] * self.rot_vec[i])  ) , self.v_))*self.ds(i+1)
            else:   
                proj[i][0,2*i]=1
                proj[i][1,2*i+1]=1
                proj[i] = as_matrix(proj[i])

                F += -(inner(dot(sigma(self.p,self.v),self.n),self.v_-(proj[i]*self.vp_)) + \
                inner(dot(sigma(self.p_,self.v_),self.n),self.v- (proj[i]*self.vp)) - \
                (2*self.beta_h)*inner(self.v- ((proj[i]*self.vp)), self.v_))*self.ds(i+1) # axi-cyl corr


        DF = derivative(F, self.w)

        self.fluid_bcs, self.mesh_bcs = self.get_bcs()

        self.print("Formulating fluid problem...")
        start = time.time()
        fluid_problem = NonlinearVariationalProblem(F, self.w, self.fluid_bcs, DF)
        self.fluid_solver = NonlinearVariationalSolver(fluid_problem)
        end = time.time()
        self.print(f"Compilation time: {end - start}")
        
        prm = self.fluid_solver.parameters
        #info(prm,True)  #get full info on the parameters
        prm['nonlinear_solver'] = 'newton'
        prm['newton_solver']['absolute_tolerance'] = self.tolerance
        prm['newton_solver']['relative_tolerance'] = self.tolerance
        prm['newton_solver']['maximum_iterations'] = 20
        prm['newton_solver']['linear_solver'] = 'mumps'

        self.dist_fun=gen_DistFun(self.balls,False,True,offset=0.5)
        
        MF = inner(self.dist_fun*grad(self.u1), grad(self.u_)) * self.dx

        print("Formulating mesh problem...")
        self.mesh_problem = LinearVariationalProblem(lhs(MF), rhs(MF), self.mesh_v,
                                                self.mesh_bcs)

    def solve_step(self):
        info("t = {}".format(self.t))
        
        if self.steps_to_remesh<1:
            self.remesh()
            self.steps_to_remesh = self.remeshing_period

        self.fluid_solver.solve()
        if self.allow_rotation:
            (self.v, self.p, self.vp, self.omg) = self.w.split(True)
        else:
            (self.v, self.p, self.vp) = self.w.split(True)
        
        self.vol_f=assemble(1.0*self.dx)

        self.move()


        # Update values at the end of timestep
        self.end_step_callback()
        self.w0.assign(self.w)
        self.steps_to_remesh -= 1
        self.t += self.dt
        self.save(self.w, self.mesh_v, self.t)
        return self.t
    
    

    def save(self, w, mesh_u, t):
        # Extract solutions:
        if self.allow_rotation:
            (v, p, vp, omg) = w.split()
        else:
            (v, p, vp) = w.split()
        v.rename("v", "velocity")
        mesh_u.rename("mesh_u", "mesh_velocity")
        p.rename("p", "pressure")
        # Save to file
        self.vfile.write(v, t)
        self.mfile.write(mesh_u, t)
        self.pfile.write(p, t)

    def move(self):
        start=time.time()
        self.mesh_solver = LinearVariationalSolver(self.mesh_problem)
        #self.mesh_solver.parameters["linear_solver"] = 'cg'
        #self.mesh_solver.parameters["preconditioner"] = 'ilu'
        self.mesh_solver.solve()      
        end = time.time()
        self.print(f"Mesh problem time: {end - start}s")

        start=time.time()
        self.dh.vector()[:] = self.mesh_v.vector()*self.dt
        self.update_particles()  
        self.before_move_callback()
        self.m.move(self.dh)
        end = time.time()
        self.print(f"Moving mesh and particles time: {end - start}s")
        
        

    def update_particles(self):
        I = Identity(self.mesh.geometry().dim())
        D = sym(grad(self.v))
        self.ballvel=np.zeros(2*len(self.balls))
        
        for i in range(len(self.balls)):
            v_x=assemble((self.vp[2*i]*self.dx))/self.vol_f
            v_y=assemble((self.vp[2*i+1]*self.dx))/self.vol_f
            if self.allow_rotation:
                omg=assemble((self.omg[i]*self.dx))/self.vol_f
            else:
                omg=0
            
            # spocitej drag:
            self.drag_x[i] = -assemble(dot((-self.p*I + 2*self.nu*D),self.n)[0]*self.ds(i+1))
            self.drag_y[i] = -assemble(dot((-self.p*I + 2*self.nu*D),self.n)[1]*self.ds(i+1))           
            
            self.balls[i].update(self.dt,v_x,v_y,omg)
            
            if self.allow_rotation:
                self.rot_vec[i].c_x = self.balls[i].center[0]
                self.rot_vec[i].c_y = self.balls[i].center[1]

            circ=self.balls[i].radius * 2 *np.pi
            
            x0 = Expression("x[0]",degree=1)
            x1 = Expression("x[1]",degree=1)

            self.ballvel[2*i] = v_x
            self.ballvel[2*i+1] = v_y
            
            if self.print_ball_info:
                self.print(f"Ball {i} velocity: vp=({v_x},{v_y})")
                self.print(f"Ball {i} drag: F_d=({self.drag_x[i]},{self.drag_y[i]})")
                self.print(f"Ball {i} move: mesh_v=({v_x*self.dt},{v_y*self.dt})")
                self.print(f"Mesh {i} move: mesh_v=({assemble((self.dh[0]*self.ds(i+1)))/circ},{assemble((self.dh[1]*self.ds(i+1)))/circ})")
                self.print(f"Ball {i} new center: ({self.balls[i].center[0]},{self.balls[i].center[1]})")
                self.print(f"Ball {i} mesh center: ({assemble((x0*self.ds(i+1)))/circ},{assemble((x1*self.ds(i+1)))/circ})")
        
        #self.ball_bndries=[CompiledSubDomain(ball.dist_string()+"<0.01*r*r && on_boundary",r=ball.get_radius()) for ball in self.balls]

    def remesh(self):
        self.print("Remeshing...")
        
        with XDMFFile(comm, "results/results_%s/before_remeshing_mesh.xdmf" % self.filename) as f:
            f.write(self.mesh) 

        mesh_distfun = gen_DistFun(self.balls,not self.linear_density_falloff,False,offset=self.starting_density ,scale=self.density_falloff,cap=self.density_cap)
        
        self.m.renew_mesh()
        self.m.set_remesh_function(mesh_distfun)

        w0_val = self.w0.vector().get_local()
        
        w0_new = Function(self.W)
        w0_new.vector().set_local(w0_val)
        w0_new.vector().apply('insert')
        w0_new.assign(self.w0)
        
        if not self.linear:
            mesh_v_val = self.mesh_v.vector().get_local()
        
            mesh_v_new = Function(self.MV)
            mesh_v_new.assign(self.mesh_v)

            mesh_v_new.set_allow_extrapolation(True)
        
        self.m.remesh((self.r_min+self.r_max)*0.3,(self.r_min+self.r_max)*0.6)
        self.print("New mesh generated")
        self.mesh = self.m.get_mesh()
        
        with XDMFFile(comm, "results/results_%s/remeshed_mesh.xdmf" % self.filename) as f:
            f.write(self.mesh) 
        
        w0_new.set_allow_extrapolation(True)
        
        if self.allow_rotation:
            (v,p,vp,omg) = w0_new.split(True)
        else:
            (v,p,vp) = w0_new.split(True)
        v.set_allow_extrapolation(True)

        self.create_boundaries_and_spaces() 

        self.print("Projecting solution on the new space...")
        start=time.time()
        assign(self.w0.sub(0), project(v,self.V,solver_type="cg", preconditioner_type="hypre_amg") )
        v_new = Function(self.VR)
        v_new.vector().set_local(vp.vector().get_local())
        v_new.vector().apply('insert')
        assign(self.w0.sub(2),v_new)
        print(self.w0.sub(2).vector()[-10:])
        if self.allow_rotation:
            v_new = Function(self.VRt)
            v_new.vector().set_local(omg.vector().get_local())
            v_new.vector().apply('insert')
            assign(self.w0.sub(3),v_new)

        end = time.time()
        self.print(f"Projection time: {end - start}")
        
        self.print("Projecting mesh velocity on the new space...")
        start=time.time()
        if not self.linear:
            self.mesh_v.assign(project(mesh_v_new,self.MV,solver_type="cg", preconditioner_type="hypre_amg"))
        end = time.time()
        self.print(f"Projection time: {end - start}")
        
        self.formulate_problem()
        
        with XDMFFile(comm, "results/results_%s/projected_w0.xdmf" % self.filename) as f:
            self.w0.rename("v", "velocity")
            f.write(self.w0,self.t)  

    def print(self,message):
        if rank == 0:
            info(message)
            if self.output_file != None:
                with open(self.output_file,"a") as f:
                    print(message,file=f)

    def before_move_callback(self):
        return

    def end_step_callback(self):
        return


if __name__ == "__main__":
    problem=FluidSystem("ALE_fall", 
                  [Ball((2,3), 0.7,0.1),Ball((3,7), 0.4,5)],
                  (5, 10),
                  t=0,
                  mesh_perimeters=[0.5,0.2,0.1], 
                  dt=0.05,
                  nu=1e-1,
                  rho_fluid=1,
                  g=9.8,
                  useADmesh=True,
                  starting_density=0.2,
                  density_falloff=1,
                  remeshing_period=2,
                  density_cap=1,
                  linear=False,
                  allow_rotation=True, 
                  )
    
    t=0
    t_end=1.5
    
    while t < t_end:
    
        t = problem.solve_step()
