from admesh4py.admesh4py import admesh
import admesh as msr
from dolfin import *

class admesh_ext(admesh):
    def __init__(self,file_mesh,comm):
        super().__init__(file_mesh,comm)

    def renew_mesh(self):
        self.coor = self.mesh_global.coordinates()
        self.cells = self.mesh_global.cells()
        self.U=FunctionSpace(self.mesh_global,"CG",1) 
        if self.rank ==0:
            self.v2d=vertex_to_dof_map(self.U)
            self.d2v=dof_to_vertex_map(self.U)
        return


    def set_remesh_function_v2(self,f,proj = None):
        if proj ==None:
            f_proj = self.project(f,self.U)
        else:
            f_proj = proj(f,self.comm,self.U)
        #try:
        #    f_proj = proj(f,self.U)
        #except ValueError:
        #    print('Something wrong with function!!')
        if self.rank == 0:
            print(self.v2d)
            f_vec = f_proj.vector()[:]
            print(len(f_vec))
            f_vec = f_vec[self.v2d]
            self.MSR.set_remesh_params(f_vec)

# if self.rank ==0:
#           self.coor = self.MSR.getCoordinates()
#           self.cells = self.MSR.getCells()
#           self.U=FunctionSpace(self.mesh_global,"CG",1) 
#
#           self.v2d=vertex_to_dof_map(self.U)
#           self.d2v=dof_to_vertex_map(self.U)