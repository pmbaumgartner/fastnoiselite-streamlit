from typing import List, Optional

import numpy as np

cimport numpy as np
from libc.math cimport sin, cos, floor

cdef class FlowGrid:
    cdef public float width, height, resolution_factor
    cdef int n_rows, n_columns
    cdef np.float_t[:,:] grid
    cdef public (int, int) shape 
    
    def __init__(self, float width, float height, float resolution_factor = 1.0):
        """Create a flow field initialized with random angles.

        Args:
            width (float): width of the flow field
            height (float): height of the flow field
            resolution_factor (float, optional): Scaling factor. <1 means lower resolution (0.5 is half as the pixels). Defaults to 1.
        """
        self.width = width
        self.height = height
        self.resolution_factor = resolution_factor
        self.n_rows = int(height * self.resolution_factor)
        self.n_columns = int(width * self.resolution_factor)
        self.grid = np.random.random((self.n_rows, self.n_columns)) * 2 * np.pi
        self.shape = (<object> self.grid).shape
        assert (self.n_rows, self.n_columns) == self.shape

    cpdef initialize_angles(self, np.float64_t[:,:] angle_array):
        if not (<object> self.grid).shape == (<object> angle_array).shape:
            passed_shape = (<object> angle_array).shape
            raise ValueError(f"Mismatched Grid Shapes. Grid: {self.shape}, Passed: {passed_shape}")
        for i in range(self.n_rows):
            for j in range(self.n_columns):
                self.grid[i, j] = angle_array[i,j]

    cpdef float locate(self, float x, float y):
        cdef float locx, locy
        cdef int locx_ix, locy_ix
        locx, locy = (
            x * self.resolution_factor,
            y * self.resolution_factor,
        )
        locx_ix, locy_ix = (<int>locx), (<int>locy)
        try:
            # remember: y = height = rows
            angle = self.grid[locy_ix, locx_ix]  
            return angle
        except IndexError:
            print(f"x={x}, y={y}, xloc={locx}, yloc={locy}, locx_ix={locx_ix}, locy_ix={locy_ix}, {self.shape})")

    cpdef bint gridcheck(self, float x, float y):
        if x >= self.width:
            return False
        elif x < 0:
            return False
        elif y >= self.height:
            return False
        elif y < 0:
            return False
        return True

cdef class Particle:
    cdef float x, y
    cdef int run_steps, steps
    cdef bint inbounds
    cdef FlowGrid flowgrid
    cdef np.float64_t[:,:] history 


    def __init__(self, float x, float y, FlowGrid flowgrid, int run_steps):
        self.x = x
        self.y = y
        self.flowgrid = flowgrid
        self.run_steps = run_steps
        self.steps = 0
        self.history = np.zeros((2, run_steps+1), dtype=np.float64)
        self.inbounds = self.flowgrid.gridcheck(x, y)
        self.history[0, self.steps] = self.x
        self.history[1, self.steps] = self.y


    cpdef void move(self, float step_length):
        cdef float angle, x_step, y_step, xvec, yvec
        if self.inbounds:
            angle = self.flowgrid.locate(self.x, self.y)
            xvec = cos(angle)
            yvec = sin(angle)
            x_step = step_length * xvec
            y_step = step_length * yvec
            self.x += x_step
            self.y += y_step
            self.inbounds = self.flowgrid.gridcheck(self.x, self.y)
            if self.inbounds:
                self.steps += 1
                try:
                    self.history[0, self.steps] = self.x
                    self.history[1, self.steps] = self.y
                except IndexError:
                    print(f"Steps: {self.steps}, ({self.x}, {self.y})")

    def get_history(self):
        return np.asarray(self.history[:,:self.steps]).T.tolist()


def run_flowfield(
    grid: FlowGrid,
    steps: int,
    particles: List[Particle],
    step_length: Optional[float] = None,
) -> List[List[float]]:
    if not step_length:
        step_length = 0.001 * min(grid.width, grid.height)
    for _ in range(steps):
        for p in particles:
            p.move(step_length)
    return [p.get_history() for p in particles]
