import math
from xml.etree.ElementTree import tostring

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon
import matplotlib.tri as tri


class Node:
    def __init__(self, id, x, y, constraint, fx = 0, fy = 0, mz = 0):

        self.id = id
        self.x = x
        self.y = y
        self.fx = fx
        self.fy = fy
        self.mz = mz
        self.constraint = constraint

    def has_beam_connection(self, structure):
        for element in structure.elements.values():
            if isinstance(element, Beam) and self in element.get_nodes():
                return True

        return False


class Beam:
    def __init__(self, id, node_i, node_j, E, A, I, c, w = 0):
        self.id = id
        self.c = c
        self.node_i = node_i

        self.node_j = node_j
        self.A = A
        self.E = E
        self.I = I
        self.w = w
        self.length = math.sqrt((node_j.x - node_i.x)**2 + (node_j.y - node_i.y)**2)

        self.V_i = 0
        self.M_i = 0
        self.N_j = 0

        self.angle = math.atan2(node_j.y - node_i.y, node_j.x - node_i.x)
        c = math.cos(self.angle)
        s = math.sin(self.angle)
        l = self.length
        axial = (self.A * self.E / l) * np.array([[1, 0, 0, -1, 0, 0],
                                                  [0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0],
                                                  [-1, 0, 0, 1, 0, 0],
                                                  [0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0]])
        bending = (self.E * self.I) / (l**3) *np.array([[0, 0, 0, 0, 0, 0],
                                                        [0, 12, 6*l, 0, -12, 6 * l],
                                                        [0, 6*l, 4 * l ** 2, 0, -6*l, 2*l**2],
                                                        [0, 0, 0, 0, 0, 0],
                                                        [0, -12, -6 * l, 0, 12, -6 * l],
                                                        [0, 6 * l, 2 * l ** 2, 0, -6 * l, 4 * l ** 2]])
        local = axial+bending
        T = np.array([[c, s, 0, 0, 0, 0],
                      [-s, c, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, c, s, 0],
                      [0, 0, 0, -s, c, 0],
                      [0, 0, 0, 0, 0, 1]])
        self.matrix = T.T @ local @ T

    def calc_stresses(self, uix, uiy, thetai, ujx, ujy, thetaj):
        c = math.cos(self.angle)
        s = math.sin(self.angle)

        global_displacements = np.array([[uix], [uiy], [thetai],
                                         [ujx], [ujy], [thetaj]])

        T = np.array([[c, s, 0, 0, 0, 0],
                      [-s, c, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, c, s, 0],
                      [0, 0, 0, -s, c, 0],
                      [0, 0, 0, 0, 0, 1]])

        local_displacements = T @ global_displacements

        l = self.length
        axial = (self.A * self.E / l) * np.array([[1, 0, 0, -1, 0, 0],
                                                  [0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0],
                                                  [-1, 0, 0, 1, 0, 0],
                                                  [0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0]])
        bending = (self.E * self.I) / (l ** 3) * np.array([[0, 0, 0, 0, 0, 0],
                                                           [0, 12, 6 * l, 0, -12, 6 * l],
                                                           [0, 6 * l, 4 * l ** 2, 0, -6 * l, 2 * l ** 2],
                                                           [0, 0, 0, 0, 0, 0],
                                                           [0, -12, -6 * l, 0, 12, -6 * l],
                                                           [0, 6 * l, 2 * l ** 2, 0, -6 * l, 4 * l ** 2]])
        k_local = axial + bending



        local_forces = k_local @ local_displacements + self.calc_end_forces_local()

        # local_forces = [N_i, V_i, M_i, N_j, V_j, M_j]

        axial_stress = local_forces[3] / self.A
        bending_stress_i = local_forces[2] * self.c/self.I
        bending_stress_j = local_forces[5] * self.c/self.I
        self.V_i = local_forces[1].item()
        self.M_i = local_forces[2].item()
        self.N_j = local_forces[3].item()
        return [axial_stress.item(), bending_stress_i.item(), bending_stress_j.item()]

    def worst_case_stress(self, axial_stress, bending_stress_i, bending_stress_j):
        candidates = [axial_stress + bending_stress_i, axial_stress - bending_stress_i,
                          axial_stress + bending_stress_j, axial_stress - bending_stress_j]
        return max(candidates, key=abs)

    def calc_end_forces_local(self):
        w = self.w
        l = self.length
        return np.array([[0],[w*l/2], [w*l**2/12], [0], [w*l/2], [-1*w*l**2/12]])

    def calc_equivalent_nodal_loads(self):
        s = math.sin(self.angle)
        c = math.cos(self.angle)
        f_fe_local = self.calc_end_forces_local()

        T = np.array([[c, s, 0, 0, 0, 0],
                      [-s, c, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, c, s, 0],
                      [0, 0, 0, -s, c, 0],
                      [0, 0, 0, 0, 0, 1]])


        return T.T @ f_fe_local

    def shear_func(self, x):
        return self.V_i - self.w * x

    def assemble_segments(self, xs, point_i, point_j):
        t = xs / self.length
        X = point_i[0] + t * (point_j[0] - point_i[0])
        Y = point_i[1] + t * (point_j[1] - point_i[1])

        points = np.array([X, Y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        return segments

    def bending_stress_func(self, n_points = 15):
        xs = np.linspace(0, self.length, n_points)

        stresses = []
        for x in xs:
            V = self.V_i - self.w*(x)
            M = -self.M_i + self.V_i * (x) - self.w*(x**2)/2
            axial_stress = self.N_j /self.A
            bending_stress = M * self.c/self.I
            combined = max([axial_stress + bending_stress, axial_stress - bending_stress], key=abs)
            stresses.append(combined)

        return xs, stresses

    def get_nodes(self):
        return [self.node_i, self.node_j]

    def get_stress_info(self, displacements):
        di = displacements[self.node_i.id]
        dj = displacements[self.node_j.id]
        return self.calc_stresses(di[0],di[1], di[2], dj[0], dj[1], dj[2])

class Truss:
    def __init__(self, id, node_i, node_j, E, A, w = 0):
        self.id = id
        self.node_i = node_i
        self.node_j = node_j
        self.A = A
        self.E = E
        self.w = w
        self.axial = 0
        self.length = math.sqrt((node_j.x - node_i.x)**2 + (node_j.y - node_i.y)**2)
        self.angle = math.atan2(node_j.y - node_i.y, node_j.x - node_i.x)
        c = math.cos(self.angle)
        s = math.sin(self.angle)
        self.matrix = (self.A * self.E / self.length) * np.array([[c**2, c*s, 0, -1*c**2, -1*c*s, 0],
                                                                  [c*s, s**2, 0, -1*c*s, -1*s**2, 0],
                                                                  [0, 0, 0, 0, 0, 0],
                                                                  [-1*c**2, -1*c*s, 0, c**2, c*s, 0],
                                                                  [-1*c*s, -1*s**2, 0, c*s, s**2, 0],
                                                                  [0, 0, 0, 0, 0, 0]])

    def calc_stresses(self, uix, uiy, i_theta, ujx, ujy, j_theta):
        c = math.cos(self.angle)
        s = math.sin(self.angle)
        displacements = np.array([[uix],
                                 [uiy],
                                 [ujx],
                                 [ujy]])
        correcting_matrix = np.array([[c, s, 0, 0],
                                     [-s, c, 0, 0],
                                     [0, 0, c, s],
                                     [0,0, -s, c]])
        local_displacements = correcting_matrix @ displacements

        d  = local_displacements[2] - local_displacements[0]
        axial_force = (self.A * self.E / self.length) * d
        axial_stress = axial_force / self.A
        self.axial = axial_stress.item()
        return [axial_stress.item(), 0, 0]

    def worst_case_stress(self, axial_stress, bending_i, bending_j):
        return axial_stress

    def calc_end_forces_local(self):
        w = self.w
        l = self.length
        return np.array([[w*l/2], [0], [0], [w * l / 2], [0], [0]])

    def calc_equivalent_nodal_loads(self):
        s = math.sin(self.angle)
        c = math.cos(self.angle)
        f_fe_local = self.calc_end_forces_local()

        T = np.array([[c, s, 0, 0, 0, 0],
                      [-s, c, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, c, s, 0],
                      [0, 0, 0, -s, c, 0],
                      [0, 0, 0, 0, 0, 1]])

        return T.T @ f_fe_local

    def shear_func(self, x):
        return 0

    def assemble_segments(self, xs, point_i, point_j):
        t = xs / self.length
        X = point_i[0] + t * (point_j[0] - point_i[0])
        Y = point_i[1] + t * (point_j[1] - point_i[1])

        points = np.array([X, Y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        return segments

    def bending_stress_func(self, n_points=15):
        xs = np.linspace(0, self.length, n_points)
        stresses = []
        for x in xs:
            stresses.append(self.axial)

        return xs, stresses

    def get_stress_info(self, displacements):
        di = displacements[self.node_i.id]
        dj = displacements[self.node_j.id]
        return self.calc_stresses(di[0],di[1], di[2], dj[0], dj[1], dj[2])


class CST:
    def __init__(self, id, nodes, E, v, h, bx = 0, by = 0,  tx = 0, ty = 0, edge_id = None):
        self.id = id
        self.node_i = nodes[0]
        self.node_j = nodes[1]
        self.node_k = nodes[2]
        self.h = h
        self.node_stresses = np.array((3, 3))
        self.tx = tx
        self.ty = ty
        self.edge_id = edge_id

        self.D = np.array([[E/(1-v**2), v*E/(1-v**2), 0],
                               [v*E/(1-v**2), E/(1-v**2), 0],
                               [0, 0, 0.5 * E/(1+v)]])

        self.bx = bx
        self.by = by
        self.axial = 0
        self.A = 0.5 * ((self.node_j.x*self.node_k.y - self.node_k.x*self.node_j.y) + (self.node_k.x*self.node_i.y - self.node_i.x*self.node_k.y) +
                        (self.node_i.x*self.node_j.y - self.node_j.x*self.node_i.y))

        self.B = np.array([[self.node_j.y - self.node_k.y, 0, self.node_k.y - self.node_i.y, 0, self.node_i.y - self.node_j.y, 0],
                           [0, self.node_k.x - self.node_j.x, 0 , self.node_i.x-self.node_k.x, 0, self.node_j.x-self.node_i.x],
                           [self.node_k.x-self.node_j.x, self.node_j.y - self.node_k.y, self.node_i.x - self.node_k.x,
                            self.node_k.y - self.node_i.y, self.node_j.x - self.node_i.x, self.node_i.y - self.node_j.y]])
        self.matrix = self.h / (self.A * 4) * self.B.T @ self.D @ self.B

    def calc_stresses(self, uix, uiy, ujx, ujy, ukx, uky):

        displacements = np.array([[uix],
                                 [uiy],
                                 [ujx],
                                 [ujy],
                                 [ukx],
                                 [uky]])

        strain = self.B @ displacements
        stress_field = self.D @ strain
        self.node_stresses = np.array([stress_field, stress_field, stress_field])
        return [stress_field[0].item(), stress_field[1].item(), stress_field[2].item()]

    def worst_case_stress(self, sxx, syy, sxy):
        avg = (sxx + syy) / 2
        R = math.sqrt(((sxx - syy) / 2) ** 2 + sxy ** 2)
        s1 = avg + R
        s2 = avg - R
        return max([s1, s2], key=abs)

    def calc_end_forces_local(self):
        if(self.edge_id == 1):
            node1 = self.node_i
            num1 = 0
            node2 = self.node_j
            num2 = 2
        elif(self.edge_id == 2):
            node1 = self.node_j
            num1 = 2
            node2 = self.node_k
            num2 = 4
        elif(self.edge_id == 3):
            node1 = self.node_i
            num1 = 0
            node2 = self.node_k
            num2 = 4
        else:
            raise ValueError("Wrong edge id")


        L = math.sqrt((node2.x - node1.x)**2 + (node2.y - node1.y)**2)


        tx1 = self.tx * L/2
        ty1 = self.ty * L/2

        f_nodes = np.zeros((6,1))
        f_nodes[num1] = tx1
        f_nodes[num1 + 1] = ty1
        f_nodes[num2]= tx1
        f_nodes[num2 + 1]= ty1

        return f_nodes



    def calc_equivalent_nodal_loads(self):
        f_b = self.A * self.h / 3 * np.array([[self.bx], [self.by], [self.bx], [self.by], [self.bx], [self.by]])
        if self.edge_id is not None:
            f_nodes = self.calc_end_forces_local()

            return f_b + f_nodes

        return f_b

    def get_nodes(self):
        return [self.node_i, self.node_j, self.node_k]

    def get_stress_info(self, displacements):
        di = displacements[self.node_i.id]
        dj = displacements[self.node_j.id]
        dk = displacements[self.node_k.id]

        return self.calc_stresses(di[0], di[1], dj[0], dj[1], dk[0], dk[1])


class Quad:
    def __init__(self, id, nodes, E, v, h, bx = 0, by = 0,  tx = 0, ty = 0, edge_id = None):
        self.id = id
        self.node1 = nodes[0]
        self.node2 = nodes[1]
        self.node3 = nodes[2]
        self.node4 = nodes[3]
        self.h = h
        self.tx = tx
        self.ty = ty
        self.edge_id = edge_id
        self.bx = bx
        self.by = by
        self.node_stresses = np.array((4,3))
        self.D = np.array([[E / (1 - v ** 2), v * E / (1 - v ** 2), 0],
                           [v * E / (1 - v ** 2), E / (1 - v ** 2), 0],
                           [0, 0, 0.5 * E / (1 + v)]])

        gauss_points = [(-1 / math.sqrt(3), -1 / math.sqrt(3)),
                        (1 / math.sqrt(3), -1 / math.sqrt(3)),
                        (1 / math.sqrt(3), 1 / math.sqrt(3)),
                        (-1 / math.sqrt(3), 1 / math.sqrt(3))]
        gauss_weights = [1.0, 1.0, 1.0, 1.0]

        k = np.zeros((8, 8))

        for (i,point) in enumerate(gauss_points):
            e = point[0]
            n = point[1]
            B = self.B(e, n)
            J = self.J(e, n)
            j = np.linalg.det(J)
            k += gauss_weights[i] * self.h * (B.T @ self.D @ B) * j

        self.matrix = k

    # in form [Ni, dNi/de, dNi/dn]
    def N1(self, e, n):
        return [1/4*(1-e)*(1-n), 1/4*n - 1/4, 1/4*e - 1/4]

    def N2(self, e, n):
        return [1/4*(1+e)*(1-n), -1/4*n + 1/4, -1/4*e - 1/4]

    def N3(self, e, n):
        return [1/4*(1+e)*(1+n), 1/4*n + 1/4, 1/4*e + 1/4]

    def N4(self, e, n):
        return [1/4*(1-e)*(1+n), -1/4*n - 1/4, -1/4*e + 1/4]

    def J(self, e, n):
        partials = np.array([[self.N1(e, n)[1], self.N2(e, n)[1], self.N3(e, n)[1], self.N4(e, n)[1]],
                             [self.N1(e, n)[2], self.N2(e, n)[2], self.N3(e, n)[2], self.N4(e, n)[2]]])
        coords = np.array([[self.node1.x, self.node1.y],
                           [self.node2.x, self.node2.y],
                           [self.node3.x, self.node3.y],
                           [self.node4.x, self.node4.y]])
        return partials @ coords
    def B(self, e, n):
        ij = np.linalg.inv(self.J(e, n))
        B1 = ij @ np.array([[self.N1(e, n)[1]], [self.N1(e, n)[2]]])
        B2 = ij @ np.array([[self.N2(e, n)[1]], [self.N2(e, n)[2]]])
        B3 = ij @ np.array([[self.N3(e, n)[1]], [self.N3(e, n)[2]]])
        B4 = ij @ np.array([[self.N4(e, n)[1]], [self.N4(e, n)[2]]])

        return np.array([[B1[0].item(), 0, B2[0].item(), 0, B3[0].item(), 0, B4[0].item(), 0],
                         [0, B1[1].item(), 0, B2[1].item(), 0, B3[1].item(), 0, B4[1].item()],
                         [B1[1].item(), B1[0].item(), B2[1].item(), B2[0].item(), B3[1].item(), B3[0].item(), B4[1].item(), B4[0].item()]])


    def calc_area(self):
        gauss_points = [(-1 / math.sqrt(3), -1 / math.sqrt(3)),
                        (1 / math.sqrt(3), -1 / math.sqrt(3)),
                        (1 / math.sqrt(3), 1 / math.sqrt(3)),
                        (-1 / math.sqrt(3), 1 / math.sqrt(3))]
        gauss_weights = [1.0, 1.0, 1.0, 1.0]

        area = 0

        for (i, point) in enumerate(gauss_points):
            e = point[0]
            n = point[1]
            J = self.J(e, n)
            j = np.linalg.det(J)
            area += gauss_weights[i] * j

        return area


    def calc_stresses(self, u1x, u1y, u2x, u2y, u3x, u3y, u4x, u4y):

        displacements = np.array([[u1x],
                                 [u1y],
                                 [u2x],
                                 [u2y],
                                 [u3x],
                                 [u3y],
                                  [u4x],
                                  [u4y]])


        strain = self.B(0, 0) @ displacements
        stress_field = self.D @ strain
        self.extrapolate_stress(displacements)

        return [stress_field[0].item(), stress_field[1].item(), stress_field[2].item()]

    def worst_case_stress(self, sxx, syy, sxy):
        avg = (sxx + syy) / 2
        R = math.sqrt(((sxx - syy) / 2) ** 2 + sxy ** 2)
        s1 = avg + R
        s2 = avg - R
        return max([s1, s2], key=abs)

    def calc_end_forces_local(self):
        if(self.edge_id == 1):
            node1 = self.node1
            num1 = 0
            node2 = self.node2
            num2 = 2
        elif(self.edge_id == 2):
            node1 = self.node2
            num1 = 2
            node2 = self.node3
            num2 = 4
        elif(self.edge_id == 3):
            node1 = self.node3
            num1 = 4
            node2 = self.node4
            num2 = 6
        elif (self.edge_id == 4):
            node1 = self.node1
            num1 = 0
            node2 = self.node4
            num2 = 6
        else:
            raise ValueError("Wrong edge id")


        L = math.sqrt((node2.x - node1.x)**2 + (node2.y - node1.y)**2)


        tx1 = self.tx * L/2
        ty1 = self.ty * L/2

        f_nodes = np.zeros((8,1))
        f_nodes[num1] = tx1
        f_nodes[num1 + 1] = ty1
        f_nodes[num2]= tx1
        f_nodes[num2 + 1]= ty1

        return f_nodes



    def calc_equivalent_nodal_loads(self):
        f_b = self.calc_area() * self.h / 4 * np.array([[self.bx], [self.by], [self.bx], [self.by], [self.bx], [self.by], [self.bx], [self.by]])
        if self.edge_id is not None:
            f_nodes = self.calc_end_forces_local()
            return f_b + f_nodes

        return f_b

    def get_nodes(self):
        return [self.node1, self.node2, self.node3, self.node4]

    def get_stress_info(self, displacements):
        d1 = displacements[self.node1.id]
        d2 = displacements[self.node2.id]
        d3 = displacements[self.node3.id]
        d4 = displacements[self.node4.id]
        return self.calc_stresses(d1[0], d1[1], d2[0], d2[1], d3[0], d3[1], d4[0], d4[1])

    def extrapolate_stress(self, displacements):
        e_matrix = np.array([[1 + 0.5*math.sqrt(3), -0.5, 1 - 0.5*math.sqrt(3), -0.5],
                             [-0.5, 1+ 0.5*math.sqrt(3), -0.5, 1 - 0.5*math.sqrt(3)],
                             [1 - 0.5*math.sqrt(3), -0.5, 1 + 0.5*math.sqrt(3), -0.5],
                             [-0.5, 1 - 0.5*math.sqrt(3), -0.5, 1 + 0.5*math.sqrt(3)]])
        gauss_points = [(-1 / math.sqrt(3), -1 / math.sqrt(3)),
                        (1 / math.sqrt(3), -1 / math.sqrt(3)),
                        (1 / math.sqrt(3), 1 / math.sqrt(3)),
                        (-1 / math.sqrt(3), 1 / math.sqrt(3))]

        stresses = np.zeros((4, 3))
        for i, point in enumerate(gauss_points):
            strain = self.B(point[0], point[1]) @ displacements
            stress_field = self.D @ strain

            stresses[i][0] = stress_field[0].item()
            stresses[i][1] = stress_field[1].item()
            stresses[i][2] = stress_field[2].item()

        node_stresses = e_matrix @ stresses
        self.node_stresses = node_stresses

class Structure:
    def __init__(self):
        self.nodes = {}
        self.elements = {}

    def add_node(self, id, x, y, constraint, fx = 0, fy = 0):
        self.nodes[id] = Node(id, x, y, constraint, fx = fx, fy = fy)

    def add_truss(self, id, node1id, node2id, E, A, w = 0):
        self.elements[id] = Truss(id, self.nodes[node1id], self.nodes[node2id], E, A, w)

    def add_beam(self, id, node1id, node2id, E, A, I, c, w=0):
         self.elements[id] = Beam(id, self.nodes[node1id], self.nodes[node2id], E, A, I, c, w)

    def add_cst(self, id, node1id, node2id, node3id, E, v, h, bx = 0, by = 0, tx = 0, ty = 0, edge_id = None):
        self.elements[id] = CST(id, [self.nodes[node1id], self.nodes[node2id], self.nodes[node3id]], E, v, h, bx, by, tx, ty, edge_id)

    def add_quad(self, id, node1id, node2id, node3id, node4id, E, v, h, bx = 0, by = 0, tx = 0, ty = 0, edge_id = None):
        self.elements[id] = Quad(id, [self.nodes[node1id], self.nodes[node2id], self.nodes[node3id], self.nodes[node4id]], E, v, h, bx, by, tx, ty, edge_id)

    def plot(self, info, show_labels=True, cmap="RdBu", figsize=(8, 8), scale_factor=50):
        fig, ax = plt.subplots(figsize=figsize)
        stresses = info[0]
        displacements = info[1]
        average_node_stresses = info[2]

        p_stresses = []
        vals = [element.worst_case_stress(stresses[element.id][0], stresses[element.id][1], stresses[element.id][2]) for element in
                self.elements.values()]
        for element in self.elements.values():
            if isinstance(element, (Truss, Beam)):
                point_i = [element.node_i.x, element.node_i.y]
                point_j = [element.node_j.x, element.node_j.y]
                p_xs, temp_stresses = element.bending_stress_func()
                p_stresses.append(max(temp_stresses, key=abs))
                ax.plot([point_i[0], point_j[0]], [point_i[1], point_j[1]], color="grey", linestyle=":")
            if isinstance(element, (CST)):
                point_i = [element.node_i.x, element.node_i.y]
                point_j = [element.node_j.x, element.node_j.y]
                point_k = [element.node_k.x, element.node_k.y]
                ax.plot([point_i[0], point_j[0]], [point_i[1], point_j[1]], color="grey", linestyle=":")
                ax.plot([point_i[0], point_k[0]], [point_i[1], point_k[1]], color="grey", linestyle=":")
                ax.plot([point_j[0], point_k[0]], [point_j[1], point_k[1]], color="grey", linestyle=":")
            if isinstance(element, (Quad)):
                point1 = [element.node1.x, element.node1.y]
                point2 = [element.node2.x, element.node2.y]
                point3 = [element.node3.x, element.node3.y]
                point4 = [element.node4.x, element.node4.y]
                ax.plot([point1[0], point2[0]], [point1[1], point2[1]], color="grey", linestyle=":")
                ax.plot([point1[0], point4[0]], [point1[1], point4[1]], color="grey", linestyle=":")
                ax.plot([point2[0], point3[0]], [point2[1], point3[1]], color="grey", linestyle=":")
                ax.plot([point3[0], point4[0]], [point3[1], point4[1]], color="grey", linestyle=":")


        cmap = plt.get_cmap(cmap)
        for stress in p_stresses:
            vals.append(stress)

        high = max(vals)
        low = min(vals)
        M = max(high, abs(low))
        if M == 0:
            M = 1

        print("Max Stress:\n", M)

        maxes = []
        for value in displacements.values():
            maxes.append(max(value))
        print("Max displacements:\n", max(maxes))
        norm = Normalize(vmin=-M, vmax=M)
        patches = []
        values = []

        node_ids_in_order = list(self.nodes.keys())
        id_to_index = {nid: i for i, nid in enumerate(node_ids_in_order)}
        triangles = []
        all_x = [self.nodes[nid].x for nid in node_ids_in_order]
        all_y = [self.nodes[nid].y for nid in node_ids_in_order]
        node_delta = {nid: displacements[nid] for nid in self.nodes.keys()}

        for nid, change in node_delta.items():
            all_x[id_to_index[nid]] += 50 * change[0]
            all_y[id_to_index[nid]] += 50 * change[1]

        for element in self.elements.values():
            if(isinstance(element, (Beam, Truss))):

                point_i = [all_x[id_to_index[element.node_i.id]], all_y[id_to_index[element.node_i.id]]]
                point_j = [all_x[id_to_index[element.node_j.id]], all_y[id_to_index[element.node_j.id]]]
                p_xs, p_stresses = element.bending_stress_func()
                segments = element.assemble_segments(p_xs, point_i, point_j)

                lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=3)
                lc.set_array(np.array(p_stresses[:-1]))  # one value per segment, N-1 values for N points
                ax.add_collection(lc)

            if(isinstance(element, CST)):

                coords = [(all_x[id_to_index[element.node_i.id]], all_y[id_to_index[element.node_i.id]]),
                          (all_x[id_to_index[element.node_j.id]], all_y[id_to_index[element.node_j.id]]),
                          (all_x[id_to_index[element.node_k.id]], all_y[id_to_index[element.node_k.id]])]
                patches.append(Polygon(coords, closed=True))
                values.append(element.worst_case_stress(stresses[element.id][0], stresses[element.id][1], stresses[element.id][2]))

            if (isinstance(element, Quad)):

                n1, n2, n3, n4 = element.get_nodes()

                triangles.append([id_to_index[n1.id], id_to_index[n2.id], id_to_index[n3.id]])
                triangles.append([id_to_index[n1.id], id_to_index[n3.id], id_to_index[n4.id]])
                values.append(element.worst_case_stress(stresses[element.id][0], stresses[element.id][1],
                                                        stresses[element.id][2]))
        if(len(triangles) > 0):
            triangulation = tri.Triangulation(all_x, all_y, triangles)
            node_values = [average_node_stresses[nid] for nid in
                           node_ids_in_order]

            ax.tripcolor(triangulation, node_values, shading='gouraud', cmap=cmap, norm=norm)
        if(len(patches) > 0):
            pc = PatchCollection(patches, cmap=cmap, norm=norm)
            pc.set_array(np.array(values))
            ax.add_collection(pc)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Total Stress (psi)")
        xs = []
        ys = []
        fx_nodes = []
        fy_nodes = []
        for node in self.nodes.values():
            nx = displacements[node.id][0] * 50
            ny = displacements[node.id][1] * 50
            xs.append(node.x + nx)
            ys.append(node.y + ny)
            if node.fx != 0 or node.fy != 0:
                ax.quiver(node.x + nx, node.y + ny,
                          node.fx, node.fy, angles='xy', scale_units='xy', scale=500, color='green')
                fx_nodes.append(node.x + nx + node.fx / 500)
                fy_nodes.append(node.y + ny + node.fy / 500)
        for i in range(len(fx_nodes)):
            xs.append(fx_nodes[i])
            ys.append(fy_nodes[i])

        x_margin = (max(xs) - min(xs)) * 0.1
        y_margin = (max(ys) - min(ys)) * 0.1
        ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
        if (min(xs) - x_margin == max(xs) + x_margin):
            ax.set_ylim(min(ys) - 10, max(ys) + 10)
        ax.set_ylim(-10, 10)
        if (min(ys) - y_margin == max(ys) + y_margin):
            ax.set_ylim(min(ys) - 10, max(ys) + 10)
        for i in range(len(fx_nodes)):
            xs.pop()
            ys.pop()

        ax.text(
            0.02, 1.05,
            f"Displacement scale: {scale_factor}x",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray")
        )
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        plt.show()
    def assemble_global_stiffness(self):
        id_to_index = {nid: i for i, nid in enumerate(self.nodes)}
        n_dof = 3*len(self.nodes.keys())
        K = np.zeros((n_dof, n_dof))

        for element in self.elements.values():
            local_k = element.matrix
            if isinstance(element, (Truss, Beam)):
                i_idx = id_to_index[element.node_i.id]
                j_idx = id_to_index[element.node_j.id]
                dof = [3*i_idx, 3*i_idx +1, 3*i_idx +2, 3*j_idx, 3*j_idx +1, 3*j_idx + 2]
                K[np.ix_(dof, dof)] += local_k

            if isinstance(element, CST):
                i_idx = id_to_index[element.node_i.id]
                j_idx = id_to_index[element.node_j.id]
                k_idx = id_to_index[element.node_k.id]
                dof  = [3*i_idx, 3*i_idx +1, 3*j_idx, 3*j_idx + 1, 3*k_idx, 3*k_idx + 1]
                K[np.ix_(dof, dof)] += local_k

            if isinstance(element, Quad):
                idx1 = id_to_index[element.node1.id]
                idx2 = id_to_index[element.node2.id]
                idx3 = id_to_index[element.node3.id]
                idx4 = id_to_index[element.node4.id]
                dof = [3 * idx1, 3 * idx1 + 1, 3 * idx2, 3 * idx2 + 1, 3 * idx3, 3 * idx3 + 1, 3 * idx4, 3*idx4 + 1]
                K[np.ix_(dof, dof)] += local_k

        return K



def solve(structure):
    K = structure.assemble_global_stiffness()

    id_to_index = {nid: i for i, nid in enumerate(structure.nodes)}
    node_ids = sorted(structure.nodes.keys())
    n_dof = 3 * len(node_ids)
    F = np.zeros(n_dof)

    for node in structure.nodes.values():
        F[3 * id_to_index[node.id]] += node.fx
        F[3 * id_to_index[node.id] + 1] += node.fy
        F[3 * id_to_index[node.id] + 2] += node.mz


    for element in structure.elements.values():
        forces = element.calc_equivalent_nodal_loads()
        if isinstance(element, (Truss, Beam)):
            F[(3 * id_to_index[element.node_i.id])] -= forces[0].item()
            F[(3 * id_to_index[element.node_i.id] + 1)] -= forces[1].item()
            F[(3 * id_to_index[element.node_i.id] + 2)] -= forces[2].item()
            F[(3 * id_to_index[element.node_j.id])] += forces[3].item()
            F[(3 * id_to_index[element.node_j.id] + 1)] -= forces[4].item()
            F[(3 * id_to_index[element.node_j.id] + 2)] -= forces[5].item()
        if isinstance(element, CST):
            F[(3 * id_to_index[element.node_i.id])] += forces[0].item()
            F[(3 * id_to_index[element.node_i.id]) + 1] += forces[1].item()
            F[(3 * id_to_index[element.node_j.id])] += forces[2].item()
            F[(3 * id_to_index[element.node_j.id]) + 1] += forces[3].item()
            F[(3 * id_to_index[element.node_k.id])] += forces[4].item()
            F[(3 * id_to_index[element.node_k.id] + 1)] += forces[5].item()
        if isinstance(element, Quad):
            F[(3 * id_to_index[element.node1.id])] += forces[0].item()
            F[(3 * id_to_index[element.node1.id]) + 1] += forces[1].item()
            F[(3 * id_to_index[element.node2.id])] += forces[2].item()
            F[(3 * id_to_index[element.node2.id]) + 1] += forces[3].item()
            F[(3 * id_to_index[element.node3.id])] += forces[4].item()
            F[(3 * id_to_index[element.node3.id] + 1)] += forces[5].item()
            F[(3 * id_to_index[element.node4.id])] += forces[6].item()
            F[(3 * id_to_index[element.node4.id] + 1)] += forces[7].item()


    free_dof = []
    reaction_nodes = []
    for node in structure.nodes.values():
        if node.constraint == "fixed":
            reaction_nodes.append(node)
        elif node.constraint == "pinned":
            if node.has_beam_connection(structure):
                free_dof.append(id_to_index[node.id] * 3 + 2)
            reaction_nodes.append(node)
        elif node.constraint == "free":
            free_dof.extend([id_to_index[node.id] * 3, id_to_index[node.id] * 3 + 1])
            if node.has_beam_connection(structure):
                free_dof.append(id_to_index[node.id] * 3 + 2)
        elif node.constraint == "roller-y":
            free_dof.append(id_to_index[node.id] * 3 + 1)
            if node.has_beam_connection(structure):
                free_dof.append(id_to_index[node.id] * 3 + 2)
            reaction_nodes.append(node)
        elif node.constraint == "roller-x":
            free_dof.append(id_to_index[node.id] * 3)
            if node.has_beam_connection(structure):
                free_dof.append(id_to_index[node.id] * 3 + 2)
            reaction_nodes.append(node)
        else:
            raise ValueError(f"Unknown constraint type '{node.constraint}' on node {node.id}")

    Kff = K[np.ix_(free_dof, free_dof)]
    Ff = F[free_dof]



    d_free = np.linalg.solve(Kff, Ff)

    d = np.zeros(n_dof)
    d[free_dof] = d_free

    R = K @ d - F

    displacements = {nid: (d[3*i], d[3*i+1], d[3*i+2]) for nid, i in id_to_index.items()}
    reactions = {nid: (R[3*i], R[3*i + 1], R[3*i + 2]) for nid, i in id_to_index.items()}


    print("Displacements:")
    for node in structure.nodes.values():
        print("Node ", node.id, " - x: ", round(float(displacements[node.id][0]), 5), " y: " , round(float(displacements[node.id][1]), 5), " theta: " , round(float(displacements[node.id][2]), 5))
    print("Reaction Forces:")
    for node in reaction_nodes:
        print("Node ", node.id, " - x: " , round(float(reactions[node.id][0]), 5), " y: " , round(float(reactions[node.id][1]), 5))

    stresses = {}
    line_stresses = {}
    tensor_stresses = {}
    plane_elements = []

    for element in structure.elements.values():
        stress_info = element.get_stress_info(displacements)
        stresses[element.id] = stress_info

        if isinstance(element, (Truss, Beam)):
            line_stresses[element.id] = stress_info

        if isinstance(element, (CST, Quad)):
            plane_elements.append(element)
            tensor_stresses[element.id] = stress_info


    print("Line Element Stresses: ")
    for element_id in line_stresses.keys():
        print("Element ", element_id, ": Axial Stress - ", round(line_stresses[element_id][0], 5), " Bending Stress (i) - ",
              round(line_stresses[element_id][1] , 5), " Bending Stress (j) - ", round(line_stresses[element_id][2], 5))

    print("2D Element Stresses: ")
    for element_id in tensor_stresses.keys():
        print("Element ", element_id, ": Stress (xx) - ", round(tensor_stresses[element_id][0], 5), " Stress (yy) - ",
              round(tensor_stresses[element_id][1], 5), " Stress (xy) - ", round(tensor_stresses[element_id][2], 5))

    node_stress_contributions = {}  # node_id -> list of [sxx, syy, sxy] arrays

    for element in plane_elements:
        node_stresses = element.node_stresses
        for local_idx, node in enumerate(element.get_nodes()):
            if(node.id not in node_stress_contributions):
                node_stress_contributions[node.id] = []

            node_stress_contributions[node.id].append(node_stresses[local_idx])


    averaged_node_stress = {}
    for node_id, contributions in node_stress_contributions.items():
        node_stresses = np.mean(contributions, axis=0)
        averaged_node_stress[node_id] = max(node_stresses, key = abs)

    return [stresses, displacements, averaged_node_stress]



def generate_quad_mesh(structure, width, height, nx, ny, E, v, h, start_id=1000):
    dx = width / nx
    dy = height / ny
    grid_to_id = {}
    node_id = start_id

    for j in range(ny + 1):
        for i in range(nx + 1):
            x, y = i * dx, j * dy
            constraint = "fixed" if j == 0 else "free"   # bottom edge fixed
            structure.add_node(node_id, x, y, constraint)
            grid_to_id[(i, j)] = node_id
            node_id += 1

    elem_id = start_id
    for j in range(ny):
        for i in range(nx):
            n1 = grid_to_id[(i, j)]       # bottom-left
            n2 = grid_to_id[(i+1, j)]     # bottom-right
            n3 = grid_to_id[(i+1, j+1)]   # top-right
            n4 = grid_to_id[(i, j+1)]     # top-left

            # apply shear traction along the top edge of top-row elements only
            if j == ny - 1:
                structure.add_quad(elem_id, n1, n2, n3, n4, E, v, h, tx=500, edge_id=3)
            else:
                structure.add_quad(elem_id, n1, n2, n3, n4, E, v, h)
            elem_id += 1

    return grid_to_id

def generate_rectangular_quad_mesh(structure, width, height, nx, ny, E, h, v,
                                start_id_offset=0):
    """
    Builds a rectangular plate mesh of CST triangles.
    nx, ny = number of divisions along width/height (so (nx+1)*(ny+1) nodes)
    Returns: dict mapping (i,j) grid position -> node id, for easy lookup later
    """
    dx = width / nx
    dy = height / ny
    grid_to_id = {}
    node_id = start_id_offset

    # Create nodes
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = i * dx
            y = j * dy
            constraint = "fixed" if i == 0 or i==nx else "free"  # left edge fixed
            structure.add_node(node_id, x, y, constraint)
            grid_to_id[(i, j)] = node_id
            node_id += 1

    # Create CST elements: split each rectangular cell into 2 triangles
    elem_id = start_id_offset
    for j in range(ny):
        for i in range(nx):
            n1 = grid_to_id[(i, j)]
            n2 = grid_to_id[(i+1, j)]
            n3 = grid_to_id[(i+1, j+1)]
            n4 = grid_to_id[(i, j+1)]


            structure.add_quad(elem_id, n1, n2, n3, n4, E, v,h, by = -100)
            elem_id += 1


    return grid_to_id

def generate_rectangular_cst_mesh(structure, width, height, nx, ny, E, h, v,
                                start_id_offset=0):
    """
    Builds a rectangular plate mesh of CST triangles.
    nx, ny = number of divisions along width/height (so (nx+1)*(ny+1) nodes)
    Returns: dict mapping (i,j) grid position -> node id, for easy lookup later
    """
    dx = width / nx
    dy = height / ny
    grid_to_id = {}
    node_id = start_id_offset

    # Create nodes
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = i * dx
            y = j * dy
            constraint = "fixed" if i == 0 else "free"
            structure.add_node(node_id, x, y, constraint)
            grid_to_id[(i, j)] = node_id
            node_id += 1

    elem_id = start_id_offset
    for j in range(ny):
        for i in range(nx):
            n1 = grid_to_id[(i, j)]
            n2 = grid_to_id[(i+1, j)]
            n3 = grid_to_id[(i+1, j+1)]
            n4 = grid_to_id[(i, j+1)]

            structure.add_cst(elem_id, n1, n2, n3, E, v,h, by = -100)
            elem_id += 1

            structure.add_cst(elem_id, n1, n3, n4, E, v, h, by = -100)
            elem_id += 1

    return grid_to_id

def nodetest():
    s = Structure()

    generate_rectangular_cst_mesh(s, 20, 5, 20, 5, 29000000, 1, 1/4)

    info = solve(s)
    s.plot(info)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    nodetest()

