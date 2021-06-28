
#
# transformations
# Classes representing geometric transformations
# Author: Johan Ofverstedt
#

import math
import numpy as np
import scipy.ndimage.interpolation

###
### Base class
###

class TransformBase:

    def __init__(self, dim, nparam):
        self.dim = dim
        self.param = np.zeros((nparam,))

    def get_dim(self):
        return self.dim

    def get_params(self):
        return self.param
    
    def set_params(self, params):
        self.param[:] = params[:]

    def get_param(self, index):
        return self.param[index]
    
    def set_param(self, index, value):
        self.param[index] = value

    def set_params_const(self, value):
        self.param[:] = value

    def step_param(self, index, step_length):
        self.param[index] = self.param[index] + step_length

    def step_params(self, grad, step_length):
        self.param = self.param + grad * step_length

    def get_param_count(self):
        return self.param.size

    def copy(self):
        t = self.copy_child()
        t.set_params(self.get_params())

        return t

    def copy_child(self):
        raise NotImplementedError

    def __call__(self, pnts):
        return self.transform(pnts)
    
    def transform(self, pnts):
        raise NotImplementedError

    def warp(self, In, Out, in_spacing=None, out_spacing=None, mode='spline', bg_value=0.0):
        linspaces = [np.linspace(0, Out.shape[i]*out_spacing[i], Out.shape[i], endpoint=False) for i in range(Out.ndim)]

        grid = np.array(np.meshgrid(*linspaces,indexing='ij'))

        grid = grid.reshape((Out.ndim, np.prod(Out.shape)))
        grid = np.moveaxis(grid, 0, 1)

        grid_transformed = self.transform(grid)
        if in_spacing is not None:
            grid_transformed[:, :] = grid_transformed[:, :] * (1.0 / in_spacing[:])
        
        grid_transformed = np.moveaxis(grid_transformed, 0, 1)
        grid_transformed = grid_transformed.reshape((Out.ndim,) + Out.shape)
        
        if mode == 'spline' or mode == 'cubic':
            scipy.ndimage.interpolation.map_coordinates(In, coordinates=grid_transformed, output=Out, cval = bg_value)
        elif mode == 'linear':
            scipy.ndimage.interpolation.map_coordinates(In, coordinates=grid_transformed, output=Out, order=1, cval = bg_value)
        elif mode == 'nearest':
            scipy.ndimage.interpolation.map_coordinates(In, coordinates=grid_transformed, output=Out, order=0, cval = bg_value)

    def itk_transform_string(self):
        s = '#Insight Transform File V1.0\n'
        #s = s + '#Transform 0\n'
        return s + self.itk_transform_string_rec(0)

    def itk_transform_string_rec(self, index):
        raise NotImplementedError        

    def grad(self, pnts, gradients, output_gradients):
        raise NotImplementedError

    def invert(self):
        raise NotImplementedError

    # Must be called on the forward transform
    # Default falls back on numerical differentiation
    def grad_inverse_to_forward(self, inv_grad):
        D = self.inverse_to_forward_matrix()
        if D is None:
            D = self.inverse_to_forward_matrix_num()
        #print(D)
        return D.dot(inv_grad)

    def inverse_to_forward_matrix(self):
        return None

    def diff(self, index, pnts, eps=1e-6):
        f = self.copy()
        b = self.copy()
        f.step_param(index, eps)
        b.step_param(index, -eps)
        fpnts = f.transform(pnts)
        bpnts = b.transform(pnts)
        delta = (fpnts - bpnts) * (1.0 / (2.0 * eps))
        return delta

    def grad_num(self, pnts, gradients, eps=1e-6):
        res = np.zeros((self.get_param_count(),))
        if self.get_param_count() == 1:
            d = self.diff(0, pnts, eps)
            res[0] = res[0] + (d * gradients).sum()
        else:
            for i in range(self.get_param_count()):
                d = self.diff(i, pnts, eps)
                summed = (d * gradients).sum()
                res[i] = res[i] + summed
        return res
        
    # Utility function to differentiate the inverse transformation
    # with respect to the forward transformation numerically
    def diff_inv(self, index, eps=1e-6):
        f = self.copy()
        b = self.copy()
        f.step_param(index, eps)
        b.step_param(index, -eps)
        return (f.invert().get_params() - b.invert().get_params()) / (2.0 * eps)

    def inverse_to_forward_matrix_num(self, eps=1e-6):
        D = np.zeros((self.get_param_count(),self.get_param_count()))
        for i in range(self.get_param_count()):
            d = self.diff_inv(i, eps)
            D[i, :] = d
        return D
        
    # Must be called on the forward transform
    def grad_inverse_to_forward_num(self, inv_grad, eps=1e-6):
        D = self.inverse_to_forward_matrix_num(eps)
        #print(D)
        G = D.dot(inv_grad)
        #print(G)
        return G
        #res = np.zeros((self.get_param_count(),))
        #for i in range(self.get_param_count()):
        #    d = self.diff_inv(i, eps)
        #    print("d(%d): %s" % (i, str(d)))
        #    res[i] = (d.dot(inv_grad))#.sum()
        #return res

###
### Translation transform
###

class TranslationTransform(TransformBase):
    def __init__(self, dim):
        TransformBase.__init__(self, dim, dim)

    def copy_child(self):
        return TranslationTransform(self.get_dim())
    
    def transform(self, pnts):
        offset = self.get_params()
        #print(pnts)
        return pnts + offset

    def grad(self, pnts, gradients, output_gradients):
        res = gradients.sum(axis=0)
        if output_gradients == True:
            return res, gradients
        else:
            return res

    def invert(self):
        self_inv = self.copy()
        self_inv.set_params(-self_inv.get_params())

        return self_inv

    def inverse_to_forward_matrix(self):
        return -np.eye(self.get_param_count(), self.get_param_count())

    def itk_transform_string_rec(self, index):
        s = '#Transform %d\n' % index
        s = s + 'Transform: TranslationTransform_double_%d_%d\n' % (self.get_dim(), self.get_dim())
        s = s + 'Parameters:'
        for i in range(self.get_param_count()):
            s = s + (' %f' % self.get_param(i))
        s = s + '\n'
        s = s + 'FixedParameters:\n'

        return s 

###
### Rotate2DTransform
###

class Rotate2DTransform(TransformBase):
    def __init__(self):
        TransformBase.__init__(self, 2, 1)

    def copy_child(self):
        return Rotate2DTransform()
    
    def transform(self, pnts):
        theta = self.get_param(0)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        M = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])
        
        return pnts.dot(M)

    def grad(self, pnts, gradients, output_gradients):
        res = np.zeros((1,))
        theta = self.get_param(0)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        Mprime = np.array([[-sin_theta, cos_theta], [-cos_theta, -sin_theta]])
        Mprimepnts = pnts.dot(Mprime)
        res[:] = (Mprimepnts * gradients).sum()

        if output_gradients == True:
            M = np.transpose(np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]]))
        
            return res, gradients.dot(M)
        else:
            return res

    def invert(self):
        self_inv = self.copy()
        self_inv.set_params(-self_inv.get_params())

        return self_inv

    def inverse_to_forward_matrix(self):
        return np.array([[-1.0]])
      
###
### Rigid2DTransform
###

class Rigid2DTransform(TransformBase):
    def __init__(self):
        TransformBase.__init__(self, 2, 3)

    def copy_child(self):
        return Rigid2DTransform()
    
    def transform(self, pnts):
        param = self.get_params()
        theta = param[0]
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        M = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])
        
        res = pnts.dot(M)
        res[..., :] = res[..., :] + param[1:]
        return res
    '''
    def transform(self, pnts):
        res = np.zeros_like(pnts)
        param = self.get_params()
        theta = param[0]
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        M = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])
        
        pnts.dot(M, out = res)
        res[..., :] = res[..., :] + param[1:]
        return res
        #return pnts.dot(M) + param[1:]
    '''
    def grad(self, pnts, gradients, output_gradients):
        res = np.zeros((3,))
        theta = self.get_param(0)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        Mprime = np.array([[-sin_theta, cos_theta], [-cos_theta, -sin_theta]])
        #Mprimepnts = pnts.dot(Mprime)
        res[0] = (pnts.dot(Mprime) * gradients).sum()
        res[1:] = gradients.sum(axis=0)

        if output_gradients == True:
            M = np.transpose(np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]]))
        
            return res, gradients.dot(M)
        else:
            return res

    def invert(self):
        self_inv = self.copy()

        inv_theta = -self.get_param(0)
        self_inv.set_param(0, inv_theta)

        cos_theta = math.cos(inv_theta)
        sin_theta = math.sin(inv_theta)

        M = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

        t = self.get_params()[1:]

        tinv = M.dot(-t)

        self_inv.set_param(1, tinv[0])
        self_inv.set_param(2, tinv[1])
        
        return self_inv

    def inverse_to_forward_matrix(self):
        theta = self.get_param(0)
        inv_theta = -theta

        cos_theta_inv = math.cos(inv_theta)
        sin_theta_inv = math.sin(inv_theta)

        Mprime = np.array([[-sin_theta_inv, -cos_theta_inv], [cos_theta_inv, -sin_theta_inv]])
        t = self.get_params()[1:]

        trot = Mprime.dot(t)
        
        D0 = [-1.0, trot[0], trot[1]]
        D1 = [0.0, -cos_theta_inv, -sin_theta_inv]
        D2 = [0.0, sin_theta_inv, -cos_theta_inv]

        return np.array([D0, D1, D2])

    def itk_transform_string_rec(self, index):
        s = '#Transform %d\n' % index        
        s = s + 'Transform: Rigid2DTransformBase_double_%d_%d\n' % (self.get_dim(), self.get_dim())
        s = s + 'Parameters:'
        for i in range(self.get_param_count()):
            s = s + (' %f' % self.get_param(i))
        s = s + '\n'
        s = s + 'FixedParameters:'
        s = s + '\n'

        return s 
    
###
### AffineTransform
###

# Format, in homogeneous coordinates
# | a00 ... a0n t0 | |x1|
# | a10 ... a1n t1 | |x2|
# | ........... .. | |..|
# | an0 ... ann tn | |xn|
# |  0  ...  0  1  | |1 |
#

class AffineTransform(TransformBase):
    def __init__(self, dim):
        TransformBase.__init__(self, dim, (dim*dim) + dim)
        param = np.zeros((dim*(dim+1),))
        param[:dim*dim:dim+1] = 1
        self.set_params(param)

    def copy_child(self):
        return AffineTransform(self.get_dim())
    
    def transform(self, pnts):
        param = self.get_params()
        dim = self.get_dim()
        m = np.transpose(param[:dim*dim].reshape((dim, dim)))
        t = param[dim*dim:]

        return pnts.dot(m) + t

    ### Special affine functions

    def get_matrix(self):
        dim = self.get_dim()
        param = self.get_params()
        
        return param[:dim*dim].reshape((dim, dim))

    def get_translation(self):
        dim = self.get_dim()
        param = self.get_params()
        
        return param[dim*dim:]

    def set_matrix(self, M):
        dim = self.get_dim()
        param = self.get_params()
        
        param[:dim*dim] = M.reshape((dim*dim,))
    def set_translation(self, t):
        dim = self.get_dim()
        param = self.get_params()
        
        param[dim*dim:] = t[:]

    # Generates homogeneous coordinate matrix
    def homogeneous(self):
        dim = self.get_dim()
        param = self.get_params()
        h = np.zeros([dim+1, dim+1])
        
        h[0:dim, 0:dim] = self.get_matrix()#param[:dim*dim].reshape((dim, dim))
        h[:dim, dim] = self.get_translation()#param[dim*dim:]
        h[dim, dim] = 1
        return h

    # Convert from homogeneous coordinate matrix
    def convert_from_homogeneous(self, h):
        dim = self.get_dim()
        self.set_matrix(h[:dim, :dim])
        self.set_translation(h[:dim, dim])

    ### End of Special affine functions

    # Invert transformation
    def invert(self):
        dim = self.get_dim()
        self_inv = AffineTransform(dim)
        h = self.homogeneous()
        h = np.linalg.inv(h)

        self_inv.convert_from_homogeneous(h)
        #self_inv.set_params(h[0:dim+1, 0:dim].reshape((self.get_param_count(),)))

        #self_inv.set_params(h[0:dim, 0:dim])

        return self_inv
        
    def grad(self, pnts, gradients, output_gradients):
        g_out = np.zeros((self.get_param_count(),))

        for i in range(self.dim):
            for j in range(self.dim):
                g_out[i*self.dim + j] = (pnts[:, j] * gradients[:, i]).sum()
        #for i in range(self.dim):
        #    for j in range(self.dim):
        #        g_out[i*self.dim:(i+1)*self.dim] = (pnts[:, j] * gradients[:, i]).sum()
        g_out[(self.dim*self.dim):] = gradients.sum(axis=0)

        if output_gradients == True:
            param = self.get_params()
            dim = self.get_dim()
            m = param[:dim*dim].reshape((dim, dim))

            upd_gradients = gradients.dot(m)

            return g_out, upd_gradients
        else:
            return g_out

    def inverse_to_forward_matrix(self):
        if self.get_dim() == 2:
            return self._inverse_to_forward_matrix_2d(self.get_params())
        elif self.get_dim() == 3:
            return self._inverse_to_forward_matrix_3d(self.get_params())
        else:
            return self.inverse_to_forward_matrix_num()
    
    def _inverse_to_forward_matrix_2d(self, param):
    
        # Generate local variables for each parameter
        a_0_0 = param[0]
        a_0_1 = param[1]
        a_1_0 = param[2]
        a_1_1 = param[3]
        a_0_2 = param[4]
        a_1_2 = param[5]
    
        # Compute determinant
        det = a_0_0*a_1_1 - a_0_1*a_1_0
    
        # Compute and return final matrix
        return np.array(
            [
                [-a_1_1**2/det**2, a_0_1*a_1_1/det**2, a_1_0*a_1_1/det**2, -a_0_1*a_1_0/det**2, -a_1_1*(a_0_1*a_1_2 - a_0_2*a_1_1)/det**2, (a_1_1*(a_0_0*a_1_2 - a_0_2*a_1_0) - a_1_2*det)/det**2],
                [a_1_0*a_1_1/det**2, -a_0_0*a_1_1/det**2, -a_1_0**2/det**2, a_0_0*a_1_0/det**2, (a_1_0*(a_0_1*a_1_2 - a_0_2*a_1_1) + a_1_2*det)/det**2, -a_1_0*(a_0_0*a_1_2 - a_0_2*a_1_0)/det**2],
                [a_0_1*a_1_1/det**2, -a_0_1**2/det**2, -a_0_0*a_1_1/det**2, a_0_0*a_0_1/det**2, a_0_1*(a_0_1*a_1_2 - a_0_2*a_1_1)/det**2, (-a_0_1*(a_0_0*a_1_2 - a_0_2*a_1_0) + a_0_2*det)/det**2],
                [-a_0_1*a_1_0/det**2, a_0_0*a_0_1/det**2, a_0_0*a_1_0/det**2, -a_0_0**2/det**2, -(a_0_0*(a_0_1*a_1_2 - a_0_2*a_1_1) + a_0_2*det)/det**2, a_0_0*(a_0_0*a_1_2 - a_0_2*a_1_0)/det**2],
                [0, 0, 0, 0, -a_1_1/det, a_1_0/det],
                [0, 0, 0, 0, a_0_1/det, -a_0_0/det]
            ])

    def _inverse_to_forward_matrix_3d(self, param):
    
        # Generate local variables for each parameter
        a_0_0 = param[0]
        a_0_1 = param[1]
        a_0_2 = param[2]
        a_1_0 = param[3]
        a_1_1 = param[4]
        a_1_2 = param[5]
        a_2_0 = param[6]
        a_2_1 = param[7]
        a_2_2 = param[8]
        a_0_3 = param[9]
        a_1_3 = param[10]
        a_2_3 = param[11]
    
        # Compute determinant
        det = a_0_0*a_1_1*a_2_2 - a_0_0*a_1_2*a_2_1 - a_0_1*a_1_0*a_2_2 + a_0_1*a_1_2*a_2_0 + a_0_2*a_1_0*a_2_1 - a_0_2*a_1_1*a_2_0
    
        # Compute and return final matrix
        return np.array(
            [
                [-(a_1_1*a_2_2 - a_1_2*a_2_1)**2/det**2, (a_0_1*a_2_2 - a_0_2*a_2_1)*(a_1_1*a_2_2 - a_1_2*a_2_1)/det**2, -(a_0_1*a_1_2 - a_0_2*a_1_1)*(a_1_1*a_2_2 - a_1_2*a_2_1)/det**2, (a_1_0*a_2_2 - a_1_2*a_2_0)*(a_1_1*a_2_2 - a_1_2*a_2_1)/det**2, (a_2_2*det - (a_0_0*a_2_2 - a_0_2*a_2_0)*(a_1_1*a_2_2 - a_1_2*a_2_1))/det**2, (-a_1_2*det + (a_0_0*a_1_2 - a_0_2*a_1_0)*(a_1_1*a_2_2 - a_1_2*a_2_1))/det**2, -(a_1_0*a_2_1 - a_1_1*a_2_0)*(a_1_1*a_2_2 - a_1_2*a_2_1)/det**2, (-a_2_1*det + (a_0_0*a_2_1 - a_0_1*a_2_0)*(a_1_1*a_2_2 - a_1_2*a_2_1))/det**2, (a_1_1*det - (a_0_0*a_1_1 - a_0_1*a_1_0)*(a_1_1*a_2_2 - a_1_2*a_2_1))/det**2, (a_1_1*a_2_2 - a_1_2*a_2_1)*(a_0_1*a_1_2*a_2_3 - a_0_1*a_1_3*a_2_2 - a_0_2*a_1_1*a_2_3 + a_0_2*a_1_3*a_2_1 + a_0_3*a_1_1*a_2_2 - a_0_3*a_1_2*a_2_1)/det**2, (det*(a_1_2*a_2_3 - a_1_3*a_2_2) - (a_1_1*a_2_2 - a_1_2*a_2_1)*(a_0_0*a_1_2*a_2_3 - a_0_0*a_1_3*a_2_2 - a_0_2*a_1_0*a_2_3 + a_0_2*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_2 - a_0_3*a_1_2*a_2_0))/det**2, (det*(-a_1_1*a_2_3 + a_1_3*a_2_1) + (a_1_1*a_2_2 - a_1_2*a_2_1)*(a_0_0*a_1_1*a_2_3 - a_0_0*a_1_3*a_2_1 - a_0_1*a_1_0*a_2_3 + a_0_1*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_1 - a_0_3*a_1_1*a_2_0))/det**2],
                [(a_1_0*a_2_2 - a_1_2*a_2_0)*(a_1_1*a_2_2 - a_1_2*a_2_1)/det**2, -(a_2_2*det + (a_0_1*a_2_2 - a_0_2*a_2_1)*(a_1_0*a_2_2 - a_1_2*a_2_0))/det**2, (a_1_2*det + (a_0_1*a_1_2 - a_0_2*a_1_1)*(a_1_0*a_2_2 - a_1_2*a_2_0))/det**2, -(a_1_0*a_2_2 - a_1_2*a_2_0)**2/det**2, (a_0_0*a_2_2 - a_0_2*a_2_0)*(a_1_0*a_2_2 - a_1_2*a_2_0)/det**2, -(a_0_0*a_1_2 - a_0_2*a_1_0)*(a_1_0*a_2_2 - a_1_2*a_2_0)/det**2, (a_1_0*a_2_1 - a_1_1*a_2_0)*(a_1_0*a_2_2 - a_1_2*a_2_0)/det**2, (a_2_0*det - (a_0_0*a_2_1 - a_0_1*a_2_0)*(a_1_0*a_2_2 - a_1_2*a_2_0))/det**2, (-a_1_0*det + (a_0_0*a_1_1 - a_0_1*a_1_0)*(a_1_0*a_2_2 - a_1_2*a_2_0))/det**2, (det*(-a_1_2*a_2_3 + a_1_3*a_2_2) - (a_1_0*a_2_2 - a_1_2*a_2_0)*(a_0_1*a_1_2*a_2_3 - a_0_1*a_1_3*a_2_2 - a_0_2*a_1_1*a_2_3 + a_0_2*a_1_3*a_2_1 + a_0_3*a_1_1*a_2_2 - a_0_3*a_1_2*a_2_1))/det**2, (a_1_0*a_2_2 - a_1_2*a_2_0)*(a_0_0*a_1_2*a_2_3 - a_0_0*a_1_3*a_2_2 - a_0_2*a_1_0*a_2_3 + a_0_2*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_2 - a_0_3*a_1_2*a_2_0)/det**2, (det*(a_1_0*a_2_3 - a_1_3*a_2_0) - (a_1_0*a_2_2 - a_1_2*a_2_0)*(a_0_0*a_1_1*a_2_3 - a_0_0*a_1_3*a_2_1 - a_0_1*a_1_0*a_2_3 + a_0_1*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_1 - a_0_3*a_1_1*a_2_0))/det**2],
                [-(a_1_0*a_2_1 - a_1_1*a_2_0)*(a_1_1*a_2_2 - a_1_2*a_2_1)/det**2, (a_2_1*det + (a_0_1*a_2_2 - a_0_2*a_2_1)*(a_1_0*a_2_1 - a_1_1*a_2_0))/det**2, -(a_1_1*det + (a_0_1*a_1_2 - a_0_2*a_1_1)*(a_1_0*a_2_1 - a_1_1*a_2_0))/det**2, (a_1_0*a_2_1 - a_1_1*a_2_0)*(a_1_0*a_2_2 - a_1_2*a_2_0)/det**2, -(a_2_0*det + (a_0_0*a_2_2 - a_0_2*a_2_0)*(a_1_0*a_2_1 - a_1_1*a_2_0))/det**2, (a_1_0*det + (a_0_0*a_1_2 - a_0_2*a_1_0)*(a_1_0*a_2_1 - a_1_1*a_2_0))/det**2, -(a_1_0*a_2_1 - a_1_1*a_2_0)**2/det**2, (a_0_0*a_2_1 - a_0_1*a_2_0)*(a_1_0*a_2_1 - a_1_1*a_2_0)/det**2, -(a_0_0*a_1_1 - a_0_1*a_1_0)*(a_1_0*a_2_1 - a_1_1*a_2_0)/det**2, (det*(a_1_1*a_2_3 - a_1_3*a_2_1) + (a_1_0*a_2_1 - a_1_1*a_2_0)*(a_0_1*a_1_2*a_2_3 - a_0_1*a_1_3*a_2_2 - a_0_2*a_1_1*a_2_3 + a_0_2*a_1_3*a_2_1 + a_0_3*a_1_1*a_2_2 - a_0_3*a_1_2*a_2_1))/det**2, (det*(-a_1_0*a_2_3 + a_1_3*a_2_0) - (a_1_0*a_2_1 - a_1_1*a_2_0)*(a_0_0*a_1_2*a_2_3 - a_0_0*a_1_3*a_2_2 - a_0_2*a_1_0*a_2_3 + a_0_2*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_2 - a_0_3*a_1_2*a_2_0))/det**2, (a_1_0*a_2_1 - a_1_1*a_2_0)*(a_0_0*a_1_1*a_2_3 - a_0_0*a_1_3*a_2_1 - a_0_1*a_1_0*a_2_3 + a_0_1*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_1 - a_0_3*a_1_1*a_2_0)/det**2],
                [(a_0_1*a_2_2 - a_0_2*a_2_1)*(a_1_1*a_2_2 - a_1_2*a_2_1)/det**2, -(a_0_1*a_2_2 - a_0_2*a_2_1)**2/det**2, (a_0_1*a_1_2 - a_0_2*a_1_1)*(a_0_1*a_2_2 - a_0_2*a_2_1)/det**2, -(a_2_2*det + (a_0_1*a_2_2 - a_0_2*a_2_1)*(a_1_0*a_2_2 - a_1_2*a_2_0))/det**2, (a_0_0*a_2_2 - a_0_2*a_2_0)*(a_0_1*a_2_2 - a_0_2*a_2_1)/det**2, (a_0_2*det - (a_0_0*a_1_2 - a_0_2*a_1_0)*(a_0_1*a_2_2 - a_0_2*a_2_1))/det**2, (a_2_1*det + (a_0_1*a_2_2 - a_0_2*a_2_1)*(a_1_0*a_2_1 - a_1_1*a_2_0))/det**2, -(a_0_0*a_2_1 - a_0_1*a_2_0)*(a_0_1*a_2_2 - a_0_2*a_2_1)/det**2, (-a_0_1*det + (a_0_0*a_1_1 - a_0_1*a_1_0)*(a_0_1*a_2_2 - a_0_2*a_2_1))/det**2, -(a_0_1*a_2_2 - a_0_2*a_2_1)*(a_0_1*a_1_2*a_2_3 - a_0_1*a_1_3*a_2_2 - a_0_2*a_1_1*a_2_3 + a_0_2*a_1_3*a_2_1 + a_0_3*a_1_1*a_2_2 - a_0_3*a_1_2*a_2_1)/det**2, (det*(-a_0_2*a_2_3 + a_0_3*a_2_2) + (a_0_1*a_2_2 - a_0_2*a_2_1)*(a_0_0*a_1_2*a_2_3 - a_0_0*a_1_3*a_2_2 - a_0_2*a_1_0*a_2_3 + a_0_2*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_2 - a_0_3*a_1_2*a_2_0))/det**2, (det*(a_0_1*a_2_3 - a_0_3*a_2_1) - (a_0_1*a_2_2 - a_0_2*a_2_1)*(a_0_0*a_1_1*a_2_3 - a_0_0*a_1_3*a_2_1 - a_0_1*a_1_0*a_2_3 + a_0_1*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_1 - a_0_3*a_1_1*a_2_0))/det**2],
                [(a_2_2*det - (a_0_0*a_2_2 - a_0_2*a_2_0)*(a_1_1*a_2_2 - a_1_2*a_2_1))/det**2, (a_0_0*a_2_2 - a_0_2*a_2_0)*(a_0_1*a_2_2 - a_0_2*a_2_1)/det**2, -(a_0_2*det + (a_0_0*a_2_2 - a_0_2*a_2_0)*(a_0_1*a_1_2 - a_0_2*a_1_1))/det**2, (a_0_0*a_2_2 - a_0_2*a_2_0)*(a_1_0*a_2_2 - a_1_2*a_2_0)/det**2, -(a_0_0*a_2_2 - a_0_2*a_2_0)**2/det**2, (a_0_0*a_1_2 - a_0_2*a_1_0)*(a_0_0*a_2_2 - a_0_2*a_2_0)/det**2, -(a_2_0*det + (a_0_0*a_2_2 - a_0_2*a_2_0)*(a_1_0*a_2_1 - a_1_1*a_2_0))/det**2, (a_0_0*a_2_1 - a_0_1*a_2_0)*(a_0_0*a_2_2 - a_0_2*a_2_0)/det**2, (a_0_0*det - (a_0_0*a_1_1 - a_0_1*a_1_0)*(a_0_0*a_2_2 - a_0_2*a_2_0))/det**2, (det*(a_0_2*a_2_3 - a_0_3*a_2_2) + (a_0_0*a_2_2 - a_0_2*a_2_0)*(a_0_1*a_1_2*a_2_3 - a_0_1*a_1_3*a_2_2 - a_0_2*a_1_1*a_2_3 + a_0_2*a_1_3*a_2_1 + a_0_3*a_1_1*a_2_2 - a_0_3*a_1_2*a_2_1))/det**2, -(a_0_0*a_2_2 - a_0_2*a_2_0)*(a_0_0*a_1_2*a_2_3 - a_0_0*a_1_3*a_2_2 - a_0_2*a_1_0*a_2_3 + a_0_2*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_2 - a_0_3*a_1_2*a_2_0)/det**2, (det*(-a_0_0*a_2_3 + a_0_3*a_2_0) + (a_0_0*a_2_2 - a_0_2*a_2_0)*(a_0_0*a_1_1*a_2_3 - a_0_0*a_1_3*a_2_1 - a_0_1*a_1_0*a_2_3 + a_0_1*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_1 - a_0_3*a_1_1*a_2_0))/det**2],
                [(-a_2_1*det + (a_0_0*a_2_1 - a_0_1*a_2_0)*(a_1_1*a_2_2 - a_1_2*a_2_1))/det**2, -(a_0_0*a_2_1 - a_0_1*a_2_0)*(a_0_1*a_2_2 - a_0_2*a_2_1)/det**2, (a_0_1*det + (a_0_0*a_2_1 - a_0_1*a_2_0)*(a_0_1*a_1_2 - a_0_2*a_1_1))/det**2, (a_2_0*det - (a_0_0*a_2_1 - a_0_1*a_2_0)*(a_1_0*a_2_2 - a_1_2*a_2_0))/det**2, (a_0_0*a_2_1 - a_0_1*a_2_0)*(a_0_0*a_2_2 - a_0_2*a_2_0)/det**2, -(a_0_0*det + (a_0_0*a_1_2 - a_0_2*a_1_0)*(a_0_0*a_2_1 - a_0_1*a_2_0))/det**2, (a_0_0*a_2_1 - a_0_1*a_2_0)*(a_1_0*a_2_1 - a_1_1*a_2_0)/det**2, -(a_0_0*a_2_1 - a_0_1*a_2_0)**2/det**2, (a_0_0*a_1_1 - a_0_1*a_1_0)*(a_0_0*a_2_1 - a_0_1*a_2_0)/det**2, (det*(-a_0_1*a_2_3 + a_0_3*a_2_1) - (a_0_0*a_2_1 - a_0_1*a_2_0)*(a_0_1*a_1_2*a_2_3 - a_0_1*a_1_3*a_2_2 - a_0_2*a_1_1*a_2_3 + a_0_2*a_1_3*a_2_1 + a_0_3*a_1_1*a_2_2 - a_0_3*a_1_2*a_2_1))/det**2, (det*(a_0_0*a_2_3 - a_0_3*a_2_0) + (a_0_0*a_2_1 - a_0_1*a_2_0)*(a_0_0*a_1_2*a_2_3 - a_0_0*a_1_3*a_2_2 - a_0_2*a_1_0*a_2_3 + a_0_2*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_2 - a_0_3*a_1_2*a_2_0))/det**2, -(a_0_0*a_2_1 - a_0_1*a_2_0)*(a_0_0*a_1_1*a_2_3 - a_0_0*a_1_3*a_2_1 - a_0_1*a_1_0*a_2_3 + a_0_1*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_1 - a_0_3*a_1_1*a_2_0)/det**2],
                [-(a_0_1*a_1_2 - a_0_2*a_1_1)*(a_1_1*a_2_2 - a_1_2*a_2_1)/det**2, (a_0_1*a_1_2 - a_0_2*a_1_1)*(a_0_1*a_2_2 - a_0_2*a_2_1)/det**2, -(a_0_1*a_1_2 - a_0_2*a_1_1)**2/det**2, (a_1_2*det + (a_0_1*a_1_2 - a_0_2*a_1_1)*(a_1_0*a_2_2 - a_1_2*a_2_0))/det**2, -(a_0_2*det + (a_0_0*a_2_2 - a_0_2*a_2_0)*(a_0_1*a_1_2 - a_0_2*a_1_1))/det**2, (a_0_0*a_1_2 - a_0_2*a_1_0)*(a_0_1*a_1_2 - a_0_2*a_1_1)/det**2, -(a_1_1*det + (a_0_1*a_1_2 - a_0_2*a_1_1)*(a_1_0*a_2_1 - a_1_1*a_2_0))/det**2, (a_0_1*det + (a_0_0*a_2_1 - a_0_1*a_2_0)*(a_0_1*a_1_2 - a_0_2*a_1_1))/det**2, -(a_0_0*a_1_1 - a_0_1*a_1_0)*(a_0_1*a_1_2 - a_0_2*a_1_1)/det**2, (a_0_1*a_1_2 - a_0_2*a_1_1)*(a_0_1*a_1_2*a_2_3 - a_0_1*a_1_3*a_2_2 - a_0_2*a_1_1*a_2_3 + a_0_2*a_1_3*a_2_1 + a_0_3*a_1_1*a_2_2 - a_0_3*a_1_2*a_2_1)/det**2, (det*(a_0_2*a_1_3 - a_0_3*a_1_2) - (a_0_1*a_1_2 - a_0_2*a_1_1)*(a_0_0*a_1_2*a_2_3 - a_0_0*a_1_3*a_2_2 - a_0_2*a_1_0*a_2_3 + a_0_2*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_2 - a_0_3*a_1_2*a_2_0))/det**2, (det*(-a_0_1*a_1_3 + a_0_3*a_1_1) + (a_0_1*a_1_2 - a_0_2*a_1_1)*(a_0_0*a_1_1*a_2_3 - a_0_0*a_1_3*a_2_1 - a_0_1*a_1_0*a_2_3 + a_0_1*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_1 - a_0_3*a_1_1*a_2_0))/det**2],
                [(-a_1_2*det + (a_0_0*a_1_2 - a_0_2*a_1_0)*(a_1_1*a_2_2 - a_1_2*a_2_1))/det**2, (a_0_2*det - (a_0_0*a_1_2 - a_0_2*a_1_0)*(a_0_1*a_2_2 - a_0_2*a_2_1))/det**2, (a_0_0*a_1_2 - a_0_2*a_1_0)*(a_0_1*a_1_2 - a_0_2*a_1_1)/det**2, -(a_0_0*a_1_2 - a_0_2*a_1_0)*(a_1_0*a_2_2 - a_1_2*a_2_0)/det**2, (a_0_0*a_1_2 - a_0_2*a_1_0)*(a_0_0*a_2_2 - a_0_2*a_2_0)/det**2, -(a_0_0*a_1_2 - a_0_2*a_1_0)**2/det**2, (a_1_0*det + (a_0_0*a_1_2 - a_0_2*a_1_0)*(a_1_0*a_2_1 - a_1_1*a_2_0))/det**2, -(a_0_0*det + (a_0_0*a_1_2 - a_0_2*a_1_0)*(a_0_0*a_2_1 - a_0_1*a_2_0))/det**2, (a_0_0*a_1_1 - a_0_1*a_1_0)*(a_0_0*a_1_2 - a_0_2*a_1_0)/det**2, (det*(-a_0_2*a_1_3 + a_0_3*a_1_2) - (a_0_0*a_1_2 - a_0_2*a_1_0)*(a_0_1*a_1_2*a_2_3 - a_0_1*a_1_3*a_2_2 - a_0_2*a_1_1*a_2_3 + a_0_2*a_1_3*a_2_1 + a_0_3*a_1_1*a_2_2 - a_0_3*a_1_2*a_2_1))/det**2, (a_0_0*a_1_2 - a_0_2*a_1_0)*(a_0_0*a_1_2*a_2_3 - a_0_0*a_1_3*a_2_2 - a_0_2*a_1_0*a_2_3 + a_0_2*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_2 - a_0_3*a_1_2*a_2_0)/det**2, (det*(a_0_0*a_1_3 - a_0_3*a_1_0) - (a_0_0*a_1_2 - a_0_2*a_1_0)*(a_0_0*a_1_1*a_2_3 - a_0_0*a_1_3*a_2_1 - a_0_1*a_1_0*a_2_3 + a_0_1*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_1 - a_0_3*a_1_1*a_2_0))/det**2],
                [(a_1_1*det - (a_0_0*a_1_1 - a_0_1*a_1_0)*(a_1_1*a_2_2 - a_1_2*a_2_1))/det**2, (-a_0_1*det + (a_0_0*a_1_1 - a_0_1*a_1_0)*(a_0_1*a_2_2 - a_0_2*a_2_1))/det**2, -(a_0_0*a_1_1 - a_0_1*a_1_0)*(a_0_1*a_1_2 - a_0_2*a_1_1)/det**2, (-a_1_0*det + (a_0_0*a_1_1 - a_0_1*a_1_0)*(a_1_0*a_2_2 - a_1_2*a_2_0))/det**2, (a_0_0*det - (a_0_0*a_1_1 - a_0_1*a_1_0)*(a_0_0*a_2_2 - a_0_2*a_2_0))/det**2, (a_0_0*a_1_1 - a_0_1*a_1_0)*(a_0_0*a_1_2 - a_0_2*a_1_0)/det**2, -(a_0_0*a_1_1 - a_0_1*a_1_0)*(a_1_0*a_2_1 - a_1_1*a_2_0)/det**2, (a_0_0*a_1_1 - a_0_1*a_1_0)*(a_0_0*a_2_1 - a_0_1*a_2_0)/det**2, -(a_0_0*a_1_1 - a_0_1*a_1_0)**2/det**2, (det*(a_0_1*a_1_3 - a_0_3*a_1_1) + (a_0_0*a_1_1 - a_0_1*a_1_0)*(a_0_1*a_1_2*a_2_3 - a_0_1*a_1_3*a_2_2 - a_0_2*a_1_1*a_2_3 + a_0_2*a_1_3*a_2_1 + a_0_3*a_1_1*a_2_2 - a_0_3*a_1_2*a_2_1))/det**2, (det*(-a_0_0*a_1_3 + a_0_3*a_1_0) - (a_0_0*a_1_1 - a_0_1*a_1_0)*(a_0_0*a_1_2*a_2_3 - a_0_0*a_1_3*a_2_2 - a_0_2*a_1_0*a_2_3 + a_0_2*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_2 - a_0_3*a_1_2*a_2_0))/det**2, (a_0_0*a_1_1 - a_0_1*a_1_0)*(a_0_0*a_1_1*a_2_3 - a_0_0*a_1_3*a_2_1 - a_0_1*a_1_0*a_2_3 + a_0_1*a_1_3*a_2_0 + a_0_3*a_1_0*a_2_1 - a_0_3*a_1_1*a_2_0)/det**2],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, (-a_1_1*a_2_2 + a_1_2*a_2_1)/det, (a_1_0*a_2_2 - a_1_2*a_2_0)/det, (-a_1_0*a_2_1 + a_1_1*a_2_0)/det],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, (a_0_1*a_2_2 - a_0_2*a_2_1)/det, (-a_0_0*a_2_2 + a_0_2*a_2_0)/det, (a_0_0*a_2_1 - a_0_1*a_2_0)/det],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, (-a_0_1*a_1_2 + a_0_2*a_1_1)/det, (a_0_0*a_1_2 - a_0_2*a_1_0)/det, (-a_0_0*a_1_1 + a_0_1*a_1_0)/det]
            ])
    
    def itk_transform_string_rec(self, index):
        s = '#Transform %d\n' % index        
        s = s + 'Transform: AffineTransform_double_%d_%d\n' % (self.get_dim(), self.get_dim())
        s = s + 'Parameters:'
        for i in range(self.get_param_count()):
            s = s + (' %f' % self.get_param(i))
        s = s + '\n'
        s = s + 'FixedParameters:'
        for i in range(self.get_dim()):
            s = s + ' 0.0'
        s = s + '\n'

        return s 

###
### CompositeTransform
###

class CompositeTransform(TransformBase):
    def __init__(self, dim, transforms, active_flags = None):
        self.dim = dim

        if active_flags is None:
            active_flags = np.ones(len(transforms), dtype='bool')
        
        self.active_flags = active_flags

        self.transforms = []
        
        cnt = 0
        for i in range(len(transforms)):
            t = transforms[i]

            if active_flags[i] == True:
                cnt = cnt + t.get_param_count()

            self.transforms.append(t.copy())

        self.param_count = cnt

    def get_transforms(self):
        return self.transforms
    
    def get_dim(self):
        return self.dim

    def get_params(self):
        res = np.zeros((self.param_count,))
        ind = 0
        for i, t in enumerate(self.transforms):
            if self.active_flags[i] == True:
                cnt = t.get_param_count()
                res[ind:ind + cnt] = t.get_params()
                ind = ind + cnt
        return res
    
    def set_params(self, params):
        ind = 0
        for i, t in enumerate(self.transforms):
            if self.active_flags[i] == True:
                cnt = t.get_param_count()
                t.set_params(params[ind:ind+cnt])
                ind = ind + cnt

    def get_param(self, index):
        assert(index >= 0)
        assert(index < self.param_count)

        for i, t in enumerate(self.transforms):
            if self.active_flags[i] == True:
                cnt = t.get_param_count()
                if index < cnt:
                    return t.get_param(index)
                else:
                    index = index - cnt
    
    def set_param(self, index, value):
        assert(index >= 0)
        assert(index < self.param_count)
        
        for i, t in enumerate(self.transforms):
            if self.active_flags[i] == True:
                cnt = t.get_param_count()
                if index < cnt:
                    t.set_param(index, value)
                    return
                else:
                    index = index - cnt

    def set_params_const(self, value):
        for i, t in enumerate(self.transforms):
            if self.active_flags[i] == True:
                t.set_params_conts(value)

    def step_param(self, index, step_length):
        self.set_param(index, self.get_param(index) + step_length)

    def step_params(self, grad, step_length):
        params = self.get_params()
        params = params + grad * step_length
        self.set_params(params)

    def get_param_count(self):
        return self.param_count

    def copy_child(self):
        return CompositeTransform(self.get_dim(), self.transforms, self.active_flags)

    def copy(self):
        return self.copy_child()

    def transform(self, pnts):
        self.input_pnts = []

        p = pnts
        for i, t in enumerate(self.transforms):
            self.input_pnts.append(p)
            p = t.transform(p)
        
        #self.input_pnts.append(p)
        return p

    def grad(self, pnts, gradients, output_gradients):
        res = np.zeros((self.param_count,))
        ind = self.param_count
        p = pnts
        gr = gradients
        tlen = len(self.transforms)
        for i, t in enumerate(reversed(self.transforms)):
            t_index = tlen - i - 1
            #print("Input points: " + str(self.input_pnts[i]))
            #print("Gr: " + str(gr))
            if output_gradients == True or i < tlen-1:
                g, gr = t.grad(self.input_pnts[t_index], gr, True)
            else:
                g = t.grad(self.input_pnts[t_index], gr, False)
            
            if self.active_flags[t_index] == True:
                cnt = t.get_param_count()
                res[ind-cnt:ind] = g
                ind = ind - cnt
        
        if output_gradients == True:
            return res, gr
        else:
            return res

    def invert(self):
        inv_transforms = []

        tlen = len(self.transforms)
        for i in range(tlen):
            inv_transforms.append(self.transforms[(tlen-1)-i].invert())

        return CompositeTransform(self.get_dim(), inv_transforms, np.flip(self.active_flags, 0))

    def inverse_to_forward_matrix(self):
        pcnt = self.get_param_count()
        res = np.zeros((pcnt, pcnt))
        find = 0
        rind = pcnt
        for i, t in enumerate(self.transforms):
            if self.active_flags[i] == True:
                cnt = t.get_param_count()
                rind = rind - cnt
                mat = t.inverse_to_forward_matrix()
                res[find:find+cnt, rind:rind+cnt] = mat
                find = find + cnt
        return res
    '''def inverse_to_forward_matrix(self):
        pcnt = self.get_param_count()
        res = np.zeros((pcnt, pcnt))
        ind = 0
        for i, t in enumerate(self.transforms):
            if self.active_flags[i] == True:
                cnt = t.get_param_count()
                mat = t.inverse_to_forward_matrix()
                res[ind:ind+cnt, ind:ind+cnt] = mat
                ind = ind + cnt
        return res
    '''

    def itk_transform_string_rec(self, index):
        s = '#Transform %d\n' % index        
        s = s + 'Transform: CompositeTransform_double_%d_%d\n' % (self.get_dim(), self.get_dim())
        for i in range(len(self.transforms)):
            index = index + 1
            s = s + self.transforms[i].itk_transform_string_rec(index)

        return s
        
    #def grad_inverse_to_forward(self, inv_grad):
    #    pcnt = self.get_param_count()
    #    res = np.zeros((pcnt,))
    #    ind = 0
    #    rev_ind = pcnt
    #    for t in self.transforms:
    #        cnt = t.get_param_count()
    #        inv_grad_t = inv_grad[rev_ind-cnt:rev_ind]
    #        res[ind:ind+cnt] = t.grad_inverse_to_forward(inv_grad_t)
    #        ind = ind + cnt
    #        rev_ind = rev_ind - cnt
    #    return res

###
### Utility functions
###

def image_center_point(image, spacing = None):
    shape = image.shape
    if spacing is None:
        return (np.array(shape)-1) * 0.5
    else:
        return ((np.array(shape)-1) * spacing) * 0.5

def image_diagonal(image, spacing = None):
    shp = np.array(image.shape)-1
    if spacing is not None:
        shp = shp * spacing
    return np.sqrt(np.sum(np.square(shp)))

def make_centered_transform(t, cp1, cp2):
    dim = t.get_dim()
    t1 = TranslationTransform(dim)
    t2 = TranslationTransform(dim)
    t1.set_params(-cp1)
    t2.set_params(cp2)
    return CompositeTransform(dim, [t1, t, t2], [False, True, False])

def make_image_centered_transform(t, image1, image2, image1_spacing = None, image2_spacing = None):
    dim = image1.ndim
    t1 = TranslationTransform(dim)
    t2 = TranslationTransform(dim)
    t1.set_params(-image_center_point(image1, image1_spacing))
    t2.set_params(image_center_point(image2, image2_spacing))
    return CompositeTransform(dim, [t1, t, t2], [False, True, False])
