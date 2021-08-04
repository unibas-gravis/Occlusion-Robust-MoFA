import torch
import torch.nn as nn
import torch.nn.functional as F
import renderer.geometry as geo
import renderer.shading as sh
import renderer.rasterize as ras


SAMPLE_IMAGE = 0
RASTERIZE_IMAGE = 1
RASTERIZE_DIFFERENTIABLE_IMAGE = 2

def resample(image, pt):
    """

    :rtype: object
    """
    batch, _, pnum = pt.shape
    _,ch,height,width = image.shape

    pt_prj = pt[:,0:2,:]+0.0

    half_max_x = (float(image.shape[3])-1.0)/2.0
    half_max_y = (float(image.shape[2])-1.0)/2.0

    smp = torch.transpose(pt_prj,1,2)
    smp = smp.reshape(batch,1,pnum,2)

    smp[:,:,:,0] = (smp[:,:,:,0]-half_max_x)/half_max_x
    smp[:,:,:,1] = (smp[:,:,:,1]-half_max_y)/half_max_y

    res = F.grid_sample(image,smp)
    res = torch.transpose(res,2,3)

    return res.reshape(batch,ch,pnum)


class Renderer(nn.Module):

    def __init__(self,block_size = 32):

        super(Renderer, self).__init__()

        self.block_size = block_size
        self.ocl_net = None
        self.raster_net = None
        self.diff_raster_net = None
        self.vertex_to_face = None

    #face: [3,face num] vertec index corresponding to each face
    #vertex: [batch,3,vertex num] vertex position
    #color: [batch,3,vertex num] vertex color
    #sh_coef: [batch,27] spherical harmonics coefficient
    #A: [batch,3,3] intrinsic camera matrix
    #R: [batch,3,3] R from extrinsic camera matrix [R|T]
    #T: [batch,3,1] T from extrinsic camera matrix [R|T]
    #image: (SAMPLE_IMAGE mode)input image for sampling, (RASTER_IMAGE mode)only image size is used
    #render_mode=SMAPLE_IMAGE: calculate vertex color and sample pixel values from image for each vertex(for MoFA)
    #render_mode=RASTER_IMAGE: render image
    #render_mode=RASTERIZE_DIFFERENTIABLE_IMAGE: render differentiable image
    #occlusion_th: used for self-occlusion detection in SMAPLE_IMAGE mode
    #ratete_normal: if True, lighting is applied in camera coordinate
    def forward(self, face, vertex, color, sh_coef, A, R, T, image, render_mode=SAMPLE_IMAGE, self_occlusion=False, occlusion_th=5, rotate_normal=True):

            batch,ch,height,width = image.shape

            #initialize
            if self.vertex_to_face is None:
                self.vertex_to_face = geo.calc_vertex_to_face_map(vertex, face)
            if self.ocl_net is None and render_mode==SAMPLE_IMAGE and self_occlusion:
                self.ocl_net = ras.OcclusionDetector(face, height, width,vertex.shape[2], self.block_size)
            if self.raster_net is None and render_mode==RASTERIZE_IMAGE:
                self.raster_net = ras.Rasterizer(face,height,width,self.block_size)
            if self.diff_raster_net is None and render_mode==RASTERIZE_DIFFERENTIABLE_IMAGE:
                self.diff_raster_net = ras.DifferentiableRasterizer(face,height,width,self.block_size)

            #cal normal
            face_normal, face_area = geo.calc_face_normal(vertex,face)
            vertex_normal = geo.calc_vertex_normal(face_normal,face_area,self.vertex_to_face)

            #projection
            projected_vertex = geo.perspective_projection(A,R,T,vertex)
            rotated_vertex_normal = R@vertex_normal

            if rotate_normal:
                shade_vertex_normal = rotated_vertex_normal
            else:
                shade_vertex_normal = vertex_normal

            #shading
            sh_basis = sh.calc_sh_basis(shade_vertex_normal)
            illumination = torch.sum(sh_basis*(sh_coef.unsqueeze(2).unsqueeze(2)),dim=1)
            shaded_color = illumination*color

            #Rendering
            occlusion = None
            sampled_color = None
            raster_image = None
            raster_mask = None

            if render_mode==SAMPLE_IMAGE:
                sampled_color = resample(image,projected_vertex)
                if self_occlusion:
                    occlusion = self.ocl_net(projected_vertex, projected_vertex, vertex, face_normal, R, T)
                else:
                    occlusion = (rotated_vertex_normal[:,2,:]<0).float()

            elif render_mode==RASTERIZE_IMAGE:
                raster_image,raster_mask = self.raster_net(projected_vertex, shaded_color, vertex, face_normal, R, T)

            elif render_mode==RASTERIZE_DIFFERENTIABLE_IMAGE:
                raster_image,raster_mask = self.diff_raster_net(projected_vertex, shaded_color, vertex, face_normal, R, T)

            return projected_vertex, sampled_color, shaded_color, occlusion,raster_image, raster_mask
