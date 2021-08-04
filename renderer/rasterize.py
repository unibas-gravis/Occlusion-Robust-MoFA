import torch
import torch.nn as nn
import math

ftiny = torch.finfo(torch.float).tiny * 10**3
inf_value = torch.finfo(torch.float).max * 10**-3
lower_inf_value = torch.finfo(torch.float).max * 10**-4

class Rasterizer(nn.Module):

    def __init__(self,face,height,width,block_size=32):

        super(Rasterizer, self).__init__()

        self.block_size = block_size
        self.width = width
        self.height = height
        self.width_exp = int(math.ceil(float(width)/float(self.block_size)))*self.block_size
        self.height_exp = int(math.ceil(float(height)/float(self.block_size)))*self.block_size

        self.face = face
        self.index_buf = torch.full((self.height_exp, self.width_exp), face.shape[1], dtype=torch.long).to(face.device)
        self.face_index = torch.LongTensor(range(0,self.face.shape[1])).to(face.device)

        self.x_grid = torch.Tensor(range(0,self.width_exp)).unsqueeze(0).to(face.device)
        self.y_grid = torch.Tensor(range(0,self.height_exp)).unsqueeze(1).to(face.device)
        self.x_grid_block = torch.Tensor(range(0,self.block_size)).unsqueeze(0).unsqueeze(2).to(face.device)
        self.y_grid_block = torch.Tensor(range(0,self.block_size)).unsqueeze(1).unsqueeze(2).to(face.device)

    def forward(self,pt_2d,color,pt_3d,normal,R,T):
        with torch.no_grad():
            batch, vnum, pnum = pt_2d.shape

            image = torch.zeros(batch, color.shape[1], self.height_exp, self.width_exp,device=pt_2d.device)
            mask = torch.zeros(batch, self.height_exp, self.width_exp,device=pt_2d.device)

            for b in range(batch):
                #removing invisible faces
                norm_cul = torch.sum((pt_3d[b,:,self.face[0, :]] + (R[b,:,:].t()@T[b,:, :])) * normal[b,:,:],0) < 0
                depth_cul = torch.min(pt_2d[b,2,self.face], 0)[0] > 0

                if torch.sum(norm_cul * depth_cul).item()==0:
                    continue

                face_red = self.face[:, norm_cul * depth_cul]
                num = face_red.shape[1]

                self.index_buf[:] = num

                p = pt_2d[b, :, face_red]

                #limitting rastrization region
                pz_min,_ = torch.min(p[2,:,:],0)
                px_min,_ = torch.min(p[0,:,:].int(),0)
                px_max,_ = torch.max(p[0,:,:].int(),0)
                py_min,_= torch.min(p[1, :, :].int(), 0)
                py_max,_ = torch.max(p[1,:,:].int(),0)
                x_min,_ = torch.min(px_min,0)
                x_max,_ = torch.max(px_max,0)
                y_min,_ = torch.min(py_min,0)
                y_max,_ = torch.max(py_max,0)

                range_x_min = max(x_min.item()-x_min.item()%self.block_size,0)
                range_y_min = max(y_min.item() - y_min.item() % self.block_size, 0)
                range_x_max = min(x_max.item(), self.width_exp)
                range_y_max = min(y_max.item(), self.height_exp)

                #precompute barycentric coordinate for all faces as a form of coefficient of x,y,1
                det = ((p[1, 1, :] - p[1, 2, :]) * (p[0, 0, :] - p[0, 2, :]) + (p[0, 2, :] - p[0, 1, :]) * (p[1, 0, :] - p[1, 2, :])).unsqueeze(0).unsqueeze(0)
                det = det.sign()*torch.clamp(det.abs(),min=ftiny)
                inv_det = 1/det

                l0_x = (p[1, 1, :] - p[1, 2, :]) * inv_det
                l0_y = (p[0, 2, :] - p[0, 1, :]) * inv_det
                l0_c = -l0_x*p[0, 2, :] - l0_y *p[1, 2, :]

                l1_x = (p[1, 2, :] - p[1, 0, :]) * inv_det
                l1_y = (p[0, 0, :] - p[0, 2, :]) * inv_det
                l1_c = -l1_x*p[0, 2, :] - l1_y *p[1, 2, :]

                l2_x = -l0_x - l1_x
                l2_y = -l0_y - l1_y
                l2_c = 1-l0_c-l1_c

                #precompute barycentric combination of vertex color
                c = color[b, :, face_red]
                c = c.unsqueeze(1).unsqueeze(1)
                C_x = (c[:,:,:,0,:]*l0_x + c[:,:,:,1,:]*l1_x + c[:,:,:,2,:]*l2_x).squeeze(1).squeeze(1)
                C_y = (c[:,:,:,0,:]*l0_y + c[:,:,:,1,:]*l1_y + c[:,:,:,2,:]*l2_y).squeeze(1).squeeze(1)
                C_c = (c[:,:,:,0,:]*l0_c + c[:,:,:,1,:]*l1_c + c[:,:,:,2,:]*l2_c).squeeze(1).squeeze(1)

                #precompute barycentric combination of vertex depth
                p = p.unsqueeze(1).unsqueeze(1)
                D_x = p[2, :, :, 0, :] * l0_x + p[2, :, :, 1, :] * l1_x + p[2, :, :, 2, :] * l2_x
                D_y = p[2, :, :, 0, :] * l0_y + p[2, :, :, 1, :] * l1_y + p[2, :, :, 2, :] * l2_y
                D_c = (p[2,:,:,0,:]*l0_c + p[2,:,:,1,:]*l1_c + p[2,:,:,2,:]*l2_c)

                #block-wise computation
                for i in range(range_y_min,range_y_max,self.block_size):

                    #apply linear combination of precomputed barycentric coordinate
                    D_yc = D_y * (float(i)+self.y_grid_block) + D_c
                    l0_yc = l0_y * (float(i) + self.y_grid_block) + l0_c
                    l1_yc = l1_y * (float(i) + self.y_grid_block) + l1_c
                    l2_yc = l2_y * (float(i) + self.y_grid_block) + l2_c

                    for k in range(range_x_min,range_x_max,self.block_size):

                        #detecting faces inside the block
                        target = (px_max>=k)*(px_min<k+self.block_size)*(py_max>=i)*(py_min<i+self.block_size)

                        #if no face remains, skip following processes
                        face_ct = torch.sum(target, 0, dtype=torch.long)
                        if face_ct == 0:
                            continue

                        # calculate barycentric coordinate and detect inner pixels of triangles among all pixels inside the block
                        kxg = float(-k)-self.x_grid_block
                        M = (l0_yc[:,:,target]  >= (l0_x[:,:,target]* kxg)) * (l1_yc[:,:,target]  >= (l1_x[:,:,target]* kxg)) * (l2_yc[:,:,target] >= (l2_x[:,:,target]* kxg))

                        #z buffer processing
                        vis_ct = torch.max(torch.sum(M, 2)).item()
                        if vis_ct==1:   #if there is only one inner pixels, store the index of the face
                            vis, idx = torch.max(M, 2)
                            self.index_buf[i:i + self.block_size, k:k + self.block_size][vis] = (self.face_index[0:target.shape[0]][target])[idx[vis]]
                        elif vis_ct>1:  #if there are multiple inner pixels, calculate and store the index of the foreground face
                            D = M.bitwise_not().float()  * inf_value + D_x[:,:,target] * (float(k)+self.x_grid_block) + D_yc[:,:,target]
                            D[D!=D]=inf_value  #NaN correction
                            depth, idx = torch.min(D,2) #process z-buffer by argmin
                            vis = depth< lower_inf_value

                            self.index_buf[i:i+self.block_size, k:k+self.block_size][vis] = (self.face_index[0:target.shape[0]][target])[idx[vis]]

                index_buf_tmp = self.index_buf+0
                index_buf_tmp[self.index_buf==num] = 0

                #calculate and store pixel value based on precomputed barycentric coeffients and detected pixel-face correspondence
                image[b,:,self.index_buf!=num] = (C_x[:,index_buf_tmp]*self.x_grid + C_y[:,index_buf_tmp]*self.y_grid  + C_c[:,index_buf_tmp])[:,self.index_buf!=num]
                mask[b,self.index_buf!=num] = 1

        return image[:,:,0:self.height,0:self.width], mask[:,0:self.height,0:self.width]


class OcclusionDetector(nn.Module):

    def __init__(self,face,height,width,pnum,block_size=32):

        super(OcclusionDetector, self).__init__()

        self.block_size = block_size
        self.width = width
        self.height = height
        self.width_exp = int(math.ceil(float(width)/float(self.block_size)))*self.block_size
        self.height_exp = int(math.ceil(float(height)/float(self.block_size)))*self.block_size

        self.face = face
        self.depth_map = torch.Tensor(self.height_exp,self.width_exp).to(face.device)
        self.index_buf = torch.full((self.height_exp, self.width_exp), face.shape[1], dtype=torch.long).to(face.device)
        self.face_index = torch.LongTensor(range(0,self.face.shape[1])).to(face.device)
        self.pt_index = torch.LongTensor(range(0,pnum)).to(face.device)
        self.x_grid = torch.Tensor(range(0,self.width_exp)).unsqueeze(0).to(face.device)
        self.y_grid = torch.Tensor(range(0,self.height_exp)).unsqueeze(1).to(face.device)
        self.x_grid_block = torch.Tensor(range(0,self.block_size)).unsqueeze(0).unsqueeze(2).to(face.device)
        self.y_grid_block = torch.Tensor(range(0,self.block_size)).unsqueeze(1).unsqueeze(2).to(face.device)

    def forward(self,pt_tar, pt_2d,pt_3d,normal,R,T,occlusion_th=5):
        with torch.no_grad():
            batch, vnum, pnum = pt_2d.shape

            occlusion = torch.zeros(batch, pnum, dtype=torch.float,device=pt_2d.device)

            for b in range(batch):

                norm_cul = torch.sum((pt_3d[b,:,self.face[0, :]] + (R[b,:,:].t()@T[b,:, :])) * normal[b,:,:],0) < 0
                depth_cul = torch.min(pt_2d[b,2,self.face], 0)[0] > 0

                if torch.sum(norm_cul * depth_cul).item()==0:
                    continue

                face_red = self.face[:, norm_cul * depth_cul]
                num = face_red.shape[1]

                self.index_buf[:] = num

                p = pt_2d[b, :, face_red]

                pz_min,_ = torch.min(p[2,:,:],0)
                px_min,_ = torch.min(p[0,:,:].int(),0)
                px_max,_ = torch.max(p[0,:,:].int(),0)
                py_min,_= torch.min(p[1, :, :].int(), 0)
                py_max,_ = torch.max(p[1,:,:].int(),0)
                x_min,_ = torch.min(px_min,0)
                x_max,_ = torch.max(px_max,0)
                y_min,_ = torch.min(py_min,0)
                y_max,_ = torch.max(py_max,0)

                range_x_min = max(x_min.item()-x_min.item()%self.block_size,0)
                range_y_min = max(y_min.item() - y_min.item() % self.block_size, 0)
                range_x_max = min(x_max.item(), self.width_exp)
                range_y_max = min(y_max.item(), self.height_exp)

                det = ((p[1, 1, :] - p[1, 2, :]) * (p[0, 0, :] - p[0, 2, :]) + (p[0, 2, :] - p[0, 1, :]) * (p[1, 0, :] - p[1, 2, :])).unsqueeze(0).unsqueeze(0)
                det = det.sign()*torch.clamp(det.abs(),min=ftiny)
                inv_det = 1/det

                l0_x = (p[1, 1, :] - p[1, 2, :]) * inv_det
                l0_y = (p[0, 2, :] - p[0, 1, :]) * inv_det
                l0_c = -l0_x*p[0, 2, :] - l0_y *p[1, 2, :]

                l1_x = (p[1, 2, :] - p[1, 0, :]) * inv_det
                l1_y = (p[0, 0, :] - p[0, 2, :]) * inv_det
                l1_c = -l1_x*p[0, 2, :] - l1_y *p[1, 2, :]

                l2_x = -l0_x - l1_x
                l2_y = -l0_y - l1_y
                l2_c = 1-l0_c-l1_c

                p = p.unsqueeze(1).unsqueeze(1)
                D_x = p[2, :, :, 0, :] * l0_x + p[2, :, :, 1, :] * l1_x + p[2, :, :, 2, :] * l2_x
                D_y = p[2, :, :, 0, :] * l0_y + p[2, :, :, 1, :] * l1_y + p[2, :, :, 2, :] * l2_y
                D_c = (p[2,:,:,0,:]*l0_c + p[2,:,:,1,:]*l1_c + p[2,:,:,2,:]*l2_c)

                for i in range(range_y_min,range_y_max,self.block_size):

                    D_yc = D_y * (float(i)+self.y_grid_block) + D_c
                    l0_yc = l0_y * (float(i) + self.y_grid_block) + l0_c
                    l1_yc = l1_y * (float(i) + self.y_grid_block) + l1_c
                    l2_yc = l2_y * (float(i) + self.y_grid_block) + l2_c

                    for k in range(range_x_min,range_x_max,self.block_size):

                        target = (px_max>=k)*(px_min<k+self.block_size)*(py_max>=i)*(py_min<i+self.block_size)

                        face_ct = torch.sum(target, 0, dtype=torch.long)
                        if face_ct == 0:
                            continue

                        kxg = float(-k)-self.x_grid_block

                        M = (l0_yc[:,:,target]  >= (l0_x[:,:,target]* kxg)) * (l1_yc[:,:,target]  >= (l1_x[:,:,target]* kxg)) * (l2_yc[:,:,target] >= (l2_x[:,:,target]* kxg))

                        vis_ct = torch.max(torch.sum(M, 2)).item()

                        if vis_ct==1:
                            vis, idx = torch.max(M, 2)
                            self.index_buf[i:i + self.block_size, k:k + self.block_size][vis] = (self.face_index[0:target.shape[0]][target])[idx[vis]]

                        elif vis_ct>1:
                            D = M.bitwise_not().float() * inf_value + D_x[:,:,target] * (float(k)+self.x_grid_block) + D_yc[:,:,target]
                            D[D!=D]=inf_value
                            depth, idx = torch.min(D,2)
                            vis = depth< lower_inf_value

                            self.index_buf[i:i+self.block_size, k:k+self.block_size][vis] = (self.face_index[0:target.shape[0]][target])[idx[vis]]

                index_buf_tmp = self.index_buf+0
                index_buf_tmp[self.index_buf==num] = 0

                self.depth_map[ :,:] = inf_value
                self.depth_map[self.index_buf!=num] = (D_x[0,0,index_buf_tmp]*self.x_grid + D_y[0,0,index_buf_tmp]*self.y_grid  + D_c[0,0,index_buf_tmp])[self.index_buf!=num]

                x0 = torch.floor(pt_tar[b,0,:]).long()
                x1 = torch.ceil(pt_tar[b, 0, :]).long()
                y0 = torch.floor(pt_tar[b,1,:]).long()
                y1 = torch.ceil(pt_tar[b, 1, :]).long()
                z = pt_tar[b, 2, :]

                valid = (x0>=0)*(x1<=self.width-1)*(y0>=0)*(y1<=self.height-1)

                pt_valid = self.pt_index[valid]
                z_valid = z[valid]

                ocl0 = (torch.abs(z_valid-self.depth_map[y0[pt_valid],x0[pt_valid]])<occlusion_th)
                ocl1 = (torch.abs(z_valid-self.depth_map[y0[pt_valid],x1[pt_valid]])<occlusion_th)
                ocl2 = (torch.abs(z_valid-self.depth_map[y1[pt_valid],x0[pt_valid]])<occlusion_th)
                ocl3 = (torch.abs(z_valid-self.depth_map[y1[pt_valid],x1[pt_valid]])<occlusion_th)

                occlusion[b,pt_valid[ocl0*ocl1*ocl2*ocl3]] = 1

        return occlusion

class DifferentiableRasterizer(nn.Module):

    def __init__(self,face,height,width,block_size=32):

        super(DifferentiableRasterizer, self).__init__()

        self.block_size = block_size
        self.width = width
        self.height = height
        self.width_exp = int(math.ceil(float(width)/float(self.block_size)))*self.block_size
        self.height_exp = int(math.ceil(float(height)/float(self.block_size)))*self.block_size
        self.face = face
        self.index_buf = torch.full((self.height_exp, self.width_exp), face.shape[1], dtype=torch.long).to(face.device)
        self.face_index = torch.LongTensor(range(0,self.face.shape[1])).to(face.device)

        self.x_grid = torch.Tensor(range(0,self.width)).unsqueeze(0).to(face.device)
        self.y_grid = torch.Tensor(range(0,self.height)).unsqueeze(1).to(face.device)
        self.x_grid_block = torch.Tensor(range(0,self.block_size)).unsqueeze(0).unsqueeze(2).to(face.device)
        self.y_grid_block = torch.Tensor(range(0,self.block_size)).unsqueeze(1).unsqueeze(2).to(face.device)

    def forward(self,pt_2d,color,pt_3d,normal,R,T):
        batch, vnum, pnum = pt_2d.shape
        cnum = color.shape[1]

        image = torch.zeros(batch,cnum,self.height,self.width,device=pt_2d.device)
        mask = torch.zeros(batch,self.height,self.width,device=pt_2d.device)

        for b in range(batch):
            with torch.no_grad():

                norm_cul = torch.sum((pt_3d[b,:,self.face[0, :]] + (R[b,:,:].t()@T[b,:, :])) * normal[b,:,:],0) < 0
                depth_cul = torch.min(pt_2d[b,2,self.face], 0)[0] > 0

                if torch.sum(norm_cul * depth_cul).item()==0:
                    continue

                face_red = self.face[:, norm_cul * depth_cul]
                face_index_red = self.face_index[norm_cul * depth_cul]
                num = face_red.shape[1]

                self.index_buf[:] = num

                p = pt_2d[b, :, face_red]

                pz_min,_ = torch.min(p[2,:,:],0)
                px_min,_ = torch.min(p[0,:,:].int(),0)
                px_max,_ = torch.max(p[0,:,:].int(),0)
                py_min,_= torch.min(p[1, :, :].int(), 0)
                py_max,_ = torch.max(p[1,:,:].int(),0)
                x_min,_ = torch.min(px_min,0)
                x_max,_ = torch.max(px_max,0)
                y_min,_ = torch.min(py_min,0)
                y_max,_ = torch.max(py_max,0)

                range_x_min = max(x_min.item()-x_min.item()%self.block_size,0)
                range_y_min = max(y_min.item() - y_min.item() % self.block_size, 0)
                range_x_max = min(x_max.item(), self.width_exp)
                range_y_max = min(y_max.item(), self.height_exp)

                det = ((p[1, 1, :] - p[1, 2, :]) * (p[0, 0, :] - p[0, 2, :]) + (p[0, 2, :] - p[0, 1, :]) * (p[1, 0, :] - p[1, 2, :])).unsqueeze(0).unsqueeze(0)
                det = det.sign()*torch.clamp(det.abs(),min=ftiny)
                inv_det = 1/det

                l0_x = (p[1, 1, :] - p[1, 2, :]) * inv_det
                l0_y = (p[0, 2, :] - p[0, 1, :]) * inv_det
                l0_c = -l0_x*p[0, 2, :] - l0_y *p[1, 2, :]

                l1_x = (p[1, 2, :] - p[1, 0, :]) * inv_det
                l1_y = (p[0, 0, :] - p[0, 2, :]) * inv_det
                l1_c = -l1_x*p[0, 2, :] - l1_y *p[1, 2, :]

                l2_x = -l0_x - l1_x
                l2_y = -l0_y - l1_y
                l2_c = 1-l0_c-l1_c

                p = p.unsqueeze(1).unsqueeze(1)
                D_x = p[2, :, :, 0, :] * l0_x + p[2, :, :, 1, :] * l1_x + p[2, :, :, 2, :] * l2_x
                D_y = p[2, :, :, 0, :] * l0_y + p[2, :, :, 1, :] * l1_y + p[2, :, :, 2, :] * l2_y
                D_c = (p[2,:,:,0,:]*l0_c + p[2,:,:,1,:]*l1_c + p[2,:,:,2,:]*l2_c)

                for i in range(range_y_min,range_y_max,self.block_size):

                    D_yc = D_y * (float(i)+self.y_grid_block) + D_c
                    l0_yc = l0_y * (float(i) + self.y_grid_block) + l0_c
                    l1_yc = l1_y * (float(i) + self.y_grid_block) + l1_c
                    l2_yc = l2_y * (float(i) + self.y_grid_block) + l2_c

                    for k in range(range_x_min,range_x_max,self.block_size):

                        target = (px_max>=k)*(px_min<k+self.block_size)*(py_max>=i)*(py_min<i+self.block_size)

                        face_ct = torch.sum(target, 0, dtype=torch.long)
                        if face_ct == 0:
                            continue

                        kxg = float(-k)-self.x_grid_block

                        M = (l0_yc[:,:,target]  >= (l0_x[:,:,target]* kxg)) * (l1_yc[:,:,target]  >= (l1_x[:,:,target]* kxg)) * (l2_yc[:,:,target] >= (l2_x[:,:,target]* kxg))

                        vis_ct = torch.max(torch.sum(M, 2)).item()

                        if vis_ct==1:
                            vis, idx = torch.max(M, 2)
                            self.index_buf[i:i + self.block_size, k:k + self.block_size][vis] = (self.face_index[0:target.shape[0]][target])[idx[vis]]

                        elif vis_ct>1:
                            D = M.bitwise_not().float() * inf_value + D_x[:,:,target] * (float(k)+self.x_grid_block) + D_yc[:,:,target]
                            D[D!=D]=inf_value
                            depth, idx = torch.min(D,2)
                            vis = depth< lower_inf_value

                            self.index_buf[i:i+self.block_size, k:k+self.block_size][vis] = (self.face_index[0:target.shape[0]][target])[idx[vis]]

                mask_ = (self.index_buf[0:self.height,0:self.width]!=num).float()
                index_buf_tmp = self.index_buf[0:self.height,0:self.width]
                index_buf_tmp[index_buf_tmp==num] = 0
                pix2pt = self.face[:,face_index_red[index_buf_tmp]]

            p = pt_2d[b, :, pix2pt]

            det = ((p[1, 1, :, :] - p[1, 2, :, :]) * (p[0, 0, :, :] - p[0, 2, :, :]) + (p[0, 2, :, :] - p[0, 1, :, :]) * (p[1, 0, :, :] - p[1, 2, :, :]))
            det = det.sign() * torch.clamp(det.abs(), min=ftiny)
            inv_det = 1 / det

            l0_x = ((p[1, 1, :, :] - p[1, 2, :, :]) * inv_det)
            l0_y = ((p[0, 2, :, :] - p[0, 1, :, :]) * inv_det)
            l0_c = (-l0_x*p[0, 2, :, :] - l0_y *p[1, 2, :, :])

            l1_x = ((p[1, 2, :, :] - p[1, 0, :, :]) * inv_det)
            l1_y = ((p[0, 0, :, :] - p[0, 2, :, :]) * inv_det)
            l1_c = (-l1_x*p[0, 2, :, :] - l1_y *p[1, 2, :, :])

            l2_x = -l0_x - l1_x
            l2_y = -l0_y - l1_y
            l2_c = 1-l0_c-l1_c

            c = color[b, :, pix2pt]
            C_x = (c[:,0,:,:]*l0_x.unsqueeze(0) + c[:,1,:,:]*l1_x.unsqueeze(0) + c[:,2,:,:]*l2_x.unsqueeze(0))
            C_y = (c[:,0,:,:]*l0_y.unsqueeze(0) + c[:,1,:,:]*l1_y.unsqueeze(0) + c[:,2,:,:]*l2_y.unsqueeze(0))
            C_c = (c[:,0,:,:]*l0_c.unsqueeze(0) + c[:,1,:,:]*l1_c.unsqueeze(0) + c[:,2,:,:]*l2_c.unsqueeze(0))

            image[b,:,:,:] += mask_ * (C_x * self.x_grid + C_y * self.y_grid + C_c)
            mask[b, :, :] += mask_

        return image,mask.detach()