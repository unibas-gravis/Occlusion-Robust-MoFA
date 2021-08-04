import torch

ftiny = torch.finfo(torch.float).tiny * 10**3

def calc_face_normal(vertex, face):

    #calc cross product of the neighboring two edges for each triangle
    tri=vertex[:,:,face]
    vec0 = tri[:, :, 1, :] - tri[:, :, 0, :]
    vec1 = tri[:, :, 2, :] - tri[:, :, 0, :]
    nx = vec0[:, 1, :] * vec1[:, 2, :] - vec0[:, 2, :] * vec1[:, 1, :]
    ny = vec0[:, 2, :] * vec1[:, 0, :] - vec0[:, 0, :] * vec1[:, 2, :]
    nz = vec0[:, 0, :] * vec1[:, 1, :] - vec0[:, 1, :] * vec1[:, 0, :]

    #calc face area and normalize cross product
    face_normal = torch.stack([nx,ny,nz],1)
    batch, ch, fnum = face_normal.shape
    face_area = torch.clamp(torch.norm(face_normal,2,1).view(batch,1,fnum),min = ftiny)
    face_normal = face_normal / torch.clamp(face_area,min = ftiny)

    return face_normal, face_area/2.0

def calc_vertex_to_face_map(vertex, face):

    vert_num = vertex.shape[2]
    face_num = face.shape[1]

    #count the number of neighboring faces for each vertex
    neighbor_count = torch.zeros(1, vert_num, dtype=torch.long, device=vertex.device)
    for f in range(face_num):
        neighbor_count[:, face[:, f]] += 1

    max_count = torch.max(neighbor_count)

    #store corresponding face id for each vertex
    vertex_to_face = torch.full((max_count, vert_num), -1, dtype=torch.long, device=vertex.device)

    for f in range(face_num):
        for v in range(3):
            for n in range(neighbor_count[0, face[v, f]]):
                if vertex_to_face[n, face[v, f]] == -1:
                    vertex_to_face[n, face[v, f]] = f   #if empty map is found, store the face id
                    break

    return vertex_to_face

def calc_vertex_normal(face_normal,face_area,vertex_to_face):

    vertex_mask = (vertex_to_face != -1).float().unsqueeze(0).unsqueeze(0)

    #set 0 as dummy value in empty map
    vertex_to_face[vertex_to_face == -1] = 0

    #generate non-empty map mask
    
    #calc weighted average of face normal
    normal_sum = torch.sum(vertex_mask * face_normal[:, :, vertex_to_face] * face_area[:, :, vertex_to_face], 2)
    weight_sum = torch.sum(vertex_mask*face_area[:, :, vertex_to_face],dim=2)
    vertex_normal = normal_sum / torch.clamp(weight_sum,min=ftiny)

    return vertex_normal

def perspective_projection(A,R,T,vertex):

    X = A @ (R @ vertex + T)
    X = X.sign() * torch.clamp(X.abs(),min=ftiny)

    projected_vertex = torch.cat([X[:, 0:2, :] / X[:, 2:3, :], X[:, 2:3, :]], 1)

    return projected_vertex