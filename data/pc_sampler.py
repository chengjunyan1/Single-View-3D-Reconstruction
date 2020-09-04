import numpy as np
import trimesh
from tqdm import tqdm_notebook as tqdm
import math


class MeshSampler:
    def __init__(self, mesh, check_faces=False):
        self.mesh = mesh
        self.sphere_points = trimesh.sample.sample_surface(mesh.bounding_sphere, count=1000)[0]*2
        self.correct_faces = {i:0 for i in range(len(mesh.faces))}
        self.ray_mesh = RayMeshIntersector(geometry=mesh)
        self.faces_centroids = self.mesh.triangles.mean(axis=1)
        self.correct_points = np.array([])
        if check_faces:
            self.compute_visible_faces()
        else:
            self.correct_faces = {i:1 for i in range(len(mesh.faces))}
    
    def visible_faces(self):
        return self.correct_faces
        
    def compute_visible_faces(self):
        for i, face in enumerate(tqdm(self.mesh.triangles)):
            for point in face:
                ray_directions = -(self.sphere_points - point)
                faces_hit = self.ray_mesh.intersects_first(self.sphere_points, ray_directions)
                if i in faces_hit:
                    self.correct_faces[i] = 1
        return self.correct_faces
    
    def sample_points(self, n_points=10000):
        points = trimesh.sample.sample_surface(self.mesh, count=n_points)
        correct_points = []
        normals_for_points = []
        for i, point in enumerate(tqdm(points[0])):
            if self.correct_faces[points[1][i]] == 1:
                correct_points += [point]
                normals_for_points += [self.mesh.face_normals[points[1][i]]]
        self.correct_points = np.array(correct_points)
        self.normals_for_points = np.array(normals_for_points)
        return self.correct_points


model_path = 'C:\\ChengJunyan1\\3d\\ShapeNetCore.v2'
common_name='\\models\\model_normalized.obj'
gt_path = 'C:\\ChengJunyan1\\3d\\ShapeNetCore.v2.gt'
model_num=52491
gt_points=4096

def all_path():
    ap=[]
    categories=os.listdir(model_path)
    for i in categories:
        if i != '.DS_Store':
            category_path=os.path.join(model_path,i)
            objs=os.listdir(category_path)
            for j in objs:
                path=os.path.join(category_path,j+common_name)
                name=i+'+'+j
                ap.append((path,name))
    return ap

def log_gt():
    gt_dir=os.listdir(gt_path)
    gt_log=[i.split('.')[0] for i in l_dir]
    return gt_log

def sample_gt(work_range=range(0,model_num)):
    ids=all_path()
    log=log_gt()
    for i in work_range:
        path=ids[i][0]
        name=ids[i][1]
        if name not in log and os.path.exists(path):
            mesh=trimesh.exchange.load.load(path)
            sampler=MeshSampler(mesh)
            pcd=sampler.sample_points(gt_points)
            np.save(os.path.join(gt_path,name))