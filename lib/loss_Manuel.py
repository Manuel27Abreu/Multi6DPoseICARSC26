import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import torch
import matplotlib.pyplot as plt
import math
import numpy as np

class LossADD(_Loss):
    def __init__(self, num_points_mesh, centro=False):
        super(LossADD, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.centro = centro

    def forward(self, pred_r, pred_t, pred_c, target, model_points_W, idx, velodyne_gt, w, refine, pontocentro=False):

        return loss_calculationv2_manuel(pred_r, pred_t, pred_c, target, model_points_W, idx, velodyne_gt, w, refine, self.num_pt_mesh, pontocentro)
    
def loss_calculationv2_manuel(pred_r, pred_t, pred_c, target, model_points_W, idx, velodyne_gt, w, refine, num_point_mesh, pontocentro=False):
    bs = target.shape[0]

    total_loss1 = []
    total_loss2 = []

    for i in range(bs):
        loss1, loss2, pred, new_points = loss_calculationv2(pred_r[i], pred_t[i], pred_c, target[i], model_points_W[i], idx, velodyne_gt[i], w, refine, num_point_mesh, pontocentro)

        total_loss1.append(loss1)
        total_loss2.append(loss2)
    
    total_loss1 = torch.stack(total_loss1)
    total_loss2 = torch.stack(total_loss2)

    return total_loss1.mean(), total_loss2, pred, new_points

def loss_calculationv2(pred_r, pred_t, pred_c, target, model_points_W, idx, velodyne_gt, w, refine, num_point_mesh, pontocentro=False):
    target = target.unsqueeze(0)    # _W
    model_points_W = model_points_W.unsqueeze(0)
    velodyne_gt = velodyne_gt.unsqueeze(0)

    T = computeT(pred_r, pred_t)

    rt_inv = torch.linalg.inv(T)
    rotation_inv = rt_inv[:3, :3]
    translation_inv = rt_inv[:3, 3]
   
    pred = (rotation_inv @ model_points_W[0].float().T).T + translation_inv
    pred = pred.unsqueeze(0)
    dis = torch.norm((pred - target), dim=2)
    #print(dis.shape)
    loss = torch.mean(dis, dim=-1)
    #print(loss.shape)
    points = (rotation_inv @ velodyne_gt[0].float().T).T + translation_inv

    if pontocentro:
        centro = (rotation_inv @ model_points_W[0, 0:1].float().T).T + translation_inv
        centro = centro.unsqueeze(0)
        loss = torch.norm((centro - target[0, 0:1]), dim=2)
        loss = torch.mean(loss, dim=-1)

    if torch.isnan(loss.cpu()):
        print(pred_r, pred_t)
        quit()

    return loss, loss, pred, points

def computeT(pred_r, pred_t):        
    bs = 1
    num_p = 1

    a = torch.norm(pred_r, dim=0)
    if a>0.001:
        pred_r = pred_r / a
    else:
        pred_r[3]=1

    x, y, z, w = pred_r
    rot = torch.tensor([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ], device=pred_r.device)

    trans = pred_t.view(3, 1)

    upper = torch.cat([rot, trans], dim=1)
    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=pred_r.device)
    transform = torch.cat([upper, bottom], dim=0)

    return transform
        
class GeodesicLoss(_Loss):
    def __init__(self):
        super(GeodesicLoss, self).__init__(True)

    def forward(self, rt, q2):

        q1 = rotation_to_quaternion(rt)

        return quaternion_geodesic_loss(q1, q2)

def rotation_to_quaternion(R):
    """
    Converte batch de matrizes de rotação [B, 3, 3] para quaternions [B, 4]
    Formato do quaternion: (x, y, z, w)
    """
    B = R.shape[0]
    dtype = R.dtype
    device = R.device

    q = torch.zeros(B, 4, dtype=dtype, device=device)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # Caso 1: traço > 0
    cond1 = trace > 0
    if cond1.any():
        s = torch.sqrt(trace[cond1] + 1.0) * 2.0
        q[cond1, 3] = 0.25 * s
        q[cond1, 0] = (R[cond1, 2, 1] - R[cond1, 1, 2]) / s
        q[cond1, 1] = (R[cond1, 0, 2] - R[cond1, 2, 0]) / s
        q[cond1, 2] = (R[cond1, 1, 0] - R[cond1, 0, 1]) / s

    # Caso 2: R[0,0] maior
    cond2 = (~cond1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if cond2.any():
        s = torch.sqrt(1.0 + R[cond2, 0, 0] - R[cond2, 1, 1] - R[cond2, 2, 2]) * 2.0
        q[cond2, 3] = (R[cond2, 2, 1] - R[cond2, 1, 2]) / s
        q[cond2, 0] = 0.25 * s
        q[cond2, 1] = (R[cond2, 0, 1] + R[cond2, 1, 0]) / s
        q[cond2, 2] = (R[cond2, 0, 2] + R[cond2, 2, 0]) / s

    # Caso 3: R[1,1] maior
    cond3 = (~cond1) & (~cond2) & (R[:, 1, 1] > R[:, 2, 2])
    if cond3.any():
        s = torch.sqrt(1.0 + R[cond3, 1, 1] - R[cond3, 0, 0] - R[cond3, 2, 2]) * 2.0
        q[cond3, 3] = (R[cond3, 0, 2] - R[cond3, 2, 0]) / s
        q[cond3, 0] = (R[cond3, 0, 1] + R[cond3, 1, 0]) / s
        q[cond3, 1] = 0.25 * s
        q[cond3, 2] = (R[cond3, 1, 2] + R[cond3, 2, 1]) / s

    # Caso 4: R[2,2] maior
    cond4 = (~cond1) & (~cond2) & (~cond3)
    if cond4.any():
        s = torch.sqrt(1.0 + R[cond4, 2, 2] - R[cond4, 0, 0] - R[cond4, 1, 1]) * 2.0
        q[cond4, 3] = (R[cond4, 1, 0] - R[cond4, 0, 1]) / s
        q[cond4, 0] = (R[cond4, 0, 2] + R[cond4, 2, 0]) / s
        q[cond4, 1] = (R[cond4, 1, 2] + R[cond4, 2, 1]) / s
        q[cond4, 2] = 0.25 * s

    # Normalizar os quaternions
    q = q / q.norm(p=2, dim=1, keepdim=True)
    return q

def quaternion_geodesic_loss(q1, q2):
    q2 = q2.to(q1.device)

    # Normalize both quaternions to ensure they are unit quaternions
    q1 = F.normalize(q1, p=2, dim=-1)
    q2 = F.normalize(q2, p=2, dim=-1)

    # Compute dot product between quaternions
    dot_product = torch.abs(torch.sum(q1 * q2, dim=-1))
    
    # Geodesic distance (in radians)
    loss = 2 * torch.acos(torch.clamp(dot_product, -1.0, 1.0))  # clamp for numerical stability
    
    return torch.mean(loss)  # Return mean loss for the batch

def plot_batch_models(modelPoints, modelPointsGT):
    """
    Plota todos os modelos e seus GTs em 3D.

    Args:
        modelPoints: Tensor [B, N, 3]
        modelPointsGT: Tensor [B, N, 3]
    """
    B = modelPoints.shape[0]

    fig = plt.figure(figsize=(6*B, 6))

    for b in range(B):
        ax = fig.add_subplot(1, B, b+1, projection='3d')

        # Plota modelo e GT
        plot_axes(ax, modelPoints[b], color='r', label='Pred')
        plot_axes(ax, modelPointsGT[b], color='b', label='Ground Truth')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)

        ax.set_title(f"Batch {b}")
        ax.legend()

    plt.show()


def plot_axes(ax, points, color='r', label='Model'):
    """
    Plota os eixos de um modelo ou GT em 3D.
    """
    points = points.detach().cpu().numpy()
    center = points[0]
    for i in range(1, 4):
        ax.plot([center[0], points[i][0]],
                [center[1], points[i][1]],
                [center[2], points[i][2]],
                color=color, label=label if i==1 else "")
        ax.scatter(*points[i], color=color)
    ax.scatter(*center, color=color, s=50, marker='o')

def apply_quaternion_transform(modelPoints, quaternions, translations=None):
    B, N, _ = modelPoints.shape

    quaternions = quaternions / torch.norm(quaternions, dim=1, keepdim=True)

    x, y, z, w = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

    row1 = torch.stack([1 - 2*(y**2 + z**2),
                        2*(x*y - z*w),
                        2*(x*z + y*w)], dim=1)

    row2 = torch.stack([2*(x*y + z*w),
                        1 - 2*(x**2 + z**2),
                        2*(y*z - x*w)], dim=1)

    row3 = torch.stack([2*(x*z - y*w),
                        2*(y*z + x*w),
                        1 - 2*(x**2 + y**2)], dim=1)

    rotation_matrix = torch.stack([row1, row2, row3], dim=1)  # [B, 3, 3]

    rotated_points = torch.bmm(modelPoints, rotation_matrix.transpose(1, 2))  # [B, N, 3]

    if translations is not None:
        transformed_points = rotated_points + translations.unsqueeze(1)
    else:
        transformed_points = rotated_points

    return transformed_points, rotation_matrix

if __name__ == "__main__":
    lossadd = LossADD(1000)
    lossgeo = GeodesicLoss()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    identity_matrix = torch.tensor([
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]
    ], device=device)
    
    pred_r = torch.tensor([
        [0.0, 0.0, 0.0, 1],

        [0.0, 0.0, 0.0, 1]
    ], device=device)

    pred_t = torch.tensor([
        [0.0, 0.0, 0.0],

        [0.0, 0.0, 0.0]
    ], device=device)

    # pred_rt = computeT(pred_r, pred_t)
    # print("Pred:", pred_rt)

    pred_r_gt = torch.tensor([
        [0.2, 0.0, 0.25, 1.0],

        [0.0, 0.0, 0.0, 1.0]
    ], device=device)

    pred_t_gt = torch.tensor([
        [0.0, 0.0, 0.0],

        [1.0, 0.0, 0.0]
    ], device=device)

    batch_size = pred_r.shape[0]
    modelPoints = identity_matrix.unsqueeze(0).repeat(batch_size, 1, 1)

    # calculo do quarteniao GT (alterar aqui a posição no mundo e rotação)
    modelPointsGT, rotation_matrix = apply_quaternion_transform(modelPoints, pred_r_gt, pred_t_gt)

    rt = torch.eye(4, device='cuda:0').repeat(batch_size, 1, 1)
    rt[:, :3, :3] = rotation_matrix
    rt[:, :3, 3] = pred_t_gt[0]
    print("GT:", rt)

    points = torch.rand((batch_size, 1000, 3), device=device)

    loss_add, dis, new_points, new_target = lossadd(pred_r, pred_t, 1, modelPoints, modelPointsGT, 1, points, 1, False)
    loss_centro, _, _, _ = lossadd(pred_r, pred_t, 1, modelPoints, modelPointsGT, 1, points, 1, False, pontocentro=True)
    loss_geo = lossgeo(rt, pred_r)

    print(f"total: {loss_add+loss_geo:.4f} \t loss add: {loss_add:.4f} \t loss centro: {loss_centro:.4f} \t loss geo: {loss_geo:.4f} = {math.degrees(loss_geo):.2f}")

    plot_batch_models(modelPoints, modelPointsGT)
