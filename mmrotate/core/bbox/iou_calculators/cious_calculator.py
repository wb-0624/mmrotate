import math
import torch
from .builder import build_iou_calculator

HBB_IOU_CALCULATOR = build_iou_calculator(dict(type='BboxOverlaps2D'))

def cious_calculate(gt_bboxes, bboxes):
    if bboxes.size(-1) == 4 and gt_bboxes.size(-1) == 4:
        # (x_min, y_min, x_max, y_max)
        return calculate_cious_hbb(gt_bboxes, bboxes)
    elif bboxes.size(-1) == 5 and gt_bboxes.size(-1) == 5:
        # (cx, cy, w, h, angle)
        raise NotImplementedError('obb not implemented yet!')
    else:
        raise NotImplementedError()


def calculate_cious_hbb(bboxes1, bboxes2):
    """
    bboxes1: gt (x_min,y_min,x_max,y_max)
    bboxes2: anchor (x_min,y_min,x_max,y_max)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bboxes1 = bboxes1.to(device)
    bboxes2 = bboxes2.to(device)


    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    # 保证bboxes1是gt，bboxes2是anchor
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True

    w1 = (bboxes1[:, 2] - bboxes1[:, 0]).unsqueeze(1)
    h1 = (bboxes1[:, 3] - bboxes1[:, 1]).unsqueeze(1)
    w2 = (bboxes2[:, 2] - bboxes2[:, 0]).unsqueeze(0)
    h2 = (bboxes2[:, 3] - bboxes2[:, 1]).unsqueeze(0)

    center_x1 = (bboxes1[:, 0] + bboxes1[:, 2])/2
    center_y1 = (bboxes1[:, 1] + bboxes1[:, 3])/2
    center_x2 = (bboxes2[:, 0] + bboxes2[:, 2])/2
    center_y2 = (bboxes2[:, 1] + bboxes2[:, 3])/2

    center_x1 = center_x1.unsqueeze(1)
    center_y1 = center_y1.unsqueeze(1)
    center_x2 = center_x2.unsqueeze(0)
    center_y2 = center_y2.unsqueeze(0)

    c_l = torch.min(bboxes1[:, 0].unsqueeze(1), bboxes2[:, 0].unsqueeze(0))
    c_r = torch.max(bboxes1[:, 2].unsqueeze(1), bboxes2[:, 2].unsqueeze(0))
    c_t = torch.min(bboxes1[:, 1].unsqueeze(1), bboxes2[:, 1].unsqueeze(0))
    c_b = torch.max(bboxes1[:, 3].unsqueeze(1), bboxes2[:, 3].unsqueeze(0))

    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    c_diag = torch.clamp((c_r - c_l),min=0)**2 + torch.clamp((c_b - c_t),min=0)**2

    u = (inter_diag) / c_diag
    iou = HBB_IOU_CALCULATOR(bboxes1, bboxes2)
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
    with torch.no_grad():
        S = (iou>0.5).float()
        alpha= S*v/(1-iou+v)
    cious = iou - u - alpha * v
    if exchange:
        cious = cious.T
    
    return cious