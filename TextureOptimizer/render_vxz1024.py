import torch, json, cv2
import numpy as np
import imageio, trimesh, sys
import o_voxel
import utils3d
import torch.nn.functional as F

RES = 1024
resolution = 512


# sha256 = '0209_a44_debug'

# Load data
# coords, attributes = o_voxel.io.read_vxz("/path/to/bake_512.vxz", num_threads=4)
input_p = sys.argv[1]
out = input_p.replace('vxz', 'mp4')
coords, attributes = o_voxel.io.read_vxz(input_p, num_threads=4)
attributes['base_color'] = attributes['top6'][:,:3]

# coords, attributes = o_voxel.io.read_vxz(f'/path/to/output.vxz', num_threads=4)
# attributes['base_color'] = attributes['top6'][:,:3]

voxel_indices = coords
# coords, data = o_voxel.io.read("ovoxel_helmet.vxz")
position = (coords / RES - 0.5).cuda()
base_color = (attributes['base_color'] / 255).cuda()




def video_to_tensor(video_path: str, normalize: bool = True, device: str = 'cpu') -> torch.Tensor:
    """
    Read a video file and convert it to a normalized PyTorch tensor.

    Args:
        video_path (str): Path to the video file (e.g., .mp4).
        normalize (bool): If True, normalize pixel values from [0,255] → [-1, 1].
        device (str): Device to place the tensor ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Video tensor of shape (T, C, H, W) in range [-1, 1] if normalized,
                      or [0, 255] if not. dtype=torch.float32.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV loads as BGR; convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()

    if not frames:
        raise ValueError(f"No frames read from {video_path}")

    # Stack into (T, H, W, C)
    video_np = np.stack(frames, axis=0)  # (T, H, W, C)

    # Convert to tensor and permute to (T, C, H, W)
    video_tensor = torch.from_numpy(video_np).float().permute(0, 3, 1, 2)  # (T, C, H, W)

    if normalize:
        # Map [0, 255] → [-1, 1]
        video_tensor = video_tensor * (2.0 / 255.0) - 1.0

    return video_tensor.to(device)


def generate_camera_positions_continuous(yaws, pitch, r):
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
    
    positions = []
    # 起始点：在 Y-Z 平面，yaw = 0 对应 +Y 方向
    pitch_rad = torch.deg2rad(torch.tensor(float(pitch), dtype=torch.float32, device='cuda'))
    start = torch.tensor([
        0.0,
        torch.cos(pitch_rad).item(),  # Y = cos(pitch)
        torch.sin(pitch_rad).item()   # Z = sin(pitch)
    ], dtype=torch.float32, device='cuda') * r

    for yaw in yaws:
        yaw_rad = torch.tensor(float(yaw), dtype=torch.float32, device='cuda')
        cos_y = torch.cos(yaw_rad)
        sin_y = torch.sin(yaw_rad)
        # 👇 绕 Z 轴旋转（因为方位角是绕 Z 轴转）
        R_z = torch.tensor([
            [ cos_y, -sin_y, 0.0],
            [ sin_y,  cos_y, 0.0],
            [   0.0,    0.0, 1.0]
        ], dtype=torch.float32, device='cuda')
        pos = R_z @ start
        positions.append(pos)
    
    if not is_list:
        positions = positions[0]
    return positions

def _safe_look_at(pos, target, pitch_deg):
    forward = F.normalize(target - pos, dim=-1)
    
    # 根据 pitch 判断相机是向上还是向下看
    if 90 < pitch_deg < 270:
        # 相机主要朝上半球（抬头），此时若仍用 [0,0,1] 会翻转
        # 所以强制 up 为 [0,0,-1] 来维持图像正立
        up = torch.tensor([0., 0., -1.], dtype=torch.float32, device='cuda')
    else:
        # 其他角度用标准向上
        up = torch.tensor([0., 0., 1.], dtype=torch.float32, device='cuda')

    return utils3d.extrinsics_look_at(pos, target, up)

def yaw_pitch_r_fov_to_extrinsics_intrinsics_robust(yaws, pitchs, rs, fovs):
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)

    extrinsics = []
    intrinsics = []

    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        yaw_shifted = -yaw 
        pos = generate_camera_positions_continuous(yaw_shifted, pitch, r)

        target = torch.zeros(3, dtype=torch.float32, device='cuda')
        
        # 👇 把 pitch_deg 传进去做判断
        extr = _safe_look_at(pos, target, pitch_deg=pitch.item() if torch.is_tensor(pitch) else float(pitch))
        
        fov_rad = torch.deg2rad(torch.tensor(float(fov), dtype=torch.float32, device='cuda'))
        intr = utils3d.intrinsics_from_fov_xy(fov_rad, fov_rad)

        extrinsics.append(extr)
        intrinsics.append(intr)

    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]

    return extrinsics, intrinsics


def yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs):
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)
    extrinsics = []
    intrinsics = []
    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        fov = torch.deg2rad(torch.tensor(float(fov))).cuda()
        yaw = torch.tensor(float(yaw)).cuda()
        pitch = torch.tensor(float(pitch)).cuda()
        orig = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ]).cuda() * r
        extr = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        extrinsics.append(extr)
        intrinsics.append(intr)
    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]
    return extrinsics, intrinsics


def render_video_horizontal_geo(bg_color=(0, 0, 0), num_frames=121, r=1.5, fov=49.13434207760448, **kwargs):
    # yaws = -torch.linspace(0, 2 * 3.1415, num_frames+1) + np.pi
    yaws = [np.float64(3.141592653589793), np.float64(3.089665502290788), np.float64(3.0377383509917832), np.float64(2.9858111996927788), np.float64(2.933884048393774), np.float64(2.881956897094769), np.float64(2.8300297457957644), np.float64(2.778102594496759), np.float64(2.7261754431977545), np.float64(2.6742482918987496), np.float64(2.6223211405997446), np.float64(2.5703939893007397), np.float64(2.5184668380017348), np.float64(2.4665396867027303), np.float64(2.4146125354037253), np.float64(2.3626853841047204), np.float64(2.310758232805716), np.float64(2.2588310815067105), np.float64(2.206903930207706), np.float64(2.154976778908701), np.float64(2.103049627609696), np.float64(2.0511224763106917), np.float64(1.9991953250116865), np.float64(1.9472681737126818), np.float64(1.8953410224136769), np.float64(1.843413871114672), np.float64(1.7914867198156672), np.float64(1.7395595685166625), np.float64(1.6876324172176576), np.float64(1.6357052659186526), np.float64(1.583778114619648), np.float64(1.531850963320643), np.float64(1.4799238120216383), np.float64(1.4279966607226333), np.float64(1.3760695094236284), np.float64(1.3241423581246237), np.float64(1.272215206825619), np.float64(1.2202880555266138), np.float64(1.168360904227609), np.float64(1.1164337529286044), np.float64(1.0645066016295996), np.float64(1.0125794503305945), np.float64(0.9606522990315898), np.float64(0.908725147732585), np.float64(0.85679799643358), np.float64(0.804870845134575), np.float64(0.7529436938355706), np.float64(0.7010165425365655), np.float64(0.6490893912375605), np.float64(0.597162239938556), np.float64(0.545235088639551), np.float64(0.49330793734054645), np.float64(0.4413807860415415), np.float64(0.38945363474253636), np.float64(0.33752648344353187), np.float64(0.2855993321445267), np.float64(0.233672180845522), np.float64(0.18174502954651728), np.float64(0.12981787824751212), np.float64(0.0778907269485074), np.float64(0.02596357564950269), np.float64(-0.025963575649502246), np.float64(-0.0778907269485074), np.float64(-0.12981787824751168), np.float64(-0.1817450295465166), np.float64(-0.23367218084552177), np.float64(-0.2855993321445265), np.float64(-0.3375264834435314), np.float64(-0.38945363474253636), np.float64(-0.44138078604154085), np.float64(-0.4933079373405458), np.float64(-0.5452350886395512), np.float64(-0.5971622399385557), np.float64(-0.649089391237561), np.float64(-0.7010165425365655), np.float64(-0.75294369383557), np.float64(-0.804870845134575), np.float64(-0.8567979964335799), np.float64(-0.9087251477325844), np.float64(-0.9606522990315898), np.float64(-1.0125794503305938), np.float64(-1.0645066016295988), np.float64(-1.1164337529286041), np.float64(-1.1683609042276086), np.float64(-1.2202880555266136), np.float64(-1.2722152068256185), np.float64(-1.324142358124623), np.float64(-1.3760695094236288), np.float64(-1.4279966607226329), np.float64(-1.4799238120216378), np.float64(-1.5318509633206432), np.float64(-1.5837781146196472), np.float64(-1.6357052659186522), np.float64(-1.687632417217658), np.float64(-1.739559568516662), np.float64(-1.791486719815667), np.float64(-1.843413871114672), np.float64(-1.8953410224136769), np.float64(-1.9472681737126813), np.float64(-1.9991953250116863), np.float64(-2.0511224763106917), np.float64(-2.1030496276096953), np.float64(-2.1549767789087007), np.float64(-2.2069039302077056), np.float64(-2.25883108150671), np.float64(-2.310758232805715), np.float64(-2.3626853841047204), np.float64(-2.414612535403725), np.float64(-2.4665396867027294), np.float64(-2.5184668380017348), np.float64(-2.5703939893007393), np.float64(-2.6223211405997446), np.float64(-2.67424829189875), np.float64(-2.7261754431977545), np.float64(-2.778102594496759), np.float64(-2.8300297457957644), np.float64(-2.881956897094769), np.float64(-2.9338840483937734), np.float64(-2.9858111996927788), np.float64(-3.0377383509917832), np.float64(-3.0896655022907877)]
    yaws = torch.tensor(yaws)
    # import pdb;pdb.set_trace()
    pitch = torch.zeros_like(yaws)
    yaws = yaws.tolist()
    pitch = pitch.tolist()
    # import pdb;pdb.set_trace()
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics_robust(yaws, pitch, r, fov)
    return extrinsics, intrinsics

def render_video_vertical_geo(bg_color=(0, 0, 0), num_frames=121, r=1.5, fov=49.13434207760448, **kwargs):
    pitch = [np.float64(0.0), np.float64(2.975206611570253), np.float64(5.950413223140494), np.float64(8.925619834710746), np.float64(11.900826446280988), np.float64(14.87603305785124), np.float64(17.851239669421492), np.float64(20.826446280991735), np.float64(23.801652892561986), np.float64(26.77685950413223), np.float64(29.75206611570248), np.float64(32.727272727272734), np.float64(35.702479338842984), np.float64(38.67768595041321), np.float64(41.65289256198347), np.float64(44.62809917355372), np.float64(47.60330578512397), np.float64(50.57851239669423), np.float64(53.55371900826446), np.float64(56.52892561983471), np.float64(59.50413223140496), np.float64(62.47933884297519), np.float64(65.45454545454547), np.float64(68.4297520661157), np.float64(71.40495867768595), np.float64(74.3801652892562), np.float64(77.35537190082643), np.float64(80.33057851239671), np.float64(83.30578512396694), np.float64(86.28099173553719), np.float64(89.25619834710744), np.float64(92.23140495867767), np.float64(95.20661157024794), np.float64(98.18181818181817), np.float64(101.15702479338843), np.float64(104.13223140495869), np.float64(107.10743801652892), np.float64(110.08264462809919), np.float64(113.05785123966942), np.float64(116.03305785123968), np.float64(119.00826446280992), np.float64(121.98347107438018), np.float64(124.9586776859504), np.float64(127.93388429752066), np.float64(130.9090909090909), np.float64(133.88429752066116), np.float64(136.85950413223142), np.float64(139.83471074380162), np.float64(142.8099173553719), np.float64(145.78512396694217), np.float64(148.7603305785124), np.float64(151.73553719008262), np.float64(154.71074380165285), np.float64(157.68595041322314), np.float64(160.66115702479343), np.float64(163.63636363636365), np.float64(166.61157024793388), np.float64(169.5867768595041), np.float64(172.56198347107437), np.float64(175.53719008264466), np.float64(178.51239669421489), np.float64(181.48760330578511), np.float64(184.46280991735534), np.float64(187.43801652892563), np.float64(190.4132231404959), np.float64(193.38842975206612), np.float64(196.36363636363635), np.float64(199.33884297520657), np.float64(202.31404958677686), np.float64(205.28925619834715), np.float64(208.26446280991738), np.float64(211.2396694214876), np.float64(214.21487603305783), np.float64(217.1900826446281), np.float64(220.16528925619838), np.float64(223.1404958677686), np.float64(226.11570247933884), np.float64(229.09090909090907), np.float64(232.06611570247935), np.float64(235.04132231404958), np.float64(238.01652892561984), np.float64(240.99173553719007), np.float64(243.96694214876035), np.float64(246.94214876033058), np.float64(249.9173553719008), np.float64(252.8925619834711), np.float64(255.86776859504133), np.float64(258.8429752066116), np.float64(261.8181818181818), np.float64(264.79338842975204), np.float64(267.7685950413223), np.float64(270.74380165289256), np.float64(273.71900826446284), np.float64(276.6942148760331), np.float64(279.6694214876033), np.float64(282.6446280991736), np.float64(285.6198347107438), np.float64(288.5950413223141), np.float64(291.5702479338843), np.float64(294.5454545454545), np.float64(297.5206611570248), np.float64(300.495867768595), np.float64(303.4710743801653), np.float64(306.44628099173553), np.float64(309.42148760330576), np.float64(312.39669421487605), np.float64(315.3719008264463), np.float64(318.34710743801656), np.float64(321.3223140495868), np.float64(324.297520661157), np.float64(327.2727272727273), np.float64(330.24793388429754), np.float64(333.2231404958678), np.float64(336.198347107438), np.float64(339.1735537190082), np.float64(342.1487603305785), np.float64(345.12396694214874), np.float64(348.099173553719), np.float64(351.07438016528926), np.float64(354.0495867768595), np.float64(357.02479338842977)]
    pitch = torch.tensor(pitch)
    # yaws = [np.float64(3.141592653589793), np.float64(3.089665502290788), np.float64(3.0377383509917832), np.float64(2.9858111996927788), np.float64(2.933884048393774), np.float64(2.881956897094769), np.float64(2.8300297457957644), np.float64(2.778102594496759), np.float64(2.7261754431977545), np.float64(2.6742482918987496), np.float64(2.6223211405997446), np.float64(2.5703939893007397), np.float64(2.5184668380017348), np.float64(2.4665396867027303), np.float64(2.4146125354037253), np.float64(2.3626853841047204), np.float64(2.310758232805716), np.float64(2.2588310815067105), np.float64(2.206903930207706), np.float64(2.154976778908701), np.float64(2.103049627609696), np.float64(2.0511224763106917), np.float64(1.9991953250116865), np.float64(1.9472681737126818), np.float64(1.8953410224136769), np.float64(1.843413871114672), np.float64(1.7914867198156672), np.float64(1.7395595685166625), np.float64(1.6876324172176576), np.float64(1.6357052659186526), np.float64(1.583778114619648), np.float64(1.531850963320643), np.float64(1.4799238120216383), np.float64(1.4279966607226333), np.float64(1.3760695094236284), np.float64(1.3241423581246237), np.float64(1.272215206825619), np.float64(1.2202880555266138), np.float64(1.168360904227609), np.float64(1.1164337529286044), np.float64(1.0645066016295996), np.float64(1.0125794503305945), np.float64(0.9606522990315898), np.float64(0.908725147732585), np.float64(0.85679799643358), np.float64(0.804870845134575), np.float64(0.7529436938355706), np.float64(0.7010165425365655), np.float64(0.6490893912375605), np.float64(0.597162239938556), np.float64(0.545235088639551), np.float64(0.49330793734054645), np.float64(0.4413807860415415), np.float64(0.38945363474253636), np.float64(0.33752648344353187), np.float64(0.2855993321445267), np.float64(0.233672180845522), np.float64(0.18174502954651728), np.float64(0.12981787824751212), np.float64(0.0778907269485074), np.float64(0.02596357564950269), np.float64(-0.025963575649502246), np.float64(-0.0778907269485074), np.float64(-0.12981787824751168), np.float64(-0.1817450295465166), np.float64(-0.23367218084552177), np.float64(-0.2855993321445265), np.float64(-0.3375264834435314), np.float64(-0.38945363474253636), np.float64(-0.44138078604154085), np.float64(-0.4933079373405458), np.float64(-0.5452350886395512), np.float64(-0.5971622399385557), np.float64(-0.649089391237561), np.float64(-0.7010165425365655), np.float64(-0.75294369383557), np.float64(-0.804870845134575), np.float64(-0.8567979964335799), np.float64(-0.9087251477325844), np.float64(-0.9606522990315898), np.float64(-1.0125794503305938), np.float64(-1.0645066016295988), np.float64(-1.1164337529286041), np.float64(-1.1683609042276086), np.float64(-1.2202880555266136), np.float64(-1.2722152068256185), np.float64(-1.324142358124623), np.float64(-1.3760695094236288), np.float64(-1.4279966607226329), np.float64(-1.4799238120216378), np.float64(-1.5318509633206432), np.float64(-1.5837781146196472), np.float64(-1.6357052659186522), np.float64(-1.687632417217658), np.float64(-1.739559568516662), np.float64(-1.791486719815667), np.float64(-1.843413871114672), np.float64(-1.8953410224136769), np.float64(-1.9472681737126813), np.float64(-1.9991953250116863), np.float64(-2.0511224763106917), np.float64(-2.1030496276096953), np.float64(-2.1549767789087007), np.float64(-2.2069039302077056), np.float64(-2.25883108150671), np.float64(-2.310758232805715), np.float64(-2.3626853841047204), np.float64(-2.414612535403725), np.float64(-2.4665396867027294), np.float64(-2.5184668380017348), np.float64(-2.5703939893007393), np.float64(-2.6223211405997446), np.float64(-2.67424829189875), np.float64(-2.7261754431977545), np.float64(-2.778102594496759), np.float64(-2.8300297457957644), np.float64(-2.881956897094769), np.float64(-2.9338840483937734), np.float64(-2.9858111996927788), np.float64(-3.0377383509917832), np.float64(-3.0896655022907877)]
    # yaws = torch.tensor(yaws)
    # import pdb;pdb.set_trace()
    yaws = torch.zeros_like(pitch) +  torch.pi
    yaws = yaws.tolist() 
    pitch = pitch.tolist()
    # import pdb;pdb.set_trace()
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics_robust(yaws, pitch, r, fov)
    return extrinsics, intrinsics


# Render


renderer = o_voxel.rasterize.VoxelRenderer(
    rendering_options={"resolution": resolution, "ssaa": 1}
    # rendering_options={"resolution": resolution, "ssaa": 2, 'near': 1, 'far': 100.0}
)



extrsh, intrsh = render_video_horizontal_geo(resolution=resolution)
extrsv, intrsv = render_video_vertical_geo(resolution=resolution)
# projec = torch.load('/path/to/projections.pt')
# extrs = projec[-121:]
# import pdb;pdb.set_trace()
extrs = extrsh + extrsv
intrs = intrsh + intrsv


vis_frames = []
voxel_indice_frames = []
import matplotlib.pyplot as plt
import os
for extr, intr in zip(extrs, intrs):

    output = renderer.render(
        position=position,          # Voxel centers
        attrs=base_color,           # Color/Opacity etc.
        voxel_size=1.0/RES,
        extrinsics=extr.cuda(),
        intrinsics=intr.cuda()
    )
    # depth = output['depth']

    # # 如果是 [H, W, 1]，squeeze 成 [H, W]
    # if depth.ndim == 3 and depth.shape[-1] == 1:
    #     depth = depth.squeeze(-1)
    # elif depth.ndim == 3:
    #     # 如果是 [B, H, W]，取第一张
    #     depth = depth[0]

    # # 转到 CPU 并转为 numpy
    # depth_np = depth.cpu().numpy()

    # # 创建有效掩码：只保留正的、合理的深度值（排除 -5e28 这类）
    # valid_mask = (depth_np > 0) & (depth_np < 1e5)

    # # 可选：将无效区域设为 NaN，这样 matplotlib 不会着色
    # depth_vis = depth_np.astype(float)
    # depth_vis[~valid_mask] = float('nan')  # 或设为 0，但 NaN 在 colormap 中默认不渲染

    # # 设置输出路径
    # output_path = "depth_visualization.png"

    # # 绘图
    # plt.figure(figsize=(10, 8))
    # plt.imshow(depth_vis, cmap='plasma')  # 'viridis', 'inferno', 'magma' 也可
    # plt.colorbar(label='Depth')
    # plt.title('Rendered Depth Map (invalid pixels masked)')
    # plt.axis('off')  # 可选：去掉坐标轴

    # # 保存
    # plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
    # plt.close()

    # print(f"Depth visualization saved to: {os.path.abspath(output_path)}")
    # import pdb;pdb.set_trace()

    recover_voxel_indice = torch.round(output.attr * RES).long()

    # import pdb;pdb.set_trace()
    voxel_indice_frames.append(recover_voxel_indice)
    
    visualzie_recover_voxel_indice = np.clip(
        output.attr.permute(1, 2, 0).cpu().numpy() * 255, 0, 255
    ).astype(np.uint8)
    
    vis_frames.append(visualzie_recover_voxel_indice)

    

imageio.mimsave(out, vis_frames, fps=15)
print(out)
    
