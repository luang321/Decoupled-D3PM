from pyvirtualdisplay import Display
import os

import pickle
import numpy as np
#import mayavi
#mayavi.engine.current_scene.scene.off_screen_rendering = True
from mayavi import mlab
from misc import is_main_process
import os 
import glob
import pickle
def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0] + 1)
    g_yy = np.arange(0, dims[1] + 1)
    sensor_pose = 10
    g_zz = np.arange(0, dims[2] + 1)

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float)

    coords_grid = (coords_grid * resolution) + resolution / 2

    temp = np.copy(coords_grid)
    temp[:, 0] = coords_grid[:, 1]
    temp[:, 1] = coords_grid[:, 0]
    coords_grid = np.copy(temp)

    return coords_grid


def draw(
    voxels,
    vox_origin,
    fov_mask,
    save_path,
    colors,
    vmax,
    voxel_size=0.2,
    
):  
    
    #  1 : Building 2 : Barrier 3 : Other 4 : Pedestrian 5 : Pole 6 : Road 7 : Ground 8 : Sidewalk 9 : Vegetation  10 : Vehicle
    # 0:unlabeld 1:"car" 2:"bicycle" 3:"motorcycle" 4:"truck" 5:"other-vehicle" 6: "person"
    # 7: "bicyclist" 8: "motorcyclist" 9:"road" 10:"parking" 11: "sidewalk" 12:"other-ground"
    # 13:"building" 14:"fence" 15 :"vegetation" 16:"trunk" 17:"terrain" 18:"pole" 19:"traffic-sign"
    mlab.options.offscreen = True


    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    )

    # Attach the predicted class to every voxel
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords[fov_mask, :]

    # Get the voxels outside FOV
    outfov_grid_coords = grid_coords[~fov_mask, :]

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 255)
    ]
    outfov_voxels = outfov_grid_coords[
        (outfov_grid_coords[:, 3] > 0) & (outfov_grid_coords[:, 3] < 255)
    ]

    figure = mlab.figure(size=(1400, 1400), bgcolor=(1, 1, 1))

    # Draw occupied inside FOV voxels
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.05 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=vmax,#10
    )
    plt_plot_fov.glyph.scale_mode =  'data_scaling_off'#"scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
    mlab.view( azimuth=45, elevation=54.735610317245346, distance= 139.8427289506119,focalpoint= [25.56833243 ,25.59999985 , 1.60527777])
    #azimuth, elevation, distance, focal_point = mlab.view()
    #print(f"Azimuth: {azimuth}, Elevation: {elevation}, Distance: {distance}, Focal point: {focal_point}")
    #print("draw_complete")
    mlab.savefig(save_path)
    mlab.close()
    #mlab.show()

def draw_once(y_pred,fov_mask_1,save_path,dataset="semantic_kitti"):
    
    vox_origin = np.array([0, -25.6, -2])
    if dataset == "semantic_kitti":
        colors = np.array(
            [
                [100, 150, 245, 255], #"car"
                [100, 230, 245, 255],#"bicycle"
                [30, 60, 150, 255],#"motorcycle"
                [80, 30, 180, 255],#"truck"
                [100, 80, 250, 255],#"other-vehicle"
                [255, 30, 30, 255],#"person"
                [255, 40, 200, 255],#"bicyclist"
                [150, 30, 90, 255],#"motorcyclist"
                [255, 0, 255, 255],#"road"
                [255, 150, 255, 255],#"parking"
                [75, 0, 75, 255],#"sidewalk" 
                [175, 0, 75, 255],#"other-ground"
                [255, 200, 0, 255],#"building"
                [255, 120, 50, 255],#"fence"
                [0, 175, 0, 255],#"vegetation"
                [135, 60, 0, 255],#"trunk"
                [150, 240, 80, 255],#"terrain"
                [255, 240, 150, 255],#"pole"
                [255, 0, 0, 255],#"traffic-sign"
            ]
        ).astype(np.uint8)
        # Visualize KITTI-360
        vmax = 19
        draw(
            y_pred,
            vox_origin,
            fov_mask_1,
            save_path= save_path,
            colors= colors,
            vmax = vmax,
            voxel_size=0.2,
        )
    else:
        colors = np.array(
        [
            [255, 200, 0, 255],    # Building
            [255, 120, 50, 255],    # Fences
            [175, 0, 75, 255],    # Other
            [255, 30, 30, 255],   # Pedestrian
            [255, 240, 150, 255],  # Pole
            [255, 0, 255, 255],     # Road
            [150, 240, 80, 255], # Ground
            [75, 0, 75, 255],  # Sidewalk
            [0, 175, 0, 255],      # Vegetation
            [100, 150, 245, 255],     # Vehicle
        ]
        ).astype(np.uint8 ) 
        vmax = 10
        draw(
            y_pred,
            vox_origin,
            fov_mask_1,
            save_path= save_path,
            colors= colors,
            vmax = vmax,
            voxel_size=0.2,

        )
    #display.stop()
    #display.close()
path  = "/data/lha/sepfusion/output/inpaint"
#"/data/lha/sepfusion/output/kitti/9_2_kitti_small_lr_444_500epoch"
#"/data/lha/diffusion/output/kitti/7_10_down_16_selective_poolfusion_one_step_argmax_poolfusion"
#"/data/lha/semantic_cgan/output/kitti/17_6_only3d_16_c_512_no_invalid_with_gan_loss_ccm_sim_loss"
#"/data/lha/semantic/output/kitti/15_26_bigger_lr_longer_schedul_sample_softmax_one_hot_gan_vox_kitti_1_WD0.0001_lr0.0005" #"/data/lha/semantic/output/kitti/15_25_sigmoid_gan_vox_kitti_1_WD0.0001_lr0.0001"

def draw_once_folder(folder):
    folder_path = os.path.join(path, folder)
    glob_path = os.path.join(
                folder_path, "*.pkl"
            )
    for ii,pic_path  in enumerate(glob.glob(glob_path)):
        frame_id =os.path.splitext( os.path.basename(pic_path))[0]
        output_path = os.path.join(folder_path,frame_id + ".png")

        if os.path.exists(output_path):
            continue
        else:
            with open(pic_path, "rb") as handle:
                b = pickle.load(handle)
                y_pred = b["y_pred"]
                print(np.max(y_pred))
                fov_mask_1  = b["fov_mask_1"] if "fov_mask_1" in b else np.ones_like(y_pred.reshape(-1), dtype=np.bool_)
                #y_pred[y_pred == 20] = 255
                draw_once(y_pred,fov_mask_1,output_path,dataset="carla")#"semantic_kitti""carla"
            print(output_path)
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = '0'  # 禁用软件渲染
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '0' 
    os.environ['DISPLAY'] = ':99'  # 设置 DISPLAY 环境
    display = Display(visible=0, size=(1280, 1024))
    display.start()
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    
    matching_folders = [f for f in folders if f.startswith("vis_")]
    draw_once_folder(path)
    '''
    draw_once_folder("vis")
    '''
    for folder in folders:
        draw_once_folder(folder)
    display.stop()