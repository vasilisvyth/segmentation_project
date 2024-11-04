import os
from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def change_img_to_label_path(path):
    """
    Replaces imagesTr with labelsTr
    """
    parts = list(path.parts)  # get all directories within the path
    parts[parts.index("imagesTr")] = "labelsTr"  # Replace imagesTr with labelsTr
    return Path(*parts)  # Combine list back into a Path object

def animate(mri, mask, animation_name):
    from celluloid import Camera

    fig = plt.figure()
    camera = Camera(fig)  # Create the camera object from celluloid

    for i in range(mri.shape[2]):  # Sagital view
        plt.imshow(mri[:,:,i], cmap="bone")
        mask_ = np.ma.masked_where(mask[:,:,i]==0, mask[:,:,i])
        plt.imshow(mask_, alpha=0.5, cmap="autumn")
        # plt.axis("off")
        camera.snap()  # Store the current slice
    animation = camera.animate()  # Create the animation
    animation.save(animation_name, writer='ffmpeg')  # Save the animation as an MP4 file



TASK2_HEART_PATH = os.path.join(os.getcwd(),'Task02_Heart','Task02_Heart')
root = Path(os.path.join(TASK2_HEART_PATH,'imagesTr'))
# root = Path("/Task02_Heart/imagesTr/")
label = Path(os.path.join(TASK2_HEART_PATH,"labelsTr"))

sample_path = list(root.glob("la*"))[0]  # Choose a subject
sample_path_label = change_img_to_label_path(sample_path)

data = nib.load(sample_path)
label = nib.load(sample_path_label)

mri = data.get_fdata()
mask = label.get_fdata().astype(np.uint8)  # Class labels should not be handled as float64

nib.aff2axcodes(data.affine)
animate(mri, mask, 'mri_animation.mp4')
