import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, FFMpegWriter
from PIL import Image

class MultiFolderAnimation:
    def __init__(self, folders, output_file='animation.mp4', fps=30, resize_images=False, image_size=(400, 300)):
        
        self.folders = folders
        self.output_file = output_file
        self.fps = fps
        self.resize_images = resize_images
        self.image_size = image_size


        self.image_paths = []

        for i, folder in enumerate(folders):
            print(f"Processing folder {i+1}: {folder}")

            if not os.path.exists(folder):
                print(f"Error: Folder does not exist: {folder}")
                continue

            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
            all_paths = []
            for ext in image_extensions:
                all_paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))

            if not all_paths:
                print(f"Warning: No image files found in folder {folder}")
                self.image_paths.append([])
                continue

            def extract_number(filename):
                numbers = re.findall(r'\d+', os.path.basename(filename))
                return int(numbers[-1]) if numbers else 0

            all_paths.sort(key=extract_number)
            self.image_paths.append(all_paths)

        if self.image_paths:
            min_len = min(len(paths) for paths in self.image_paths)
            for i in range(len(self.image_paths)):
                if len(self.image_paths[i]) > min_len:
                    self.image_paths[i] = self.image_paths[i][:min_len]
                else:

                    self.image_paths[i].extend([None] * (min_len - len(self.image_paths[i])))
            print(f"All folders processed. Minimum number of frames across folders: {min_len}")
        else:
            print("No valid image paths found in any folder. Exiting.")
            return

        n = len(folders)
        if n == 1:
            fig, axes = plt.subplots(1, 1, figsize=(6, 6))
            axes = [axes]
        elif n == 2:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        elif n == 3:
      
            fig = plt.figure(figsize=(12, 8))
            gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1]) 
            ax1 = fig.add_subplot(gs[0, :])   
            ax2 = fig.add_subplot(gs[1, 0])  
            ax3 = fig.add_subplot(gs[1, 1])   
            axes = [ax1, ax2, ax3]
        elif n == 4:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()
        else:
      
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
            axes = axes.flatten()

        self.fig = fig
        self.axes = axes[:n]  

        self.img_plots = [None] * n

        for i, ax in enumerate(self.axes):
            ax.axis('off')
            folder_name = os.path.basename(folders[i])
            ax.set_title(folder_name, fontsize=14, fontweight='bold')

        for ax in axes[n:]:
            ax.set_visible(False)

        plt.tight_layout()

    def load_image(self, path):

        if path is None:
            size = self.image_size
            return np.ones((size[1], size[0], 3), dtype=np.uint8) * 255

        try:
            img = Image.open(path)
            if img.mode == 'RGBA':
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[3])
                img = rgb_img
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            if self.resize_images:
                img = img.resize(self.image_size, Image.Resampling.LANCZOS)
            return np.array(img)
        except Exception as e:
    
            size = self.image_size
            return np.ones((size[1], size[0], 3), dtype=np.uint8) * 255

    def update_frame(self, frame_idx):
  
        total_frames = len(self.image_paths[0])
        if frame_idx >= total_frames:
            return self.img_plots

        n = len(self.image_paths)
        for i in range(n):
            if frame_idx < len(self.image_paths[i]):
                img_path = self.image_paths[i][frame_idx]
                img_array = self.load_image(img_path)
                height, width = img_array.shape[:2]
                extent = (-0.5, width - 0.5, height - 0.5, -0.5)
                if self.img_plots[i] is None:
                    self.img_plots[i] = self.axes[i].imshow(img_array, extent=extent)
                else:
                    self.img_plots[i].set_data(img_array)
                    self.img_plots[i].set_extent(extent)
                self.axes[i].set_xlim(-0.5, width - 0.5)
                self.axes[i].set_ylim(height - 0.5, -0.5)

        ###self.fig.suptitle(f'Frame: {frame_idx+1}/{total_frames}', fontsize=16, fontweight='bold',x=0.02,y=0.9)
        return self.img_plots

    def create_animation(self):
       
        if not self.image_paths or not self.image_paths[0]:
            print("No valid image paths found. Cannot create animation.")
            return

        total_frames = len(self.image_paths[0])
        
        anim = FuncAnimation(
            self.fig,
            self.update_frame,
            frames=total_frames,
            interval=1000/self.fps,
            blit=True
        )

        if self.output_file.lower().endswith('.gif'):
            anim.save(
                self.output_file,
                writer='pillow',
                fps=self.fps,
                progress_callback=lambda i, n: print(f'\r Speed: {i+1}/{n} frames ({i/n*100:.1f}%)', end='')
            )
        else:
            try:
                try:
                    import imageio_ffmpeg
                    plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
                    print(f"\nUsing bundled ffmpeg: {plt.rcParams['animation.ffmpeg_path']}")
                except ImportError:
                    pass

                writer = FFMpegWriter(
                    fps=self.fps,
                    codec='libx264',
                    extra_args=['-pix_fmt', 'yuv420p']
                )
                anim.save(
                    self.output_file,
                    writer=writer,
                    dpi=100,
                    progress_callback=lambda i, n: print(f'\r Speed: {i+1}/{n} frames ({i/n*100:.1f}%)', end='')
                )
            except Exception as e:
                print(f"\nffmpeg unavailable ({e}), saving as GIF instead")
                gif_output = self.output_file.rsplit('.', 1)[0] + '.gif'
                anim.save(
                    gif_output,
                    writer='pillow',
                    fps=self.fps,
                    progress_callback=lambda i, n: print(f'\r Speed: {i+1}/{n} frames ({i/n*100:.1f}%)', end='')
                )
                self.output_file = gif_output

        plt.show()

    def preview_frames(self, frame_indices=None):
  
        if frame_indices is None:
            frame_indices = [0, 100, 200, 300, 400, 500]

        for idx in frame_indices:
            if idx < len(self.image_paths[0]):
                
                for i in range(len(self.folders)):
                    if idx < len(self.image_paths[i]) and self.image_paths[i][idx]:
                        filename = os.path.basename(self.image_paths[i][idx])
                      
                    else:
                        print(f"  Folder {self.folders[i]} does not have a frame at index {idx}.")
                self.update_frame(idx)
                plt.show()
            else:
                print(f"Frame {idx+1} is out of range.")


folders = [
        "R-factor",
        "LEED-IV",
        "xy_view",
    ]
    
animator = MultiFolderAnimation(
        folders=folders,
        output_file='four_folders_animation.mp4',  
        fps=30,
        resize_images=True)

animator.create_animation()
