import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import io
from skimage import measure

im_collection = io.imread_collection('path/to/tif/files/*.tif', plugin='tifffile')

print(im_collection)
im_3d = im_collection.concatenate()

print(im_3d.ndim)

for image in im_collection:
    print(image.shape)


# You can change the threshold as you like, whatever may be applicable to MRIs. With the CT scans I worked with
# I'd change the threshold to extract different organs/body parts (e.g., -300 was skeleton, I think).
threshold = 100
im_3d = im_3d.transpose(2,1,0)
verts, faces, norm, val = measure.marching_cubes_lewiner(im_3d)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
face_color = [0.5, 0.5, 1]
mesh.set_facecolor(face_color)
ax.add_collection3d(mesh)
ax.set_xlim(0, im_3d.shape[0])
ax.set_ylim(0, im_3d.shape[1])
ax.set_zlim(0, im_3d.shape[2])

plt.show()

plt.close(fig)


