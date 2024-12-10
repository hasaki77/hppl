# Import Library

from mpi4py import MPI
from PIL import Image
import numpy as np
import time
import imageio

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Import image
if rank == 0:
    img = Image.open("rick.png").convert("RGB")
    img_array = np.array(img, dtype=np.uint8)
    rows, cols, _ = img_array.shape
else:
    img_array = None
    rows = cols = None

rows = comm.bcast(rows, root=0)
cols = comm.bcast(cols, root=0)

block_size = cols // size
local_img = np.zeros((rows, block_size, 3), dtype=np.uint8)

if (rank == 0):
    local_img = np.copy(img_array[:, :block_size, :])
    comm.Send(np.copy(img_array[:, block_size:, :]), dest=1, tag=0)
elif (rank == 1):
    comm.Recv(local_img, source=0, tag=0)

# List for animation frames
frames = []

# Moving to the right
for step in range(cols):
    right_column = local_img[:, -1, :]
    left_columns = local_img[:, :-1, :]
    local_img[:, 1:, :] = left_columns

    if (rank == 0):
        comm.Send(np.copy(right_column), dest=1, tag=2)

    elif (rank == 1):
        buffer = np.zeros((rows, 3), dtype=np.uint8)
        comm.Recv(buffer, source=0, tag=2)
        local_img[:, 0, :] = buffer

    if (rank == 0):
        buffer = np.zeros((rows, 3), dtype=np.uint8)
        comm.Recv(buffer, source=1, tag=3)
        local_img[:, 0, :] = buffer
    elif (rank == 1):
        comm.Send(np.copy(right_column), dest=0, tag=3)

    if rank == 1:
        comm.Send(np.copy(local_img), dest=0, tag=4)
    elif rank == 0:
        # First part
        combined_img = np.zeros((rows, cols, 3), dtype=np.uint8)
        combined_img[:, :block_size, :] = local_img
        # Second part
        buffer = np.zeros((rows, block_size, 3), dtype=np.uint8)
        comm.Recv(buffer, source=1, tag=4)
        combined_img[:, block_size:, :] = buffer
        # Saving frame
        frames.append(combined_img)
        
    comm.Barrier()
MPI.Finalize()
with imageio.get_writer('HW_6_MPI_animation.gif', mode='I', duration=0.0001) as writer:
    for frame in frames:
        writer.append_data(frame)
