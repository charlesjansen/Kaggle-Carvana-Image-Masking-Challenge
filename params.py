from model.u_net import get_unet_128, get_unet_256, get_unet_512, get_unet_1024, get_unet_renorm_1024
from model.w_net import get_wnet_128
from batch_renorm import BatchRenormalization

input_size = 128

max_epochs = 500#initialy 50
batch_size = 16#16 for unet128 ; 8 for 512; 4 for 1024

orig_width = 1918
orig_height = 1280

threshold = 0.5

model = get_unet_128()
