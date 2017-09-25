#from model_moi.u_net import get_unet_128, get_unet_256, get_unet_512, get_unet_1024, get_unet_renorm_1024, get_unet_renorm_1280
#from model_moi.u_net_incep import get_unet_renorm_128, get_unet_renorm_incep_128, get_wnet_renorm_incep_1024, get_unet_renorm_incep_eco_1024, get_wnet_renorm_incep_eco_layer1incep_1024
#from model_moi.u_net_incep import get_unet_renorm_incep_eco_1280_alternate_adam
from model_moi.u_net_incep import get_unet_Norm_incep_eco_512_alternate_RMS

input_size = 512

input_width = input_size
input_height = input_size

max_epochs = 250 #initialy 50
batch_size = 1   #16 for unet128 ; 8 for 512; 4 for 1024, sans incep
test_size = 0.2

orig_width = 1918
orig_height = 1280

threshold = 0.5

model = get_unet_Norm_incep_eco_512_alternate_RMS()
