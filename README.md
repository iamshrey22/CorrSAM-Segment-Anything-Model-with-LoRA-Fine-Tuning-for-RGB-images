# Corr-SAM
Segment Anything Model with LoRA Fine Tuning for RGB images

The data folder structure is like below-  
  data/    
         - /images   
         - /rgb_masks (generated from json annotation file)   
         - /masks (label masks when each pixel is between 0 - num_class generated from rgb_masks)  (These masks are fed into model)
