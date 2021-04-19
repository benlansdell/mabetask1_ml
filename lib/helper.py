from itertools import product 

mouse_ids = ['mouse_0', 'mouse_1']
xy_ids = ['x', 'y']
bodypart_ids = ['nose', 'l_ear', 'r_ear', 'neck', 'l_hip', 'r_hip', 'tail_base']
colnames = ['_'.join(a) for a in product(mouse_ids, xy_ids, bodypart_ids)]

API_KEY = "0ba231d61506b40a4ae00df011cf0cb9"