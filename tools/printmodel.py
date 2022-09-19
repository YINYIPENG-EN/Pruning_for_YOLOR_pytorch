import torch.nn

from models.models import *
pred_dict = torch.load('../yolor_csp.pt')['model']
model = Darknet('../cfg/yolor_csp.cfg', 640).cuda()
model.load_state_dict(pred_dict)
print(model)
