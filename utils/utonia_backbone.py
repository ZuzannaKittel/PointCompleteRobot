import torch
import torch.nn as nn
import sys

class UtoniaBackbone(nn.Module):
    def __init__(self, use_mock=False):
        super().__init__()
        on_cluster = torch.cuda.is_available()
        
        if not use_mock and on_cluster:
            try:
                from utonia.model import PointTransformerV3 as RealUtonia
                self.model = RealUtonia(
                    in_channels=3,
                    enc_channels=(36, 72, 144, 288, 576), 
                    enc_num_head=(3, 6, 12, 24, 48), 
                    enable_rpe=True,
                    enable_flash=False,
                    enc_mode=True
                )
                self.proj = nn.Linear(576, 1024)
                self.use_mock = False
                print("🚀 Real Utonia Loaded (Cluster Mode)")
            except Exception as e:
                print(f"❌ ERROR: Failed to load Real Utonia: {e}")
                sys.exit(1)
        else:
            print("⚠️ MOCK mode active")
            self.use_mock = True
            

    def forward(self, x):
        if self.use_mock:
            return torch.randn(x.shape[0], 1024, device=x.device), (x, torch.randn(x.shape[0], 1024, device=x.device))
        
        B, N, C = x.shape
        x = torch.clamp(x, min=-2.0, max=2.0)
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0)

        coord = x.view(-1, 3) 
        feat = x.view(-1, 3)  
        batch = torch.arange(B, device=x.device).repeat_interleave(N)
        
        data_dict = {"coord": coord, "feat": feat, "batch": batch, "grid_size": 0.005}
        
        # Call the underlying compiled network model
        output = self.model(data_dict)
        
        # Safely extract values out of the custom Pointcept PointTensor structure
        actual_coords = output.coord if hasattr(output, 'coord') else output['coord']
        point_features = output.feat if hasattr(output, 'feat') else output['feat']
        point_batch = output.batch if hasattr(output, 'batch') else output['batch']
        
        # Global max aggregation across batch elements
        res_list = []
        for b in range(B):
            mask = (point_batch == b)
            if mask.any():
                max_feat = point_features[mask].max(dim=0)[0]
            else:
                max_feat = torch.zeros(point_features.shape[-1], device=x.device)
            res_list.append(max_feat)
            
        res = torch.stack(res_list, dim=0)
        global_feat = self.proj(res)
        per_point_feat = self.proj(point_features) 
        
        return global_feat, (actual_coords, per_point_feat)