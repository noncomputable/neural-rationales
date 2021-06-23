import torch, copy
from collections import defaultdict

class Identity(torch.nn.Module):
    """
    Pruned ops/layers/modules are replaced with this.
    """

    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x

identity = Identity()

def prune_ops_after_target(module, target_conv_idx, conv_idx = None, level = 0):
    """
    Prune all layers and operations in the module after the target_conv_idxth
    convolution.

    Args:
    module - The module to prune.
    target_conv_idx - The idx of the conv in the seq of all convs in the module.
    conv_idx - The idx of the last conv found in the traversal.
    level - The depth of the traversal.
    """

    conv_idx = [-1] if conv_idx is None else conv_idx
    for child_name, child in module.named_children():
        #Do this before incrementing conv_idx to avoid pruning the target conv.
        if conv_idx[0] >= target_conv_idx:
            if isinstance(child_name, str):
                setattr(module, child_name, identity)
            else:
                module[child_name] = identity

        if isinstance(child, torch.nn.Conv2d):
            conv_idx[0] += 1
        elif conv_idx[0] < target_conv_idx and isinstance(child, torch.nn.Module):
            prune_ops_after_target(child, target_conv_idx, conv_idx, level + 1)

    if level == 0:
        return module

def get_rationale(model, conv_idx):
    """
    Extract a rationale from the network for the feature map of the conv_idxth convolution.
    """
    
    model = copy.deepcopy(model)
    rationale = prune_ops_after_target(model, conv_idx)
    return rationale

def get_rationale_feature_map_recorder(feature_map_log, class_idx, class_feat_map_idx):
    """
    Get a forward hook function to record the output feature maps of 
    class's rationales in the feature_map_log.

    Args:
    feature_map_log - A list to record class rationale feature maps in.
        feature_map_log[i] := feature map of class i.
    class_feat_map_idx - Index of class' feature map in the output of its rationale's final conv.
    """

    def record_feature_maps(module, input, output):
        feature_maps = output[:, class_feat_map_idx].detach()
        feature_map_log[class_idx] = feature_maps

    return record_feature_maps

def register_rationale_hooks(model, feature_map_log,
                             conv_idx_to_class_idxs, class_feat_map_idxs):
    """
    Register hooks to record the feature maps on the convs for each class' rationale.
    """
    
    conv_idx = -1
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_idx += 1
            if conv_idx in conv_idx_to_class_idxs:
                for class_idx in conv_idx_to_class_idxs[conv_idx]:
                    class_feat_map_idx = class_feat_map_idxs[class_idx]
                    feature_map_recorder = get_rationale_feature_map_recorder(
                        feature_map_log, class_idx, class_feat_map_idx
                    )
                    module.register_forward_hook(feature_map_recorder)

def rationale_to_classifier_network(rationale, class_feature_map_idxs):
    """
    Expand a multi-class rationale into a classifier, using the averages of
    each class's feature map as logits.
    """

    class ClassifierNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rationale = rationale
            self.class_feature_map_log = [None for _ in range(len(class_feature_map_idxs))]
            
            _, class_feat_map_conv_idxs, class_feat_map_idxs = class_feature_map_idxs.T
            conv_idx_to_class_idxs = defaultdict(lambda: list())
            for class_idx, class_feat_map_conv_idx in enumerate(class_feat_map_conv_idxs):
                conv_idx_to_class_idxs[class_feat_map_conv_idx.item()].append(class_idx)
            register_rationale_hooks(self.rationale, self.class_feature_map_log,
                                     conv_idx_to_class_idxs, class_feat_map_idxs)
    
        def forward(self, x):
            self.rationale(x)
            
            class_logits = torch.stack(
                [torch.mean(feature_maps, dim=[1,2])
                 for feature_maps
                 in self.class_feature_map_log]
            ).T
            
            self.class_feature_map_log = [None for _ in range(len(class_feature_map_idxs))]
            
            return class_logits
            
    return ClassifierNetwork
