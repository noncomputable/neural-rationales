import torch, json
from torchvision.models import vgg16
from core.generate_data import make_line_imgs
import core.dataset
import core.get_features as get_feats

make_line_imgs([0, 45, 90, 135], 60, 256, "random", "data/lines")

class_datasets = [
    core.dataset.Dataset("data/lines", "img_annotations.csv",
                         "class_names.csv", only_class_id, core.dataset.preprocess)
    for only_class_id in [0, 1, 2, 3]
]

model = vgg16(pretrained = True)

feature_map_log = []
hooked_layers = []
feature_map_recorder = get_feats.get_feature_map_recorder(feature_map_log)
get_feats.register_hooks(model, hooked_layers, feature_map_recorder, feature_map_log)

batch_size = 1
class_dataloaders = [torch.utils.data.DataLoader(class_dataset, batch_size=batch_size,
                                                 shuffle=True, num_workers=2)
                     for class_dataset in class_datasets]

get_feats.forward_pass(model, class_dataloaders, feature_map_log)

get_feats.save_feature_maps("data/lines", feature_map_log, hooked_layers)

class_expectations, class_fitnesses, most_fit_for_class = get_feats.analyse_log(
    feature_map_log, save_dir = "data/lines"
)

print(most_fit_for_class)
