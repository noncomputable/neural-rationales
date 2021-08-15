import torch, json, pathlib
from torchvision.models import vgg16
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset
from core.generate_data import make_line_imgs
import core.dataset as dataset
import core.get_features as get_feats
import core.get_rationales as get_rats
import core.validate as val

def get_class_datasets(n, type_ = "train"):
    ds = CIFAR10((pathlib.Path(__file__).parent/"data").resolve(), train = True, download = True, transform = dataset.preprocess)
    #class_img_idxs = [[] for _ in range(len(ds.class_to_idx))]
    class_img_idxs = [[] for _ in range(3)]
 
    for i in range(len(ds)):
        if ds[i][1] > 2:
            continue
        if all(len(img_idxs) >= 5 for img_idxs in class_img_idxs):
            break
        current_class = ds[i][1]
        class_img_idxs[current_class].append(i)

    direction = 1 if type_ == "train" else -1
    limit_idx = direction * n
    class_datasets = [Subset(ds, img_idxs[limit_idx:]) for img_idxs in class_img_idxs]
    return class_datasets

def analyse_features(): 
    class_datasets = get_class_datasets(3, "train")
    model = vgg16(pretrained = True)

    feature_map_log = []
    hooked_layers = []
    feature_map_recorder = get_feats.get_feature_map_recorder(feature_map_log)
    get_feats.register_hooks(model, hooked_layers, feature_map_recorder, feature_map_log)

    batch_size = 1
    class_dataloaders = [torch.utils.data.DataLoader(class_dataset, batch_size=batch_size,
                                                     shuffle=False, num_workers=2)
                         for class_dataset in class_datasets]

    get_feats.forward_pass(model, class_dataloaders, feature_map_log)

    get_feats.save_feature_maps("data/", feature_map_log, hooked_layers)
    
    class_stats, class_fitnesses, most_fit_for_class = get_feats.analyse_log(
        feature_map_log, get_feats.FitnessFunc.mean_deviation_fitness, save_dir = "data/"
    )

    print(most_fit_for_class)

def assemble_and_validate_network():
    class_stats, class_fitnesses, class_feature_map_idxs = get_feats.load_analysis("data/")
    class_expectations = class_stats["mean"]

    val_ds = CIFAR10((pathlib.Path(__file__).parent/"data").resolve(),
            train = True, download = True, transform = dataset.preprocess)
    n_samples = 60
    
    dl = [val_ds[idx] for idx in torch.randint(len(val_ds), (n_samples,))]
    dl = [(d[0].unsqueeze(0), torch.tensor(d[1]).unsqueeze(0)) for d in dl]

    deepest_class_feature_map_conv_idx = torch.max(class_feature_map_idxs[:,1])
    model = vgg16(pretrained = True)
    multi_class_rationale = get_rats.get_rationale(model, deepest_class_feature_map_conv_idx)
    ClassifierNetwork = get_rats.rationale_to_classifier_network(multi_class_rationale, class_feature_map_idxs)
    classifier = ClassifierNetwork()
    metrics = [val.get_ideal_vs_observed_class_expectations, val.get_max_expectation, val.get_most_extreme_observation]
    num_correct, num_tries = val.validate(classifier, dl, class_expectations, class_feature_map_idxs, metrics)

    return num_correct, num_tries, classifier

if __name__ == "__main__":
    analyse_features()
    assemble_and_validate_network()
