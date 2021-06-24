import torch, json
from torchvision.models import vgg16
from core.generate_data import make_line_imgs
import core.dataset as dataset
import core.get_features as get_feats
import core.get_rationales as get_rats
import core.validate as val

def generate_rationales():
    make_line_imgs([0, 45, 90, 135], 60, 256, "random", "data/lines/criterion")

    class_datasets = [
        dataset.CriterionDataset("data/lines/criterion", "img_annotations.csv",
                             "class_names.csv", only_class_id, dataset.preprocess)
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

def validate_assembled_network():
    class_expectations, class_fitnesses, class_feature_map_idxs = get_feats.load_log("data/lines")
    ds = dataset.ValidationDataset("data/lines/validation", dataset.preprocess)

    n_samples = 50
    
    dl = [ds[idx] for idx in torch.randint(len(ds), (n_samples,))]
    dl = [(d[0].unsqueeze(0), d[1].unsqueeze(0)) for d in dl]

    deepest_class_feature_map_conv_idx = torch.max(class_feature_map_idxs[:,1])
    model = vgg16(pretrained = True)
    multi_class_rationale = get_rats.get_rationale(model, deepest_class_feature_map_conv_idx)
    ClassifierNetwork = get_rats.rationale_to_classifier_network(multi_class_rationale, class_feature_map_idxs)
    classifier = ClassifierNetwork()
    metrics = [val.get_ideal_vs_observed_class_expectations, val.get_max_expectation, val.get_most_extreme_observation]
    val.validate(classifier, dl, class_expectations, class_feature_map_idxs, metrics)

generate_rationales()
validate_assembled_network()

