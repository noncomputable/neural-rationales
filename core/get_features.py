import torch, json

def get_feature_map_recorder(feature_map_log):
    """
    Get a forward hook function to record all feature maps produced
    by a network in the feature_map_log.

    Args:
    feature_map_log - A list to record feature maps in.
        feature_map_log[i][j][k][m] := mth feature map
                                       of the ith layer
                                       for data sample k
                                       of class j
    """

    def record_feature_map(module, input, output):
        feature_map_log[module.id][-1].extend(list(output))

    return record_feature_map

def register_hooks(module, hooked_layers, feature_map_recorder, feature_map_log):
    """
    Register a forward hook on a model that records the outputs (feature maps)
    of convolutional layers and stores layer names in hooked_layers.

    Args:
    hooked_layers - A list to record strings describing each layer in.
    feature_map_recorder - A torch.nn.module forward hook function to record feature maps.
    feature_map_log - 
    """

    for child in module.children():
        if isinstance(child, torch.nn.Conv2d):
            child.id = len(hooked_layers)
            hooked_layers.append(str(child))
            child.register_forward_hook(feature_map_recorder)
            feature_map_log.append([])
        elif isinstance(child, torch.nn.Module):
            register_hooks(child, hooked_layers, feature_map_recorder, feature_map_log)

def forward_pass(model, class_dataloaders, feature_map_log):
    """
    Pass every sample of every class through the model to log
    all their feature maps. Accepts separate dataloaders for
    each class so their feature maps can be logged in separate
    sublists.
    """

    model.eval()
    with torch.no_grad():
         for i, class_dataloader in enumerate(class_dataloaders):
            #Add a list for feature maps for this class at each layer.
            for layer in feature_map_log:
                 layer.append([])
                    
            for j, batch in enumerate(class_dataloader):
                if j % 10 == 0: print(j)
                images, class_idxs = batch
                logits = model(images)

def get_class_expectations(feature_map_log):
    """
    Get average value of each feature map for each class of samples.
    """

    """
    feature_map_class_expectations[i][j][m] := ave values of the mth feature map
                                               of the ith layer
                                               in the jth class
    """ 
    feature_map_class_expectations = []

    for i, layer_feature_maps in enumerate(feature_map_log):
        layer = feature_map_log[i]
        feature_map_class_expectations.append([])
        for j, layer_feature_maps_for_class in enumerate(layer_feature_maps):
            layer_feature_maps_for_class = torch.stack(layer_feature_maps_for_class)
            #Reduce over samples and features, only distinguish by feature maps.
            feature_map_expectations_for_class = torch.mean(layer_feature_maps_for_class, dim = [0,2,3])
            feature_map_class_expectations[-1].append(feature_map_expectations_for_class)
        feature_map_class_expectations[-1] = torch.stack(feature_map_class_expectations[-1])

    return feature_map_class_expectations

def get_class_fitnesses(feature_map_class_expectations):
    """
    Get fitness of each feature map for each class of samples.
    A feature map's fitness for a class will be its expectation minus the max expectation of the *other* classes.
    """
    
    """
    feature_map_class_fitness[i][j][m] := fitness of the mth feature map
                                          of the ith layer
                                          for the jth class
    """
    feature_map_class_fitnesses = []

    #most_fit_feature_map_for_class[j] := (fitness, layer index, feature map index)
    most_fit_feature_map_for_class = torch.full((len(feature_map_class_expectations[0]), 3), -1)

    for i, layer_feature_map_class_expectations in enumerate(feature_map_class_expectations):
        layer_all_class_fitnesses = []
        for class_j, (fitness, _, _) in enumerate(most_fit_feature_map_for_class):
            other_class_idxs = [i for i in range(len(most_fit_feature_map_for_class)) if i != class_j]
            max_expectations_for_other_classes = torch.max(
                layer_feature_map_class_expectations[other_class_idxs], dim = 0
            ).values
            layer_class_fitnesses = layer_feature_map_class_expectations[class_j] - max_expectations_for_other_classes
            layer_all_class_fitnesses.append(layer_class_fitnesses)
            most_fit_feature_map_in_layer_for_class = torch.max(layer_class_fitnesses, dim = 0)
            if most_fit_feature_map_in_layer_for_class.values > fitness:
                most_fit_feature_map_for_class[class_j, 0] = most_fit_feature_map_in_layer_for_class.values
                most_fit_feature_map_for_class[class_j, 1] = i
                most_fit_feature_map_for_class[class_j, 2] = most_fit_feature_map_in_layer_for_class.indices
        feature_map_class_fitnesses.append(torch.stack(layer_all_class_fitnesses))

    return feature_map_class_fitnesses, most_fit_feature_map_for_class

def analyse_log(feature_map_log, save_dir = None):
    """
    Analyse a feature map log to see the relationship between different
    classes and feature maps.
    
    Args:
    feature_map_log - A list to record feature maps in.
        feature_map_log[i][j][k][m] := mth feature map
                                       of the ith layer
                                       for data sample k
                                       of class j
    
    Returns: See docs for get_class_expectations and get_class_fitnesses.
    """
    
    feature_map_class_expectations = get_class_expectations(feature_map_log)
    feature_map_class_fitnesses, most_fit_feature_map_for_class = (
        get_class_fitnesses(feature_map_class_expectations)
    )

    if save_dir is not None:
        torch.save(feature_map_class_expectations, f"{save_dir}/class_expectations.pt")
        torch.save(feature_map_class_fitnesses, f"{save_dir}/class_fitnesses.pt")
        torch.save(most_fit_feature_map_for_class, f"{save_dir}/class_most_fit.pt")

    return (feature_map_class_expectations, feature_map_class_fitnesses,
            most_fit_feature_map_for_class)

def load_log(save_dir):
    class_expectations = torch.load(f"{save_dir}/class_expectations.pt")
    class_fitnesses = torch.load(f"{save_dir}/class_fitnesses.pt")
    class_most_fit_feature_map = torch.load(f"{save_dir}/class_most_fit.pt")

    return class_expectations, class_fitnesses, class_most_fit_feature_map

def save_feature_maps(save_dir, feature_map_log, hooked_layers):
    torch.save(feature_map_log, f"{save_dir}/feature_maps.pt")

    with open(f"{save_dir}/layer_list.json", "w") as layer_list:
        json.dump(hooked_layers, layer_list)

    return True

def load_feature_maps(save_dir):
    feature_map_log = torch.load(f"{save_dir}/feature_maps.pt")

    with open(f"{save_dir}/layer_list.json", "r") as layer_list:
        layer_list = json.load(layer_list)

    return feature_map_log, layer_list
