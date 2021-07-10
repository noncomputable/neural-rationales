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
                model(images)

def get_class_stats(feature_map_log):
    """
    Get mean and std deviation of each feature map for each class of samples.
    """

    """
    feature_map_class_stats[stat][i][j][m] :=  stat of the mth feature map
                                               of the ith layer
                                               in the jth class
    """

    n_layers = len(feature_map_log)
    n_classes = len(feature_map_log[0])
    stats = ["mean", "std", "skew", "kurtosis"]
    feature_map_class_stats = {}
    for stat in stats:
        feature_map_class_stats[stat] = []
        for i in range(n_layers):
            n_filters = len(feature_map_log[i][0][0])
            feature_map_class_stats[stat].append(torch.empty(n_classes, n_filters))
    
    for i, layer_feature_maps in enumerate(feature_map_log):
        for j, layer_feature_maps_for_class in enumerate(layer_feature_maps):
            layer_feature_maps_for_class = torch.stack(layer_feature_maps_for_class)
            #Reduce over samples and features, only distinguish by feature maps.
            feature_map_class_stats["mean"][i][j] = torch.mean(layer_feature_maps_for_class, dim = [0,2,3])
            #Put 1D tensor of means in same number of dims as tensor of feature maps.
            broadcastable_means = feature_map_class_stats["mean"][i][j].unsqueeze(1).unsqueeze(1).unsqueeze(0)
            diffs_for_class = layer_feature_maps_for_class - broadcastable_means
            vars_for_class = torch.mean(torch.pow(diffs_for_class, 2.0), dim = [0,2,3])
            feature_map_class_stats["std"][i][j] = torch.pow(vars_for_class, .5)
            broadcastable_stds = feature_map_class_stats["std"][i][j].unsqueeze(1).unsqueeze(1).unsqueeze(0)
            zscores_for_class = diffs_for_class / broadcastable_stds
            feature_map_class_stats["skew"][i][j] = torch.mean(
                torch.pow(zscores_for_class / broadcastable_stds, 3.0),
                dim = [0,2,3]
            )
            feature_map_class_stats["kurtosis"][i][j] = torch.mean(
                torch.pow(zscores_for_class / broadcastable_stds, 4.0),
                dim = [0,2,3]
            )

    return feature_map_class_stats

def get_class_fitnesses(feature_map_class_stats, fitness_func):
    """
    Get fitness of each feature map for each class of samples.
    A feature map's fitness for a class will be its expectation minus the max expectation of the *other* classes.
    Also get the fitness, conv idx, and filter idx of the most fit feature map for each class, i.e. its
    rationalizing feature map.

    Args:
    feature_map_class_stats - See get_class_stats.
    fitness_func - F: (layer_feature_map_class_stats, class_idx, other_class_idxs) -> layer_fitnesses_for_class 
    """
    
    """
    feature_map_class_fitness[i][j][m] := fitness of the mth feature map
                                          of the ith layer
                                          for the jth class
    """
    feature_map_class_fitnesses = []

    #most_fit_feature_map_for_class[j] := (fitness, layer index, feature map index)
    num_classes = len(feature_map_class_stats["mean"][0])
    most_fit_feature_map_for_class = torch.full((num_classes, 3), -1)

    for layer_i in range(len(feature_map_class_stats)):
        layer_all_class_fitnesses = []
        layer_feature_map_class_stats = {stat: feature_map_class_stats[stat][layer_i]
                                         for stat in feature_map_class_stats}
        for class_j, (fitness, _, _) in enumerate(most_fit_feature_map_for_class):
            other_class_idxs = [k for k in range(len(most_fit_feature_map_for_class)) if k != class_j]
            layer_class_fitnesses = fitness_func(layer_feature_map_class_stats, class_j, other_class_idxs)
            layer_all_class_fitnesses.append(layer_class_fitnesses)
            most_fit_feature_map_in_layer_for_class = torch.max(layer_class_fitnesses, dim = 0)
            if most_fit_feature_map_in_layer_for_class.values > fitness:
                most_fit_feature_map_for_class[class_j, 0] = most_fit_feature_map_in_layer_for_class.values
                most_fit_feature_map_for_class[class_j, 1] = layer_i
                most_fit_feature_map_for_class[class_j, 2] = most_fit_feature_map_in_layer_for_class.indices
        feature_map_class_fitnesses.append(torch.stack(layer_all_class_fitnesses))

    return feature_map_class_fitnesses, most_fit_feature_map_for_class

class FitnessFunc:
    """
    Methods to compute fitness for each feature map in a layer for a certain class,
    as a function of statistics of those feature maps for each class.
    
    Args:
    layer_feature_map_class_stats - Dict where ...[stat][j][m] :=
        stat of the mth feature map for the jth class of samples.
    class_idx - Index of the class to compute fitnesses for.
    other_class_idxs - Indices of all the other classes the fitness will be compared to.
    """

    @staticmethod
    def top_expectation_gap_fitness(layer_feature_map_class_stats, class_idx, other_class_idxs):
        """
        Get the difference between the expectations of feature maps in a layer for a given class
        and the max expectations of the feature maps in that layer among all the other classes.
        i.e. how much bigger the given class's expectation is than the biggest expectation among the others.
        """

        #Get the max expectation for any feature map in this layer for every class besides j.
        max_expectations_for_other_classes = torch.max(
            layer_feature_map_class_stats["mean"][other_class_idxs], dim = 0
        ).values
        layer_class_fitnesses = layer_feature_map_class_stats["mean"][class_idx] - max_expectations_for_other_classes

        return layer_class_fitnesses

    @staticmethod
    def mean_deviation_fitness(layer_feature_map_class_stats, class_idx, other_class_idxs):
        #Get the mean expectation for any feature map in this layer for every class besides j.
        mean_expectations_for_other_classes = torch.mean(
            layer_feature_map_class_stats["mean"][other_class_idxs], dim = 0
        )
        layer_class_fitnesses = (layer_feature_map_class_stats["mean"][class_idx] - mean_expectations_for_other_classes)**2
        
        return layer_class_fitnesses

def analyse_log(feature_map_log, fitness_func, save_dir = None):
    """
    Analyse a feature map log to see the relationship between different
    classes and feature maps.
    
    Args:
    feature_map_log - A list to record feature maps in.
        feature_map_log[i][j][k][m] := mth feature map
                                       of the ith layer
                                       for data sample k
                                       of class j
    fitness_func - See FitnessFunc docs.
    save_dir - 
    
    Returns: See docs for get_class_stats and get_class_fitnesses.
    """
    
    feature_map_class_stats = get_class_stats(feature_map_log)
    feature_map_class_fitnesses, most_fit_feature_map_for_class = (
        get_class_fitnesses(feature_map_class_stats, fitness_func)
    )

    if save_dir is not None:
        torch.save(feature_map_class_stats, f"{save_dir}/class_stats.pt")
        torch.save(feature_map_class_fitnesses, f"{save_dir}/class_fitnesses.pt")
        torch.save(most_fit_feature_map_for_class, f"{save_dir}/class_most_fit.pt")

    return (feature_map_class_stats, feature_map_class_fitnesses,
            most_fit_feature_map_for_class)

def load_analysis(save_dir):
    class_stats = torch.load(f"{save_dir}/class_stats.pt")
    class_fitnesses = torch.load(f"{save_dir}/class_fitnesses.pt")
    class_most_fit_feature_map = torch.load(f"{save_dir}/class_most_fit.pt")

    return class_stats, class_fitnesses, class_most_fit_feature_map

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
