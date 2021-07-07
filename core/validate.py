import torch

def validate(model, dataloader, class_expectations, class_feature_map_idxs, metrics):
    num_tries = 0
    num_correct = [0 for _ in range(len(metrics))]
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, class_idx = batch
            observed_expectations = model(images)
            print("\nTrue class:", class_idx[0].item())

            for j, metric in enumerate(metrics):
                values, selected_class_idx = metric(
                    observed_expectations, class_expectations, class_feature_map_idxs
                )
                print(f"{metric.__name__} picked {selected_class_idx[0].item()}")
                num_correct_in_batch = (class_idx == selected_class_idx).sum().detach().item()
                num_correct[j] += num_correct_in_batch
            
            num_tries += len(images)

    print()
    for j, metric in enumerate(metrics):
        print(f"{metric.__name__} got {num_correct[j]} right out of {num_tries} tries. Accuracy: {num_correct[j]/num_tries}")

    return num_correct, num_tries

def get_feature_map_expectations_per_class(class_expectations, class_feature_map_idxs):
    """
    Get the prior expectation of each class's rationalizing feature map for each other class (including itself).
    """

    num_classes = len(class_feature_map_idxs)
    
    #rationale_output_expectations[i][j] := expectation of jth class's feature map for the ith class
    feature_map_expectations = torch.zeros(num_classes, num_classes)
    
    for feature_map_class_idx, (fitness, conv_idx, feature_map_idx) in enumerate(class_feature_map_idxs):
        for class_idx in range(num_classes):
            feature_map_expectations[class_idx][feature_map_class_idx] = class_expectations[conv_idx][class_idx][feature_map_idx]

    return feature_map_expectations

def get_ideal_vs_observed_class_expectations(observed_expectations, class_expectations, class_feature_map_idxs):
    """
    Get the difference between the observed feature map expectation
    and the ideal feature map expectations for each class, and
    the idx of the most likely class under this metric.
    """

    all_feature_map_expectations = get_feature_map_expectations_per_class(class_expectations, class_feature_map_idxs)
    ideal_feature_map_expectations = all_feature_map_expectations.diagonal()

    differences = torch.nn.functional.mse_loss(observed_expectations, ideal_feature_map_expectations, reduction = "none")
    most_likely_class_idx = torch.min(differences, dim = 1).indices
    
    return differences, most_likely_class_idx

def get_max_expectation(observed_expectations, class_expectations, class_feature_map_idxs):
    """
    Get the max observed expectation and the idx of the most likely class under this metric.
    """

    max_observed = torch.max(observed_expectations, dim = 1)
    max_observed_expectation, most_likely_class_idx = max_observed.values, max_observed.indices
    
    return max_observed_expectation, most_likely_class_idx

def get_most_extreme_observation(observed_expectations, class_expectations, class_feature_map_idxs):
    """
    Get the most extreme observation in terms of its deviation from the mean.
    """

    mean = torch.mean(observed_expectations, dim = 1)
    deviations = torch.nn.functional.mse_loss(mean, observed_expectations, reduction = "none")
    max_deviation = torch.max(deviations, dim = 1)

    max_deviation_val, most_likely_class_idx = max_deviation.values, max_deviation.indices

    return max_deviation_val, most_likely_class_idx
