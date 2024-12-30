import ast
import pandas as pd
import numpy as np
from medmnist import INFO
from app.utils import get_acc, get_auc, get_balanced_acc, get_cohen, get_precision

def generate_task_dict(dataset_names):
    """
    param dataset_names: iterable which holds strings naming a dataset each
    """

    task_dict = {}  # dictionary that holds the different tasks corresponding to the datasets
    # fills task_dict with the task corresponding to the dataset
    for dataset_name in dataset_names:
        task_string = INFO[dataset_name]['task']
        task_dict[dataset_name] = task_string
    return task_dict
def string_to_list(s):
    return ast.literal_eval(s)

def string_to_list_2(s):
    s_corrected = s.strip("[]").replace(" ", ",")
    return ast.literal_eval(s_corrected)

def computeMetrics(df):
    results_df = df
    dataset_names = pd.unique(results_df['dataset_name'])
    task_dict = generate_task_dict(dataset_names)

    for col in ['truth', 'probability', 'prediction']:
        results_df[col] = results_df[col].apply(string_to_list)
        results_df[col]
    # Initialize a dictionary to store metrics computed from the CSV
    metrics_from_csv = {}
    # Iterate over each dataset
    for dataset_name in dataset_names:
        # Filter the Dataframe for the current dataset
        df_dataset = results_df[results_df['dataset_name'] == dataset_name]

        # Extract truths, probabilities, and predictions
        truths = df_dataset['truth'].tolist()
        probabilities = df_dataset['probability'].tolist()
        predictions = df_dataset['prediction'].tolist()

        task = task_dict[dataset_name]

        if task == 'multi-label, binary-class':
            # For multi-label tasks, truths and probabilities are lists of lists
            y_true = np.array(truths)
            y_score = np.array(probabilities)
        else:
            # For multi-class tasks, truths and probabilities need special handling
            y_true = np.array([t[0] if isinstance(t, list) else t for t in truths]).astype(int)
            y_score = np.array(probabilities)
            predictions = np.array([p if isinstance(p, int) else p[0] for p in predictions]).astype(int)

        # Store the metrics
        metrics_from_csv[dataset_name + '_test_acc'] = get_acc(y_true, y_score, task)
        metrics_from_csv[dataset_name + '_test_AUC'] = get_auc(y_true, y_score, task)
        metrics_from_csv[dataset_name + '_test_balacc'] = get_balanced_acc(y_true, y_score, task)
        metrics_from_csv[dataset_name + '_test_co'] = get_cohen(y_true, y_score, task)
        metrics_from_csv[dataset_name + '_test_prec'] = get_precision(y_true, y_score, task)

    acc_csv = [value for key, value in metrics_from_csv.items() if '_test_acc' in key]
    auc_csv = [value for key, value in metrics_from_csv.items() if '_test_AUC' in key]
    balanced_acc_csv = [value for key, value in metrics_from_csv.items() if '_test_balacc' in key]
    co_csv = [value for key, value in metrics_from_csv.items() if '_test_co' in key]
    prec_csv = [value for key, value in metrics_from_csv.items() if '_test_prec' in key]

    # Compute the average metrics, 'weighting' each dataset equally
    average_acc = np.mean(acc_csv)
    average_auc = np.mean(auc_csv)
    average_balanced_acc = np.mean(balanced_acc_csv)
    average_co = np.mean(co_csv)
    average_prec = np.mean(prec_csv)

    # add the metrics which got averaged over each dataset to the dictionary
    metrics_from_csv['accuracy_test'] = average_acc
    metrics_from_csv['auc_test'] = average_auc
    metrics_from_csv['balacc_test'] = average_balanced_acc
    metrics_from_csv['co_test'] = average_co
    metrics_from_csv['prec_test'] = average_prec



    # return_dict = {'acc': average_acc, 'auc': average_auc, 'balacc': average_balanced_acc, 'co': average_co}
    metrics_from_csv['dataset_names'] = dataset_names
    print(metrics_from_csv)

    return metrics_from_csv


def get_task_by_partial_name(task_dict, dataset_name):
    for base_name in task_dict:
        if base_name in dataset_name:
            return task_dict[base_name]
    raise ValueError(f"No matching base dataset found for '{dataset_name}'")
def string_to_int_list(s):
    # Remove the brackets and replace spaces with commas, then evaluate
    s_corrected = s.strip("[]").replace(" ", ",")
    while ",," in s_corrected:
        s_corrected = s_corrected.replace(",,", ",")
    # Safely evaluate the string as a list
    parsed_list = ast.literal_eval(f"[{s_corrected}]")
    # Convert each element to integer (rounded if necessary)
    return [float(i) for i in parsed_list]



def computeMetrics_corrupted(df):
    results_df = df
    dataset_names = pd.unique(results_df['dataset_name'])
    datasets = [ 'breastmnist','bloodmnist', 'chestmnist', 'dermamnist', 'octmnist', 'organamnist', 'organcmnist',
                    'organsmnist', 'pathmnist', 'pneumoniamnist', 'retinamnist', 'tissuemnist']
    task_dict = generate_task_dict(datasets)
   
    for col in ['truth','probability']:
        
        results_df[col] = results_df[col].apply(string_to_list)
    print(results_df[col])
    # Initialize a dictionary to store metrics computed from the CSV
    metrics_from_csv = {}
    # Iterate over each dataset
    for dataset_name in dataset_names:
        # Filter the Dataframe for the current dataset
        df_dataset = results_df[results_df['dataset_name'] == dataset_name]
        print(dataset_name)
        # Extract truths, probabilities, and predictions
        truths = df_dataset['truth'].tolist()
        probabilities = df_dataset['probability'].tolist()
        
        #probabilities = [string_to_int_list(s) for s in probabilities_org]
        #predictions = df_dataset['prediction'].tolist()

        task = get_task_by_partial_name(task_dict, dataset_name)

        if task == 'multi-label, binary-class':
            # For multi-label tasks, truths and probabilities are lists of lists
            y_true = np.array(truths)
            y_score = np.array(probabilities)
        else:
            # For multi-class tasks, truths and probabilities need special handling
            y_true = np.array([t[0] if isinstance(t, list) else t for t in truths]).astype(int)
            y_score = np.array(probabilities)
            #predictions = np.array([p if isinstance(p, int) else p[0] for p in predictions]).astype(int)

        # Store the metrics
        metrics_from_csv[dataset_name + '_test_acc'] = get_acc(y_true, y_score, task)
        metrics_from_csv[dataset_name + '_test_AUC'] = get_auc(y_true, y_score, task)
        metrics_from_csv[dataset_name + '_test_balacc'] = get_balanced_acc(y_true, y_score, task)
        metrics_from_csv[dataset_name + '_test_co'] = get_cohen(y_true, y_score, task)
        metrics_from_csv[dataset_name + '_test_prec'] = get_precision(y_true, y_score, task)

    acc_csv = [value for key, value in metrics_from_csv.items() if '_test_acc' in key]
    auc_csv = [value for key, value in metrics_from_csv.items() if '_test_AUC' in key]
    balanced_acc_csv = [value for key, value in metrics_from_csv.items() if '_test_balacc' in key]
    co_csv = [value for key, value in metrics_from_csv.items() if '_test_co' in key]
    prec_csv = [value for key, value in metrics_from_csv.items() if '_test_prec' in key]

    # Compute the average metrics, 'weighting' each dataset equally
    average_acc = np.mean(acc_csv)
    average_auc = np.mean(auc_csv)
    average_balanced_acc = np.mean(balanced_acc_csv)
    average_co = np.mean(co_csv)
    average_prec = np.mean(prec_csv)

    # add the metrics which got averaged over each dataset to the dictionary
    metrics_from_csv['accuracy_test'] = average_acc
    metrics_from_csv['auc_test'] = average_auc
    metrics_from_csv['balacc_test'] = average_balanced_acc
    metrics_from_csv['co_test'] = average_co
    metrics_from_csv['prec_test'] = average_prec



    # return_dict = {'acc': average_acc, 'auc': average_auc, 'balacc': average_balanced_acc, 'co': average_co}
    metrics_from_csv['dataset_names'] = dataset_names
    print(metrics_from_csv)

    return metrics_from_csv