from datasetsloader import *
from utils.normalizer import *
from evaluation.vis import *
from collections import OrderedDict
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
np.warnings.filterwarnings('ignore')

#=========================================================================
#
#  This file compares the distribution similarity between different 
#  datasets and different emotions.       
#
#=========================================================================

def compare_distribution(datasets, save_path=None):
    """
    Plot the mean and std of the distribution of different datasets in terms 
    of some representative feature dimensions
    @param datasets: Dictionary of datasets to be plotted, e.g.
     {'source': ..., 'target': ...}
    @save_path: File path to save the generated image  
    """
    repre_features = ['pcm_loudness_sma_amean', 
                      'pcm_loudness_sma_stddev', 
                      'pcm_fftMag_mfcc_sma[4]_amean', 
                      'pcm_fftMag_mfcc_sma[4]_stddev', 
                      'logMelFreqBand_sma[0]_amean', 
                      'logMelFreqBand_sma[0]_stddev', 
                      'voicingFinalUnclipped_sma_amean', 
                      'voicingFinalUnclipped_sma_stddev', 
                      'jitterLocal_sma_amean', 
                      'jitterLocal_sma_stddev', 
                      'shimmerLocal_sma_amean', 
                      'shimmerLocal_sma_stddev']

    plot_distribution(datasets, ['source', 'target', 'syn_0', 
                                 'syn_1', 'syn_2', 'syn_3'], repre_features)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print("Successfully generated {}".format(save_path))
    else:
        plt.show()
    plt.close()
        
        
def overlap(mu_1, mu_2, sigma_1, sigma_2):
    """
    Compute the overlaps between two distributions
    @param mu_1: Mean of the first distribution
    @param mu_2: Mean of the second distribution
    @param sigma_1: Standard deviation of the first distribution
    @param sigma_2: Standard deviation of the second distribution
    @return: Overlap between them
    """
    return (mu_1 - mu_2)**2 / (sigma_1**2 + sigma_2**2)


def compare_overlaps_between_sets(datasets, emos, labels, save_path=None):
    """
    Plot the overlaps between different datasets in a heat map
    @param datasets: Dictionary of datasets to be plotted, e.g.
     {'source': ..., 'target': ...}
    @param emos: List of emotion names
    @param labels: Dictionary of labels to the datasets
    @param save_path: File path to save the generated image
    """
    df_mean = {}
    df_mean_str = {}
    df_max_str = {}
    for k, v in datasets.items():
        if k is not 'target':
            for i in range(len(emos)):
                if k is 'source':
                    diff = overlap(datasets[k].mean(), 
                                   datasets['target'].mean(),
                                   datasets[k].std(), 
                                   datasets['target'].std())
                else:
                    diff = overlap(
                        datasets[k][labels[k]==i].mean(), 
                        datasets['target'][labels['target']==i].mean(),
                        datasets[k][labels[k]==i].std(), 
                        datasets['target'][labels['target']==i].std())
                diff_max = diff.max()
                diff_mean = diff.mean()
                diff_max_str = str(round(diff_max, 2))
                diff_mean_str = str(round(diff_mean, 2))     
                pair = "{}--target".format(k)
                if pair not in df_mean.keys():
                        df_mean[pair] = [diff_mean]
                else:
                    df_mean[pair].append(diff_mean)
                if pair not in df_mean_str.keys():                       
                    df_mean_str[pair] = [diff_mean_str]
                else:
                    df_mean_str[pair].append(diff_mean_str)
                if pair not in df_max_str.keys():
                    df_max_str[pair] = [diff_max_str]
                else:                       
                    df_max_str[pair].append(diff_max_str)
    plot_overlaps(df_mean, df_mean_str, df_max_str, emos)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print("Successfully generated {}".format(save_path))
    else:
        plt.show()
    plt.close()
        
        
def compare_overlaps_between_emotions(datasets, emos, labels, 
                                      save_path=None):
    """
    Plot the overlaps between different emotions in a heat map
    @param datasets: Dictionary of datasets to be plotted, e.g.
     {'source': ..., 'target': ...}
    @param emos: List of emotion names
    @param labels: Dictionary of labels to the datasets
    @param save_path: File path to save the generated image
    """
    df_mean = {}
    df_mean_str = {}
    df_max_str = {}
    for k, v in labels.items():
        for i in range(len(emos)):
            for j in range(len(emos)):
                if j > i:
                    diff = overlap(
                        datasets[k][labels[k]==i].mean(), 
                        datasets[k][labels[k]==j].mean(), 
                        datasets[k][labels[k]==i].std(), 
                        datasets[k][labels[k]==j].std())
                    diff_max = diff.max()
                    diff_mean = diff.mean()
                    diff_max_str = str(round(diff_max, 2))
                    diff_mean_str = str(round(diff_mean, 2))        
                    pair = "{}--{}".format(emos[i], emos[j])
                    if pair not in df_mean.keys():
                        df_mean[pair] = [diff_mean]
                    else:
                        df_mean[pair].append(diff_mean)
                    if pair not in df_mean_str.keys():                       
                        df_mean_str[pair] = [diff_mean_str]
                    else:
                        df_mean_str[pair].append(diff_mean_str)
                    if pair not in df_max_str.keys():
                        df_max_str[pair] = [diff_max_str]
                    else:                       
                        df_max_str[pair].append(diff_max_str)
    plot_overlaps(df_mean, df_mean_str, df_max_str, labels.keys())
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print("Successfully generated {}".format(save_path))
    else:
        plt.show() 
    plt.close()


if __name__ == '__main__':   
    X, y = load_iemocap()
    X_scaled = normalize(pd.concat(X), norm="min_max")

    X_u = load_tedlium()
    X_u_scaled = normalize(X_u, norm="min_max")

    X_syn_0, y_syn_0 = load_synthetic_data(0)
    X_syn_1, y_syn_1 = load_synthetic_data(1)
    X_syn_2, y_syn_2 = load_synthetic_data(2)
    X_syn_3, y_syn_3 = load_synthetic_data(3)

    datasets = OrderedDict([('source', X_u_scaled), 
                            ('target', X_scaled), 
                            ('syn_0', X_syn_0), 
                            ('syn_1', X_syn_1), 
                            ('syn_2', X_syn_2), 
                            ('syn_3', X_syn_3)])
    labels = OrderedDict([('target', pd.concat(y)), 
                          ('syn_0', y_syn_0), 
                          ('syn_1', y_syn_1), 
                          ('syn_2', y_syn_2), 
                          ('syn_3', y_syn_3)])
    emos = ['ang', 'hap', 'neu', 'sad']
    compare_distribution(datasets, ROOT+'cyclegan-emotion-transfer/images/comp_dist.png')
    compare_overlaps_between_sets(
        datasets, emos, labels, 
        ROOT+'cyclegan-emotion-transfer/images/overlap_between_sets.png')
    compare_overlaps_between_emotions(
        datasets, emos, labels, 
        ROOT+'cyclegan-emotion-transfer/images/overlap_between_emotions.png')
        
