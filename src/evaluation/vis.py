import numpy as np
import itertools
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns

#=========================================================================
#
#  This file includes functions for visualizing different results.             
#
#=========================================================================

def plot_losses(losses, colors, start=0):
    """
    Plot mutiple losses 
    @param losses: Dictionary of losses, e.g {'d': [...], 'g': [...]}
    @param colors: List of curve colors, e.g. ['r', 'g', 'b']
    """
    handles = []
    i = 0
    for k, v in losses.items():
        leg, = plt.plot(range(len(v))[start:], v[start:], color=colors[i], 
                        label=k)
        handles.append(leg)
        i += 1
    plt.legend(handles=handles)
    

def plot_code_vectors(code_vectors, labels):
    """
    Plot code vectors for AAE (4 component GMM)
    @params code_vectors: Output vectors in the bottleneck of autoencoders
    @params labels: Labels of the code vectors
    """
    colors = ['r', 'k', 'b', 'g']
    emotions = ['angry', 'happy', 'neutral', 'sad']
    legends = {}
    for i in range(len(colors)):
        df = code_vectors[labels==i]
        legends[emotions[i]] = plt.scatter(df['0'], df['1'], 
                                           marker='.', 
                                           color=colors[i], 
                                           label=emotions[i])
    plt.legend(handles=[legends[emo] for emo in emotions])
    
    
def compare_recalls(recalls):
    """
    List mean and std of recalls in a dataframe for the classifiers 
    trained on only real data, only synthetic data and a combination of both
    @param recalls: Dictionary of recalls, e.g. {'only_real': [], 
    'only_syn': [], 'real_plus_syn': []}
    @return A dataframe of recalls in different experiments
    """
    def bold(df):
        mean_row = abs(df - df.mean()) < 1e-5
        return ['font-weight: bold' if v else '' for v in mean_row]
    recalls = np.array(recalls)
    titles = np.array(['only_real', 'only_real', 'only_syn', 'only_syn', 
                       'real_plus_syn', 'real_plus_syn'])
    subtitles = np.array(['mean', 'std']*3)
    recalls_df = pd.DataFrame(
        data=recalls.T, 
        columns=pd.MultiIndex.from_tuples(zip(titles, subtitles)), 
        index=["session "+str(i+1) for i in range(len(recalls[0]))])
    recalls_df.loc['mean of mean'] = recalls_df.mean()
    recalls_df = recalls_df.round(3)
    return recalls_df


def compare_cms(cms, sess_id=0):
    """
    List three confusion matrices for the classifiers trained on only real data, 
    only synthetic data and a combination of both
    @param cms: Dictionary of confusion matrices, e.g. {'only_real': [], 
    'only_syn': [], 'real_plus_syn': []}
    @sess_id: Session id for cross validation
    """
    titles = ['only_real', 'only_syn', 'real_plus_syn']
    plt.rcParams['figure.figsize'] = [18, 5]
    j = 0
    for j in range(len(cms)):
        plt.subplot(1, len(titles), j+1)
        v = cms[j]
        plot_confusion_matrix(cm=v[sess_id], normalize=True, 
                              title=titles[j])
        j += 1
    plt.show()


def plot_confusion_matrix(cm, 
                          classes=['angry', 'happy', 'neutral', 'sad'],
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          labels=True):
    """
    From : https://scikit-learn.org/stable/auto_examples/model_selection/
           plot_confusion_matrix.html
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, vmin=0, vmax=0.8, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    if labels:
        plt.yticks(tick_marks, classes)
    else:
        plt.yticks([])

    fmt = '.2f' #if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 verticalalignment='center',
                 color="white" if cm[i, j] > thresh else "black")

    if labels:
        plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    
def plot_mean_and_std(X_train, X_syn, start=0, end=30):
    """
    Plot mean and std of feature distributions for training and 
    synthetic datasets
    @param X_train: Training dataset
    @param X_syn: Synethtic dataset
    @param start: Start index of feature dimension
    @param end: End index of feature dimension
    """
    index = range(end-start)
    index1 = [i-0.15 for i in index]
    index2 = [i+0.15 for i in index]
    name = X_train.columns.values[start:end]
    real_mean = X_train.mean().values[start:end]
    real_std = X_train.std().values[start:end]
    syn_mean = X_syn.mean().values[start:end]
    syn_std = X_syn.std().values[start:end]
    plt.xticks(index, name, rotation='vertical')
    real = plt.errorbar(index1, real_mean, real_std, color='b', 
                        linestyle='None', marker='^', label='real')
    syn = plt.errorbar(index2, syn_mean, syn_std, color='r', 
                       linestyle='None', marker='o', label='syn')
    plt.legend(handles=[real, syn])
    plt.show()
    
    
def plot_distribution(datasets, names, features):
    """
    Plot mean and std of mulitple datasets for arbitrary features
    @param datasets: Dictionary of datasets to be plotted, e.g. 
    {'source': ..., 'target': ...}
    @param names: List of dataset names that will be shown in legends
    @param features: List of feature dimensions
    """
    plt.rcParams['figure.figsize'] = [18, 5]
    f_indices = range(len(features))
    handles = []
    colors = ['r', 'b', 'y', 'g', 'k', 'c', 'm']
    markers = ['^', 'o', '*', '>', '<', 'v', '8']
    for i in range(len(names)):
        index = [j-0.3+0.12*i for j in f_indices]
        mean = datasets[names[i]].mean()[features].values
        std = datasets[names[i]].std()[features].values
        handle = plt.errorbar(index, mean, std, color=colors[i], 
                              linestyle='None', marker=markers[i], 
                              label=names[i])
        handles.append(handle)
    plt.xticks(index, features, ha='right', rotation=45)
    plt.legend(handles=handles)
    
    
def plot_overlaps(df_mean, df_mean_str, df_max_str, index):
    """
    Plot average and maximum of overlap values in a heat map
    @param df_mean: Average of overlap values in form of dataframe
    @param df_mean_str: String version of the average of overlap values
    @param df_max_str: String version of the maximum of overlap values,
    the string versions are used for annotation in each cell of the heat map
    """
    plt.rcParams['figure.figsize'] = [10, 5]
    df_mean = pd.DataFrame(df_mean, index=index)
    df_mean_str = pd.DataFrame(df_mean_str, index=index)
    df_max_str = pd.DataFrame(df_max_str, index=index)
    df = df_mean_str.apply(
        lambda col: col.astype(str)) + "\n(" + df_max_str + ")"        
    sns.heatmap(df_mean, cmap='coolwarm', annot=df.values, fmt='', 
                annot_kws={"size": 12})
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    
def plot_pca(X, y):
    """
    Plot 2d pricipal component analysis of a high dimensional dataset 
    with labels
    @param X: Highdimensional dataset
    @param y: Labels of the dataset
    """
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data=principalComponents, columns = ['pc1', 'pc2'], index=y.index.values)
    colors = ['r', 'k', 'b', 'g']
    emotions = ['angry', 'happy', 'neutral', 'sad']
    legends = {}
    for i in range(len(colors)):
        df = principalDf[y==i]
        legends[emotions[i]] = plt.scatter(df['pc1'], df['pc2'], marker='.', color=colors[i], label=emotions[i])
    plt.legend(handles=[legends[emo] for emo in emotions])

    
