import time
import argparse
from datasetsloader import *
from utils.normalizer import *
from evaluation.evaluator import Evaluator
from evaluation.classifiers.ann import ANN
from evaluation.vis import *
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
np.warnings.filterwarnings('ignore')

#=========================================================================
#
#  This file implements the classification for cross-corpus evaluation. 
#  It trains three models: on only real data, on only synthetic data and 
#  on the combination of both and tests the models on the same test dataset
#  which is another labeled emotional dataset other than a part of training
#  dataset.
#
#=========================================================================


def time_converter(start, end):
    """
    Outputs time duration in a pretty way
    @param start: Start time
    @param end: End time
    @return: Formatted representation of time duration
    """
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return ("time used: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),
                                                       int(minutes),seconds))

def split_dev_test(data, labels, dev_portion):
    """
    Split development and test set
    @data: The entire cross-corpus dataset, here MSP
    @labels: Labels of the entire cross-corpus dataset
    @dev_portion: Portion of the development set in the entire cross-corpus 
    dataset, this value ranges between 0 and 1
    @return: List of indicators whether an element in the entire cross-corpus
    dataset belongs to the test set, e.g. [True, True, False, ...]
    """
    def split_single_class(data_, dev_portion_=dev_portion):
        m = len(data_)
        split = int(np.round(m * (1 - dev_portion_)))
        data_shuffle = data_.sample(frac=1)
        return data_shuffle[:split]
    ang_test_x = split_single_class(data[labels == 0])
    hap_test_x = split_single_class(data[labels == 1])
    neu_test_x = split_single_class(data[labels == 2])
    sad_test_x = split_single_class(data[labels == 3])
    test = pd.concat([ang_test_x, hap_test_x, neu_test_x, sad_test_x])
    return data.index.isin(test.index)


def test(cls, X_train, y_train, X_test, y_test, X_syn, y_syn, train_on, 
         title, repeated, X_dev=None, y_dev=None, save_path=None):
    """
    Train and test classification model for cross-corpus evaluation. 
    If development set is not given, the entire cross-corpus dataset will
    be used as test set.
    @param cls: Initialized classifier given hyperparameters
    @param X_train: Real training data
    @param y_train: Labels for real training data
    @param X_test: Test data
    @param y_test: Labels for test data
    @param X_syn: Synthetic training data
    @param y_syn: Labels for synthetic training data
    @param train_on: Which training data will be used
    @param title: Title for generated image of confusion matrix 
    @param repeated: Number of repeated times for each classification
    @param X_dev: Development data
    @param y_dev: Labels for development data
    @save_path: File path to save the generated image
    @return: Mean and standard deviation of test recall
    """
    recalls_dev = []
    recalls_test = []
    cms_test = []
    start = time.time()    
    for j in range(repeated):
        if X_dev is not None:
            cls.fit(X_train, y_train, X_dev, y_dev)  
        else:
            cls.fit(X_train, y_train, X_test, y_test)
        print("Run {}".format(j))
        if X_dev is not None:
            y_pred_dev = cls.predict(X_dev)
            recall_dev = cls.get_recall(y_pred_dev, y_dev)
            recalls_dev.append(recall_dev)          
        y_pred_test = cls.predict(X_test)            
        recall_test = cls.get_recall(y_pred_test, y_test)            
        recalls_test.append(recall_test)
        cm_test = cls.get_confusion_matrix(y_pred_test, y_test)
        cms_test.append(cm_test)
    end = time.time()   
    print("time used: {} s".format(time_converter(start, end)))
    if X_dev is not None:
        mean_recall_dev = np.mean(recalls_dev)
        std_recall_dev = np.std(recalls_dev)
        print("cross-corpus dev - mean: {}, std: {}".format(
            mean_recall_dev, std_recall_dev))
    mean_recall_test = np.mean(recalls_test)
    std_recall_test = np.std(recalls_test)
    print("cross-corpus test - mean: {}, std: {}".format(
        mean_recall_test, std_recall_test))    
    mean_cm_test = np.mean(cms_test, axis=0)    
    plot_confusion_matrix(mean_cm_test, title=title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print("Successfully generated {}".format(save_path))
    else:
        plt.show() 
    plt.close()
    return mean_recall_test, std_recall_test


if __name__ == '__main__':       
    parser = argparse.ArgumentParser()
    parser.add_argument("train_on", choices=['real', 'syn', 'real+syn'], 
                        help="training dataset")
    parser.add_argument("syn_data", choices=[0, 1, 2, 3],
                        type=int, help="lambda_cls for synthetic dataset")
    parser.add_argument("--dim_c", nargs='+', type=int, 
                        help="layer size of classifier, e.g. 1582 100 100 4")
    parser.add_argument("--lr", default=1e-5, type=float, 
                        help="learning rate, e.g. 0.00001")
    parser.add_argument("--batch_size", type=int,
                        help="batch size, e.g. 64")
    parser.add_argument("--keep_prob", type=float,
                        help="1-dropout, e.g. 0.8")
    parser.add_argument("--train_epochs", type=int, 
                        help="training epoch, e.g. 100")
    parser.add_argument("--repeated", default=5, type=int, 
                        help="number of repeated times for \
                        classification, e.g. 5")
    parser.add_argument("--f_names", nargs='+', 
                        help="selected features")
    parser.add_argument('--dev_portion', type=float, default=0., 
                        choices=[f/10. for f in range(10)],
                        help="portion of development set")
    args = parser.parse_args()
    X, y = load_iemocap()
    X_train = pd.concat(X)
    y_train = pd.concat(y)
    X_train_scaled = normalize(X_train)
    
    X_msp, y_msp = load_msp()
    X_msp_scaled = normalize(X_msp)

    X_syn, y_syn = load_synthetic_data(args.syn_data)
    X_syn_scaled = normalize(X_syn)
    
    if args.dev_portion > 0:
        test_index = split_dev_test(X_msp_scaled, y_msp, args.dev_portion)        
        X_test_scaled = X_msp_scaled.loc[test_index]
        y_test = y_msp.loc[test_index]
        X_dev_scaled = X_msp_scaled.loc[~test_index]
        y_dev = y_msp.loc[~test_index] 
    else:
        X_test_scaled = X_msp_scaled
        y_test = y_msp
        X_dev_scaled = None
        y_dev = None
    
    if args.f_names is not None:
        X_train_scaled = X_train_scaled[args.f_names]
        X_test_scaled = X_test_scaled[args.f_names]
        X_syn_scaled = X_syn_scaled[args.f_names]
        d_input = len(args.f_names)
        if args.dev_portion > 0:
            X_dev_scaled = X_dev_scaled[args.f_names]
    else:
        d_input = 1582
    
    dim_c = {'real': [d_input, 50, 50, 4],
             'syn': [d_input, 200, 200, 4],
             'real+syn': [d_input, 200, 200, 4]}
    lr = {'real': 1e-5, 'syn': 5e-6, 'real+syn': 5e-6}
    batch_size = {'real': 64, 'syn': 256, 'real+syn': 256}
    keep_prob = {'real': 0.8, 'syn': 0.5, 'real+syn': 0.5}
    train_epochs = {'real': 70, 'syn': 5, 'real+syn': 30}
    n ={'real': 10, 'syn': 1, 'real+syn': 5}
    
    if args.dim_c is None:
        args.dim_c = dim_c[args.train_on]
    if args.lr is None:
        args.lr = lr[args.train_on]
    if args.batch_size is None:
        args.batch_size = batch_size[args.train_on]
    if args.keep_prob is None:
        args.keep_prob = keep_prob[args.train_on]
    if args.train_epochs is None:
        args.train_epochs = train_epochs[args.train_on]
        
    cls = ANN(dim_c=args.dim_c, 
              save_path=RESULTS+'cls/', 
              GPU_CONFIG=GPU_CONFIG, 
              session_id='total',
              lr=args.lr,
              batch_size=args.batch_size,
              keep_prob=args.keep_prob,
              train_epochs=args.train_epochs,
              n=n[args.train_on])
    
    title = "{}".format(args.train_on)
    with open(RESULTS+'cls/'+title+'.txt', 'a+') as file:
        mean_recall, std_recall = test(cls, X_train_scaled, y_train, 
                                       X_test_scaled, y_test, X_syn_scaled, 
                                       y_syn, args.train_on, title, 
                                       args.repeated, X_dev_scaled, y_dev,
                                       save_path=ROOT+'master-thesis/images/cm_'+ \
                                       args.train_on+'_total_cls'+ \
                                       str(args.syn_data)+'.png')
        file.write("{}\t{}\t{}\n".format(d_input, mean_recall, std_recall))
