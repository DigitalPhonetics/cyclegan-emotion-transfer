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
#  This file implements the classification for inner-corpus evaluation. 
#  It trains three models: on only real data, on only synthetic data and 
#  on the combination of both and tests the models on the same test dataset
#  which is one session from IEMOCAP in cross validation.
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


def test(cls, X_train, y_train, X_test, y_test, X_syn, y_syn, train_on, 
         title, repeated, save_path=None):
    """
    Train and test classification model for inner-corpus evaluation
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
    @save_path: File path to save the generated image
    @return: Mean and standard deviation of test recall
    """
    start = time.time()
    eva = Evaluator(X_train, y_train, X_test, y_test, X_syn, y_syn, cls=cls, 
                    repeated=repeated)
    if train_on == 'real':
        mean_recall, std_recall, mean_cm = eva.real()
    elif train_on == 'syn':
        mean_recall, std_recall, mean_cm = eva.syn()
    elif train_on == 'real+syn':
        mean_recall, std_recall, mean_cm = eva.real_plus_syn()    
    end = time.time()
    print("time used: {} s".format(time_converter(start, end)))
    print("inner-corpus test - mean: {}, std: {}".format(mean_recall, 
                                                         std_recall))
    plot_confusion_matrix(mean_cm, title=title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print("Successfully generated {}".format(save_path))
    else:
        plt.show() 
    plt.close()
    return mean_recall, std_recall


if __name__ == '__main__':       
    parser = argparse.ArgumentParser()
    parser.add_argument("session_id", choices=[0, 1, 2, 3, 4],
                        type=int, help="session id for test set")
    parser.add_argument("train_on", choices=['real', 'syn', 'real+syn'], 
                        help="training dataset")
    parser.add_argument("syn_data", choices=[0, 1, 2, 3],
                        type=int, help="lambda_cls for synthetic dataset")
    parser.add_argument("-dim_c", nargs='+', type=int, 
                        help="layer size of classifier, e.g. 1582 100 100 4")
    parser.add_argument("-lr", default=1e-5, type=float, 
                        help="learning rate, e.g. 0.00001")
    parser.add_argument("-batch_size", type=int,
                        help="batch size, e.g. 64")
    parser.add_argument("-keep_prob", type=float,
                        help="1-dropout, e.g. 0.8")
    parser.add_argument("-train_epochs", type=int, 
                        help="training epoch, e.g. 100")
    parser.add_argument("-repeated", default=5, type=int, 
                        help="number of repeated times for \
                        classification, e.g. 5")
    args = parser.parse_args()
    i = args.session_id
    X, y = load_iemocap()
    X_train = pd.concat(X[:i] + X[i+1:])
    y_train = pd.concat(y[:i] + y[i+1:])
    X_train_scaled = normalize(X_train)
    X_test = X[i]
    y_test = y[i]
    X_test_scaled = normalize(X_test)
    
    X_syn, y_syn = load_synthetic_data(args.syn_data)
    X_syn_scaled = normalize(X_syn)
    
    dim_c = {'real': [1582, 100, 100, 4],
             'syn': [1582, 200, 200, 4],
             'real+syn': [1582, 1000, 1000, 4]}
    lr = {'real': 1e-5, 'syn': 1e-5, 'real+syn': 5e-6}
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
              session_id='sess'+str(i),
              lr=args.lr,
              batch_size=args.batch_size,
              keep_prob=args.keep_prob,
              train_epochs=args.train_epochs,
              n=n[args.train_on])
    
    title = "{}_sess{}".format(args.train_on, args.session_id)
    with open(RESULTS+'cls/'+title+'.txt', 'a+') as file:
        mean_recall, std_recall = test(cls, X_train_scaled, y_train, 
                                       X_test_scaled, y_test, X_syn_scaled, 
                                       y_syn, args.train_on, title, 
                                       repeated=args.repeated, 
                                       save_path=ROOT+'master-thesis/images/cm_'+ \
                                       args.train_on+'_sess'+str(i)+'_cls'+ \
                                       str(args.syn_data)+'.png')
        file.write("{}\t{}\n".format(mean_recall, std_recall))
