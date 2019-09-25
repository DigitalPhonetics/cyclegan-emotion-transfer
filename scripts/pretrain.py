import sys
import argparse
from datasetsloader import *
from utils.normalizer import *
from models.pretrain_cycle import Pretrain

#=========================================================================
#
#  This file includes the functions to run the pre-training for inner and
#  cross corpus experiments.
#
#=========================================================================


def _pretrain(X_u, X_train, y_train, emo_name1, emo_name2, session_id, 
              dim_g, lr, batch_size, keep_prob, train_epochs):
    map_dict = {"ang": 0, "hap": 1, "exc": 1, "neu": 2, "sad": 3}
    real_emo1 = X_u
    real_emo2 = X_train[y_train==map_dict[emo_name2]]
    pre = Pretrain(real_emo1, real_emo2, emo_name1, emo_name2, 
                   session_id, dim_g, RESULTS+'pretrain/', GPU_CONFIG, 
                   lr, batch_size, keep_prob, train_epochs)
    params = pre.train()
    

def pretrain_inner(X, y, X_u, dim_g, lr, batch_size, keep_prob, train_epochs):
    for i in range(len(X)):
        X_train, X_test, y_train, y_test = normalize_cv(X, y, i, 
                                                        norm="min_max")
        X_u_scaled = normalize(X_u, norm="min_max")
        _pretrain(X_u_scaled, X_train, y_train, "unl", "ang", "sess"+str(i), 
                  dim_g, lr, batch_size, keep_prob, train_epochs)
        _pretrain(X_u_scaled, X_train, y_train, "unl", "hap", "sess"+str(i), 
                  dim_g, lr, batch_size, keep_prob, train_epochs)
        _pretrain(X_u_scaled, X_train, y_train, "unl", "neu", "sess"+str(i), 
                  dim_g, lr, batch_size, keep_prob, train_epochs)
        _pretrain(X_u_scaled, X_train, y_train, "unl", "sad", "sess"+str(i), 
                  dim_g, lr, batch_size, keep_prob, train_epochs)

        
def pretrain_cross(X, y, X_u, dim_g, lr, batch_size, keep_prob, train_epochs):
    X_train = normalize(pd.concat(X), norm="min_max")
    y_train = pd.concat(y)
    X_u_scaled = normalize(X_u, norm="min_max")
    _pretrain(X_u_scaled, X_train, y_train, "unl", "ang", "total", 
              dim_g, lr, batch_size, keep_prob, train_epochs)
    _pretrain(X_u_scaled, X_train, y_train, "unl", "hap", "total", 
              dim_g, lr, batch_size, keep_prob, train_epochs)
    _pretrain(X_u_scaled, X_train, y_train, "unl", "neu", "total", 
              dim_g, lr, batch_size, keep_prob, train_epochs)
    _pretrain(X_u_scaled, X_train, y_train, "unl", "sad", "total", 
              dim_g, lr, batch_size, keep_prob, train_epochs)
    
    
if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["inner", "cross"], 
                        help="inner corpus (cross validation) \
                        or cross corpus")
    parser.add_argument("--dim_g", default=[1582, 1000, 500, 1000, 1582], 
                        nargs='+', type=int, help="layer size of \
                        generator, e.g. 1582 1000 500 1000 1582")
    parser.add_argument("--lr", default=2e-4, type=float, 
                        help="learning rate, e.g. 0.0002")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help= "batch size, e.g. 64")
    parser.add_argument("--keep_prob", type=float, default=0.8, 
                        help="1-dropout, e.g. 0.8")
    parser.add_argument("--train_epochs", type=int, default=10000, 
                        help="training epoch, e.g. 10000")
    args = parser.parse_args()
    
    X, y = load_iemocap()
    X_u = load_tedlium()
    if args.mode == 'inner':
        pretrain_inner(X, y, X_u, args.dim_g, args.lr, args.batch_size, 
                       args.keep_prob, args.train_epochs)
    elif args.mode == 'cross':
        pretrain_cross(X, y, X_u, args.dim_g, args.lr, args.batch_size, 
                       args.keep_prob, args.train_epochs)
        
    