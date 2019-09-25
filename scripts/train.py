import sys
import argparse
import json
from datasetsloader import *
from utils.normalizer import *
from models.cycleGAN_improved import QuadCycleGAN
import time

#=========================================================================
#
#  This file includes the functions to run the training for inner and
#  cross corpus experiments.
#
#=========================================================================

def _transfer(X_u, X_train, y_train, session_id, dim_g, dim_d, dim_c, lr, 
              batch_size, keep_prob, train_epochs, lambda_cyc, lambda_cls, 
              pretrain=True):
    map_dict = {"ang": 0, "hap": 1, "exc": 1, "neu": 2, "sad": 3}
    if pretrain:
        params = {}
        params['unl2ang'] = np.load(
            RESULTS+'pretrain/params_'+session_id+'_unl2ang.npz')
        params['unl2hap'] = np.load(
            RESULTS+'pretrain/params_'+session_id+'_unl2hap.npz')
        params['unl2neu'] = np.load(
            RESULTS+'pretrain/params_'+session_id+'_unl2neu.npz')
        params['unl2sad'] = np.load(
            RESULTS+'pretrain/params_'+session_id+'_unl2sad.npz')
    else:
        params = {'unl2ang': None, 'unl2hap': None, 'unl2neu': None, 
                  'unl2sad': None}
    X_source_1 = X_u
    X_target_1 = X_train[y_train==map_dict["ang"]]
    X_source_2 = X_u
    X_target_2 = X_train[y_train==map_dict["hap"]]
    X_source_3 = X_u
    X_target_3 = X_train[y_train==map_dict["neu"]]
    X_source_4 = X_u
    X_target_4 = X_train[y_train==map_dict["sad"]]
    quad_cycle_gan = QuadCycleGAN(
                X_source_1, X_target_1, "unl", "ang", 
                X_source_2, X_target_2, "unl", "hap",
                X_source_3, X_target_3, "unl", "neu",
                X_source_4, X_target_4, "unl", "sad",
                session_id, dim_g, dim_d, dim_c,
                RESULTS+'train/', GPU_CONFIG, 
                lr, keep_prob,batch_size, train_epochs,
                lambda_cyc, lambda_cls, params)
    return quad_cycle_gan.learn_representation()


def train_inner(X, y, X_u, dim_g, dim_d, dim_c, lr, batch_size, keep_prob, 
                train_epochs, lambda_cyc, lambda_cls):
    for i in range(len(X)):
        X_train, X_test, y_train, y_test = normalize_cv(X, y, i, 
                                                        norm="min_max")
        X_u_scaled = normalize(X_u, norm="min_max")
        _transfer(X_u_scaled, X_train, y_train, "sess"+str(i), dim_g, 
                  dim_d, dim_c, lr, batch_size, keep_prob, train_epochs, 
                  lambda_cyc, lambda_cls)    

        
def train_cross(X, y, X_u, dim_g, dim_d, dim_c, lr, batch_size, keep_prob, 
                train_epochs, lambda_cyc, lambda_cls):
    X_train = normalize(pd.concat(X), norm="min_max")
    y_train = pd.concat(y)
    X_u_scaled = normalize(X_u, norm="min_max")
    _transfer(X_u_scaled, X_train, y_train, "total", dim_g, dim_d, dim_c, lr, 
              batch_size, keep_prob, train_epochs, lambda_cyc, lambda_cls)    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["inner", "cross"], 
                        help="inner corpus (cross validation) \
                        or cross corpus")    
    parser.add_argument("--dim_g", default=[1582, 1000, 500, 1000, 1582], 
                        nargs='+', type=int, help="layer size of \
                        generator, e.g. 1582 1000 500 1000 1582")
    parser.add_argument("--dim_d", default=[1582, 1000, 1000, 1], 
                        nargs='+', type=int, help="layer size of \
                        discriminator, e.g. 1582 1000, 1000 4")
    parser.add_argument("--dim_c", default=[1582, 100, 100, 4], 
                        nargs='+', type=int, help="layer size of \
                        classifier, e.g. 1582 100 100 4")
    parser.add_argument("--lr", default={'d': 2e-4, 'g': 2e-4}, 
                        type=json.loads, help="""learning rate for \
                        discriminator and generator, \
                        e.g. '{"d": 2e-4, "g": 2e-4}' """)
    parser.add_argument("--batch_size", type=int, default=64, 
                        help= "batch size, e.g. 64")
    parser.add_argument("--keep_prob", 
                        default={'d': 0.8, 'g': 0.8, 'c': 0.5}, 
                        type=json.loads, help="""1-dropout for \
                        discriminator, generator and classifier, \
                        e.g. '{"d": 0.8, "g": 0.8, "c": 0.5}' """)
    parser.add_argument("--train_epochs", type=int, default=2000, 
                        help="training epoch, e.g. 2000")
    parser.add_argument("--lambda_cyc", type=int, default=5,
                        help="weight for cycle consistency loss, e.g. 5")
    parser.add_argument("--lambda_cls", type=int, default=2,
                        help="weight for classification loss, e.g. 2")
    args = parser.parse_args()
    
    X, y = load_iemocap()
    X_u = load_tedlium()
    if args.mode == 'inner':
        train_inner(X, y, X_u, args.dim_g, args.dim_d, args.dim_c, args.lr, 
                    args.batch_size, args.keep_prob, args.train_epochs, 
                    args.lambda_cyc, args.lambda_cls)
    elif args.mode == 'cross':
        train_cross(X, y, X_u, args.dim_g, args.dim_d, args.dim_c, args.lr, 
                    args.batch_size, args.keep_prob, args.train_epochs,
                    args.lambda_cyc, args.lambda_cls)
        
        
        