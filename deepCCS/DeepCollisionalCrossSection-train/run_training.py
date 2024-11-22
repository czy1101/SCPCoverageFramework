import os
import sys
import glob

if __name__ == '__main__':
 
    print('running ccs')
    train=[

        'data_final/Tests/1_train.pkl'

    ]

    test=[

        'data_final/Tests/1_test.pkl'

    ]

    for ttrain, ttest in zip(train,test):
        mdir = '_'.join(ttrain.split('_')[:-1])
        mdir = 'out/' + '_'.join(mdir.split('/')[1:]) + '/'
        # mdir = 'out/long' + '_'.join(mdir.split('/')[1:]) + '/'
        os.system('python3 bidirectional_lstm.py {} {} {}'.format(mdir, ttrain, ttest))
