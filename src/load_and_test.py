# -*- coding: utf-8 -*-
import tensorflow as tf
import os

import create_models as create
from network_parameters import params
from history import History
import generators as gen
import test_functions as test
import config as conf
import cycle_clf_checkpoints as cycle_clf_chckp
import clustering_functions as clust

import parse as p



def load_cl_and_test_one_set(checkpoint_path, domain, mean_val=None, stddev_val=None, 
                             color_mode='rgb', gray2rgb=False, ms2rgb=False, 
                             source=True, ae_preprocess=False, 
                             enc_scope_name='encoder', dec_scope_name='decoder', 
                             resize=None, folder='validation', chckp_dir='', data_dir='',
                             chckp_basename='', exp_name='', chckp=''):
    if ae_preprocess == False:
        model = create.create_baseline_classifier(params, test_clf=True)
    else:
        model = create.create_classifier_with_ae(
                params, test_clf=True, enc_scope_name=enc_scope_name, 
                dec_scope_name=dec_scope_name)

    with tf.Session() as sess:
        base_dir = 'pickles/test/'
        # if not os.path.isdir(base_dir):
        #     os.mkdir(base_dir)
        pkl_dir = os.path.join(base_dir, chckp_basename, exp_name, data_dir)
        if not os.path.isdir(pkl_dir):
            os.makedirs(pkl_dir, exist_ok=True)
        losses_hist = History(path=os.path.join(pkl_dir, os.path.basename(checkpoint_path) + '.pkl'))
        # saver object
        saver = tf.train.Saver()
        # restore from the checkpoint
        saver.restore(sess, checkpoint_path)
        
        if source: 
            inp_shape = params['inp_shape_s']
        else: 
            inp_shape = params['inp_shape_t']
            
        p.threshold = data_dir
        conf.basepath, conf.inputpath_s, conf.inputpath_t, _, _ = conf.get_basepath(conf.hostname)
    
        valid_flow = gen.get_val_generator(
                inp_shape, params['otp_shape'], 
                params['num_class'], params['batch_size'], color_mode, 
                source=source, mean=mean_val, stddev=stddev_val, 
                dsn_normalize=False, folder=folder)
        
        
        test.test_one_set_val(model, sess, valid_flow, losses_hist, params, 
                              gray2rgb=gray2rgb, diff_inputs=False, 
                              source=source, resize=resize, ms2rgb=ms2rgb)
        losses_hist.save_epoch()
        
        
def load_adapter_and_test(
        checkpoint_path, domain_s, domain_t, mean_val_s=None, mean_val_t=None, 
        stddev_val_s=None, stddev_val_t=None, color_mode_s='rgb', 
        color_mode_t='rgb', gray2rgb=False, ae_preprocess=False, 
        aes_with_crit=False, butterfly_ae=False, two_fes=False, folder='validation', chckp_dir='', data_dir='',
        chckp_basename='', exp_name='', chckp=''):
    if ae_preprocess == True or aes_with_crit == True:
        enc_scope_name_s, dec_scope_name_s = 'encoder', 'decoder'
        if aes_with_crit:
            enc_scope_name_s += '_1'
            dec_scope_name_s += '_1'
        model = create.create_wdgrl_adapter_with_aes(
                params, test_clf=True, test_crit=False, 
                enc_scope_name_s=enc_scope_name_s, 
                dec_scope_name_s=dec_scope_name_s)
    elif butterfly_ae:
        model = create.create_wdgrl_adapter_with_butterfly_ae(
                params, test_clf=True, test_crit=False)
    elif two_fes:
        model = create.create_wdgrl_adapter_with_two_fes(params, test_clf=True, 
                                            test_crit=False)
    else:
        model = create.create_wdgrl_adapter(params, test_clf=True, 
                                            test_crit=False)
    with tf.Session() as sess:
        # base_dir = 'pickles/test'
        base_dir = os.path.join('pickles', p.exp_name, 'test')
        # if not os.path.isdir(base_dir):
        #     os.mkdir(base_dir)
        if 'pseudo' not in p.labels:
            pkl_dir = os.path.join(base_dir, chckp_basename, exp_name, data_dir)
        else:
            pkl_dir = os.path.join(chckp_basename, base_dir)
        if not os.path.isdir(pkl_dir):
            os.makedirs(pkl_dir, exist_ok=True)
        losses_hist = History(path=os.path.join(pkl_dir, os.path.basename(checkpoint_path) + '.pkl'))
#        # saver object
#        if aes_with_crit:
#            saver = tf.train.Saver([v for v in tf.trainable_variables() if 'decoder' not in v.name])
#        else:
        saver = tf.train.Saver()
        # restore from the checkpoint
        saver.restore(sess, checkpoint_path)
        
        # p.data_dir = data_dir
        p.threshold = data_dir
        # conf.basepath, conf.inputpath_s, conf.inputpath_t, _, _ = conf.get_basepath(conf.hostname)
        conf.basepath, conf.inputpath_s, conf.inputpath_t = conf.get_basepath(conf.logdirpath)
        
        valid_flow_s, valid_flow_t = gen.get_val_generators(
                params['inp_shape_s'], params['inp_shape_t'], params['otp_shape'], 
                params['num_class'], params['batch_size'], color_mode_s, 
                color_mode_t, mean_s=mean_val_s, mean_t=mean_val_t, 
                stddev_s=stddev_val_s, stddev_t=stddev_val_t, folder=folder)
        
        diff_inputs = ae_preprocess or aes_with_crit or butterfly_ae or two_fes
        test.test_two_sets_val(model, sess, valid_flow_s, valid_flow_t, 
                               losses_hist, params, gray2rgb=gray2rgb, 
                               diff_inputs=diff_inputs)
        losses_hist.save_epoch()
        
        # Calculate pseudo labels with clustering (from CAN)
        if conf.pseudo_clustering:
            
            train_flow_s_cdd, train_flow_t_cdd = \
                gen.get_class_aware_train_generators(
                    params['inp_shape_s'], params['inp_shape_t'], 
                    params['otp_shape'], params['num_class'], 
                    params['batch_size_cdd'], color_mode_s, color_mode_t)
            clust.save_init_properties_classwise(train_flow_t_cdd)
            
            clust.restore_init_properties_classwise(train_flow_t_cdd)
            clust.cluster_and_filter(
                sess, params, model, train_flow_s_test, 
                train_flow_t_test, clust.clustering, {},
                train_flow_s_cdd, train_flow_t_cdd)
            
            clustered_target_samples = spherical_k_means(
                sess, params, model, train_flow_s_cluster, train_flow_t_cluster, 
                clustering, history)
                
                # Pseudo-label accuracy
            clust_predict_correct = np.sum(samples['label'] == flow_train.classes[samples['data_ind']])
            clust_acc = clust_predict_correct / len(samples['data_path'])
            print("Clustering accuracy: %.6f" % clust_acc)
        
        
## Test RGB classifier with AE on one sets  
#checkpoint_path = 'checkpoints/wdgrl-baseline-03-cl-2019-06-11-6156'
#inputpath = inputpath_s
#load_and_test_one_set(checkpoint_path, inputpath, mean_val=means[staining_s], stddev_val=stddevs[staining_s], 
#                      color_mode='rgb', cl=True, ae_preprocess=True, enc_scope_name='encoder_2', dec_scope_name='decoder_2')


# ## Test RGB checkpoint on grayscale
# #cl_checkpoints = [
# #            'classifier-ae-02-rgb-2020-5-4_1-5-53-163',
# #            'classifier-ae-02-rgb-2020-5-8_22-59-12-815',
# #            'classifier-ae-02-rgb-2020-5-9_1-23-53-1141',
# #            'classifier-ae-02-rgb-2020-5-9_3-48-44-2608',
# #            'classifier-ae-02-rgb-2020-5-9_6-12-34-1630',
# #            'classifier-ae-02-rgb-2020-5-9_14-6-59-3749',
# #            'classifier-ae-02-rgb-2020-5-9_16-36-1-4564',
# #            'classifier-ae-02-rgb-2020-5-9_19-2-34-2934',
# #            'classifier-ae-02-rgb-2020-5-9_21-34-13-4890',
# #            'classifier-ae-02-rgb-2020-5-10_0-4-20-5216',
# #            'classifier-ae-02-rgb-2020-5-10_2-28-49-1304',
# #            'classifier-ae-02-rgb-2020-5-10_4-53-57-1304',
# #            'classifier-ae-02-rgb-2020-5-10_7-18-43-1141',
# #            'classifier-ae-02-rgb-2020-5-10_9-42-21-1793',
# #            'classifier-ae-02-rgb-2020-5-10_12-6-55-1630',
# #            'classifier-ae-02-rgb-2020-5-10_14-31-17-2119',
# #            'classifier-ae-02-rgb-2020-5-10_16-56-5-3423',
# #            'classifier-ae-02-rgb-2020-5-10_19-20-19-2445',
# #            'classifier-ae-02-rgb-2020-5-10_21-44-4-1630',
# #            'classifier-ae-02-rgb-2020-5-11_0-15-58-4564',
# #            'classifier-ae-02-rgb-2020-5-11_2-48-26-326',
# #            'classifier-ae-02-rgb-2020-5-11_5-15-2-4564',
# #            'classifier-ae-02-rgb-2020-5-11_7-40-10-489',
# #            'classifier-ae-02-rgb-2020-5-11_10-4-51-3749',
# #            'classifier-ae-02-rgb-2020-5-11_12-29-26-2608']
# #
# #
# #domain = '03_greyscale'
# #for chckp in cl_checkpoints:
# #    checkpoint_path = os.path.join("checkpoints", chckp)
# #    print(chckp)
# #    print(domain)
# #    mean_val = conf.means[domain]
# #    stddev_val = conf.stddevs[domain]
# #    load_cl_and_test_one_set(checkpoint_path, domain, mean_val=mean_val, 
# #              stddev_val=stddev_val, color_mode='greyscale',
# #              gray2rgb=True, source=True, ae_preprocess=False)


if 'pseudo' in p.labels:
    if 'semi' in p.labels:
        thresholds = ['6_25', '2_5', '1_25']
    else:
        thresholds = ['0']
    data_dirs = thresholds
    
    domain_s = p.source
    domain_t = p.target
    if p.source == 'resisc':
        color_mode_s, color_mode_t= 'rgb', p.color_mode
    else:
        color_mode_s, color_mode_t= p.color_mode , 'rgb'
    
    for threshold in thresholds:
        print(threshold)
        if 'semi' in p.labels:
            cyclegans = cycle_clf_chckp.cyclegan_ss_checkpoint_times[threshold]
        else:
            cyclegans = cycle_clf_chckp.cyclegan_checkpoint_times
        for cyclegan in cyclegans:
            cycle_dir = 'cyclegan-' + cyclegan
            cycle_dir_path = os.path.join(conf.base_outdir, cycle_dir, 'ckpt-49')
            rep_idx = int(cyclegan[-1])
            clf_dir = cycle_clf_chckp.get_clf_checkpoint_by_rep_idx(rep_idx)
            clf_dir_path = os.path.join(cycle_dir_path, clf_dir)
            sshida_dir_path = os.path.join(clf_dir_path, "sshida" + p.filt_threshold)
            chckp = "adapter-2_fes_1_clf_semi-"+str(p.num_class)+"000"
            checkpoint_path = os.path.join(sshida_dir_path, "checkpoints")
            if p.exp_name in os.listdir(checkpoint_path):
                checkpoint_path = os.path.join(checkpoint_path, p.exp_name)
            checkpoint_path = os.path.join(checkpoint_path, chckp)
            print(checkpoint_path)
            print(domain_s, domain_t)
            mean_val_s = conf.means[domain_s]
            stddev_val_s = conf.stddevs[domain_s]
            mean_val_t = conf.means[domain_t]
            stddev_val_t = conf.stddevs[domain_t]
            load_adapter_and_test(checkpoint_path, domain_s, domain_t, 
                                  mean_val_s=mean_val_s, mean_val_t=mean_val_t,
                                  stddev_val_s=stddev_val_s, stddev_val_t=stddev_val_t,
                                  color_mode_s=color_mode_s, color_mode_t=color_mode_t,
                                  two_fes=True, folder='test', chckp_dir='', data_dir=threshold,
                                  chckp_basename=sshida_dir_path, exp_name='', chckp=chckp)

else:
    if p.usecase == '2_fes_1_clf':
        csv_folder = 'analyze/uda'
        csv_file = os.path.join(csv_folder, p.exp_name + '.csv')
        # csv_file = 'analyze/uda/adapter-2_fes_1_clf-'+p.source+'-'+p.target+'-'+p.color_mode+'-aug-no_batch_norm-cl_loss_s-'+str(p.num_class)+'_classes.csv'
    # csv_file = 'analyze/8_classes/classifier-resisc-rgb-aug-no_batch_norm-cl_loss-8_classes.csv'
    # csv_file = p.csv_file
    elif p.usecase ==  '2_fes_1_clf_semi':
        if not p.with_critic:
            dirname = 'ablation_wo_dc'
            if p.sep:
                dirname += '_sep_fes'
        elif p.sep:
            dirname = 'ablation_sep_fes'
        else:
            dirname = p.color_mode
        csv_file = 'analyze/'+dirname+'/adapter-'+p.usecase+'-'+p.source+'-'+p.target+'-'+p.color_mode+'-aug-no_batch_norm-'
        if p.use_unlabelled:
            csv_file += 'all_unlabelled-'
        csv_file += 'cl_loss_s-'+str(p.num_class)+'_classes.csv'
        # csv_file = 'analyze/'+p.color_mode+'/adapter-'+p.usecase+'-'+p.source+'-'+p.target+'-'+p.color_mode+'-aug-no_batch_norm-all_unlabelled-cl_loss_s-'+str(p.num_class)+'_classes.csv'
    else:
        csv_file = 'analyze/'+p.color_mode+'/classifier-'+p.source+'-'+p.color_mode+'-aug-no_batch_norm-cl_loss-fixed_length-'+str(p.num_class)+'_classes.csv'
        
    import pandas
    df = pandas.read_csv(csv_file, sep=';')
    data_dirs = list(df.columns)[1:]
    chckp_basename = os.path.basename(csv_file)[:-4]
    
    # if p.usecase == '2_fes_1_clf':
    #     chckp_root_folder = os.path.join("checkpoints", chckp_basename)
    # else:
    #     chckp_root_folder = os.path.join("checkpoints", chckp_basename, p.exp_name)
    # chckp_root_folder = os.path.join("checkpoints", chckp_basename, p.exp_name)
    chckp_root_folder = "checkpoints"
    # chckp_folders = [chckp_basename + '-' + perc for perc in data_dirs]
    chckp_folders = data_dirs
    all_checkpoints = [list(df[key]) for key in data_dirs]
    
    if p.usecase == 'baseline':
        # # data_dirs = sorted(['100', '50', '25', '12_5', '6_25', '2_5', '1_25'])
        # data_dirs = ['6_25', '2_5', '1_25']
        # # chckp_folders =   [
        # #   'classifier-eurosat-multispectral-aug-no_batch_norm-cl_loss-5_classes-6_25',
        # #   'classifier-eurosat-multispectral-aug-no_batch_norm-cl_loss-5_classes-2_5',
        # #   'classifier-eurosat-multispectral-aug-no_batch_norm-cl_loss-5_classes-1_25'
        # #     ]
        # chckp_folders =   [
        #   'classifier-resisc-rgb-aug-no_batch_norm-cl_loss-5_classes-6_25',
        #   'classifier-resisc-rgb-aug-no_batch_norm-cl_loss-5_classes-2_5',
        #   'classifier-resisc-rgb-aug-no_batch_norm-cl_loss-5_classes-1_25'
        #     ]
        # # all_cl_checkpoints = [
        # #     ['classifier-eurosat-multispectral-aug-no_batch_norm-cl_loss-5_classes-6_25-run_1-2021-7-3_2-0-12-500', 'classifier-eurosat-multispectral-aug-no_batch_norm-cl_loss-5_classes-6_25-run_3-2021-7-3_2-0-12-1125', 'classifier-eurosat-multispectral-aug-no_batch_norm-cl_loss-5_classes-6_25-run_2-2021-7-3_2-0-12-1125', 'classifier-eurosat-multispectral-aug-no_batch_norm-cl_loss-5_classes-6_25-run_4-2021-7-3_2-0-12-1000', 'classifier-eurosat-multispectral-aug-no_batch_norm-cl_loss-5_classes-6_25-run_5-2021-7-3_2-0-12-500'],
        # #     ['classifier-eurosat-multispectral-aug-no_batch_norm-cl_loss-5_classes-2_5-run_5-2021-7-3_2-0-21-125', 'classifier-eurosat-multispectral-aug-no_batch_norm-cl_loss-5_classes-2_5-run_4-2021-7-3_2-0-13-125', 'classifier-eurosat-multispectral-aug-no_batch_norm-cl_loss-5_classes-2_5-run_1-2021-7-3_2-0-12-125', 'classifier-eurosat-multispectral-aug-no_batch_norm-cl_loss-5_classes-2_5-run_3-2021-7-3_2-0-13-125', 'classifier-eurosat-multispectral-aug-no_batch_norm-cl_loss-5_classes-2_5-run_2-2021-7-3_2-0-13-125'],
        # #     ['classifier-eurosat-multispectral-aug-no_batch_norm-cl_loss-5_classes-1_25-run_4-2021-7-3_2-0-22-125', 'classifier-eurosat-multispectral-aug-no_batch_norm-cl_loss-5_classes-1_25-run_5-2021-7-3_2-0-22-125', 'classifier-eurosat-multispectral-aug-no_batch_norm-cl_loss-5_classes-1_25-run_1-2021-7-3_2-0-34-250', 'classifier-eurosat-multispectral-aug-no_batch_norm-cl_loss-5_classes-1_25-run_2-2021-7-3_2-0-34-250', 'classifier-eurosat-multispectral-aug-no_batch_norm-cl_loss-5_classes-1_25-run_3-2021-7-3_2-0-34-125']
        # #     ]
        # all_cl_checkpoints = [
        #     ['classifier-resisc-rgb-aug-no_batch_norm-cl_loss-5_classes-6_25-run_3-2021-7-3_1-58-54-500', 'classifier-resisc-rgb-aug-no_batch_norm-cl_loss-5_classes-6_25-run_5-2021-7-3_1-58-58-750', 'classifier-resisc-rgb-aug-no_batch_norm-cl_loss-5_classes-6_25-run_4-2021-7-3_1-58-58-1000', 'classifier-resisc-rgb-aug-no_batch_norm-cl_loss-5_classes-6_25-run_1-2021-7-3_1-58-49-750', 'classifier-resisc-rgb-aug-no_batch_norm-cl_loss-5_classes-6_25-run_2-2021-7-3_1-58-49-750'],
        #     ['classifier-resisc-rgb-aug-no_batch_norm-cl_loss-5_classes-2_5-run_1-2021-7-3_1-58-54-125', 'classifier-resisc-rgb-aug-no_batch_norm-cl_loss-5_classes-2_5-run_5-2021-7-3_1-59-0-500', 'classifier-resisc-rgb-aug-no_batch_norm-cl_loss-5_classes-2_5-run_4-2021-7-3_1-59-0-375', 'classifier-resisc-rgb-aug-no_batch_norm-cl_loss-5_classes-2_5-run_2-2021-7-3_1-59-0-125', 'classifier-resisc-rgb-aug-no_batch_norm-cl_loss-5_classes-2_5-run_3-2021-7-3_1-59-0-250'],
        #     ['classifier-resisc-rgb-aug-no_batch_norm-cl_loss-5_classes-1_25-run_1-2021-7-3_1-59-9-125', 'classifier-resisc-rgb-aug-no_batch_norm-cl_loss-5_classes-1_25-run_3-2021-7-3_1-59-9-125', 'classifier-resisc-rgb-aug-no_batch_norm-cl_loss-5_classes-1_25-run_2-2021-7-3_1-59-9-125', 'classifier-resisc-rgb-aug-no_batch_norm-cl_loss-5_classes-1_25-run_4-2021-7-3_1-59-9-125', 'classifier-resisc-rgb-aug-no_batch_norm-cl_loss-5_classes-1_25-run_5-2021-7-3_1-59-12-125']
        #     ]
        all_cl_checkpoints = all_checkpoints
        
            
        # cl_checkpoints = ['classifier-resisc-rgb-2021-4-15_23-24-26-3375', 
        #                   'classifier-resisc-rgb-2021-4-15_23-24-34-4125', 
        #                   'classifier-resisc-rgb-2021-4-15_23-24-44-4500', 
        #                   'classifier-resisc-rgb-2021-4-15_23-24-54-2250', 
        #                   'classifier-resisc-rgb-2021-4-15_23-25-6-3500']
        
        # domain = 'eurosat_ms'
        domain = p.source
        # resize = [256, 256]
        if p.source == 'resisc':
            color_mode = 'rgb'
        else:
            color_mode = p.color_mode
            
        for i, (data_dir, chckp_dir) in enumerate(zip(data_dirs, chckp_folders)):
            cl_checkpoints = all_cl_checkpoints[i]
            for chckp in cl_checkpoints:
                # checkpoint_path = os.path.join("checkpoints", chckp_dir, chckp)
                checkpoint_path = os.path.join(chckp_root_folder, chckp_dir, chckp)
                print(chckp)
                print(domain)
                mean_val = conf.means[domain]
                stddev_val = conf.stddevs[domain]
                load_cl_and_test_one_set(checkpoint_path, domain, mean_val=mean_val, 
                      stddev_val=stddev_val, color_mode=color_mode,
                      gray2rgb=False, ms2rgb=False, source=True, ae_preprocess=False, 
                      folder='train_1', chckp_dir=chckp_dir, data_dir=data_dir,
                      chckp_basename=chckp_basename, exp_name=p.exp_name, chckp=chckp)
        
    else:    
        # # # Test adapter on source and target
        
        # # data_dirs = sorted(['100', '50', '25', '12_5', '6_25', '2_5', '1_25'])
        # data_dirs = ['6_25', '2_5', '1_25']
        
        # # # Resisc - Eurosat
        # # # Choose by cl_loss_t
        # # chckp_folders =    ['adapter-two_fes-2_fes_1_clf_semi-resisc-eurosat-aug-no_batch_norm-all_unlabelled-cl_loss_t-5_classes-1_25',
        # #   'adapter-two_fes-2_fes_1_clf_semi-resisc-eurosat-aug-no_batch_norm-all_unlabelled-cl_loss_t-5_classes-2_5',
        # #   'adapter-two_fes-2_fes_1_clf_semi-resisc-eurosat-aug-no_batch_norm-all_unlabelled-cl_loss_t-5_classes-6_25']
        # # # Choose by cl_loss_s
        # # chckp_folders =     ['adapter-two_fes-2_fes_1_clf_semi-resisc-eurosat-aug-no_batch_norm-all_unlabelled-cl_loss_s-5_classes-1_25',
        # #   'adapter-two_fes-2_fes_1_clf_semi-resisc-eurosat-aug-no_batch_norm-all_unlabelled-cl_loss_s-5_classes-2_5',
        # #   'adapter-two_fes-2_fes_1_clf_semi-resisc-eurosat-aug-no_batch_norm-all_unlabelled-cl_loss_s-5_classes-6_25']
        # # # Choose by cl_loss
        # # chckp_folders =    ['adapter-two_fes-2_fes_1_clf_semi-resisc-eurosat-aug-no_batch_norm-all_unlabelled-cl_loss-5_classes-1_25',
        # #   'adapter-two_fes-2_fes_1_clf_semi-resisc-eurosat-aug-no_batch_norm-all_unlabelled-cl_loss-5_classes-2_5',
        # #   'adapter-two_fes-2_fes_1_clf_semi-resisc-eurosat-aug-no_batch_norm-all_unlabelled-cl_loss-5_classes-6_25']
        # # # Eurosat - Resisc
        # # # Choose by cl_loss_t
        # # chckp_folders = ['adapter-two_fes-2_fes_1_clf_semi-eurosat-resisc-aug-no_batch_norm-all_unlabelled-cl_loss_t-5_classes-1_25',
        # #   'adapter-two_fes-2_fes_1_clf_semi-eurosat-resisc-aug-no_batch_norm-all_unlabelled-cl_loss_t-5_classes-2_5',
        # #   'adapter-two_fes-2_fes_1_clf_semi-eurosat-resisc-aug-no_batch_norm-all_unlabelled-cl_loss_t-5_classes-6_25']
        # # # Choose by cl_loss_s
        # # chckp_folders = ['adapter-two_fes-2_fes_1_clf_semi-eurosat-resisc-aug-no_batch_norm-all_unlabelled-cl_loss_s-5_classes-1_25',
        # #   'adapter-two_fes-2_fes_1_clf_semi-eurosat-resisc-aug-no_batch_norm-all_unlabelled-cl_loss_s-5_classes-2_5',
        # #   'adapter-two_fes-2_fes_1_clf_semi-eurosat-resisc-aug-no_batch_norm-all_unlabelled-cl_loss_s-5_classes-6_25']
        # # # Choose by cl_loss
        # chckp_folders = ['adapter-two_fes-2_fes_1_clf_semi-eurosat-resisc-aug-no_batch_norm-all_unlabelled-cl_loss-5_classes-1_25',
        #  'adapter-two_fes-2_fes_1_clf_semi-eurosat-resisc-aug-no_batch_norm-all_unlabelled-cl_loss-5_classes-2_5',
        #  'adapter-two_fes-2_fes_1_clf_semi-eurosat-resisc-aug-no_batch_norm-all_unlabelled-cl_loss-5_classes-6_25',]
        
        # chckp_folders.reverse()
        
        # all_adapter_checkpoints = [
        #     ['adapter-two_fes-2_fes_1_clf_semi-eurosat-resisc-aug-no_batch_norm-all_unlabelled-cl_loss-5_classes-6_25-run_5-2021-7-3_2-9-12-5000', 'adapter-two_fes-2_fes_1_clf_semi-eurosat-resisc-aug-no_batch_norm-all_unlabelled-cl_loss-5_classes-6_25-run_3-2021-7-3_2-9-2-4125', 'adapter-two_fes-2_fes_1_clf_semi-eurosat-resisc-aug-no_batch_norm-all_unlabelled-cl_loss-5_classes-6_25-run_4-2021-7-3_2-9-2-4500', 'adapter-two_fes-2_fes_1_clf_semi-eurosat-resisc-aug-no_batch_norm-all_unlabelled-cl_loss-5_classes-6_25-run_2-2021-7-3_2-9-0-3750', 'adapter-two_fes-2_fes_1_clf_semi-eurosat-resisc-aug-no_batch_norm-all_unlabelled-cl_loss-5_classes-6_25-run_1-2021-7-3_2-9-0-4000'],
        #     ['adapter-two_fes-2_fes_1_clf_semi-eurosat-resisc-aug-no_batch_norm-all_unlabelled-cl_loss-5_classes-2_5-run_2-2021-7-3_2-9-12-2375', 'adapter-two_fes-2_fes_1_clf_semi-eurosat-resisc-aug-no_batch_norm-all_unlabelled-cl_loss-5_classes-2_5-run_3-2021-7-3_2-9-12-3125', 'adapter-two_fes-2_fes_1_clf_semi-eurosat-resisc-aug-no_batch_norm-all_unlabelled-cl_loss-5_classes-2_5-run_1-2021-7-3_2-9-12-4125', 'adapter-two_fes-2_fes_1_clf_semi-eurosat-resisc-aug-no_batch_norm-all_unlabelled-cl_loss-5_classes-2_5-run_5-2021-7-3_2-9-12-2375', 'adapter-two_fes-2_fes_1_clf_semi-eurosat-resisc-aug-no_batch_norm-all_unlabelled-cl_loss-5_classes-2_5-run_4-2021-7-3_2-9-12-3875'],
        #     ['adapter-two_fes-2_fes_1_clf_semi-eurosat-resisc-aug-no_batch_norm-all_unlabelled-cl_loss-5_classes-1_25-run_5-2021-7-3_2-9-12-4625', 'adapter-two_fes-2_fes_1_clf_semi-eurosat-resisc-aug-no_batch_norm-all_unlabelled-cl_loss-5_classes-1_25-run_4-2021-7-3_2-9-12-1375', 'adapter-two_fes-2_fes_1_clf_semi-eurosat-resisc-aug-no_batch_norm-all_unlabelled-cl_loss-5_classes-1_25-run_1-2021-7-3_2-9-12-3625', 'adapter-two_fes-2_fes_1_clf_semi-eurosat-resisc-aug-no_batch_norm-all_unlabelled-cl_loss-5_classes-1_25-run_2-2021-7-3_2-9-12-4375', 'adapter-two_fes-2_fes_1_clf_semi-eurosat-resisc-aug-no_batch_norm-all_unlabelled-cl_loss-5_classes-1_25-run_3-2021-7-3_2-9-12-3250']
        #     ]
        all_adapter_checkpoints = all_checkpoints
        
        # domain_s = 'resisc'
        # domain_t = 'eurosat_ms'
        domain_s = p.source
        domain_t = p.target
        if p.source == 'resisc':
            color_mode_s, color_mode_t= 'rgb', p.color_mode
        else:
            color_mode_s, color_mode_t= p.color_mode , 'rgb' 
        
        # if p.usecase == '2_fes_1_clf':
        #     for rep_idx in range(5):
        #         exp_name = p.exp_name + "_" + str(rep_idx+1)
        #         exp_folder = os.path.join(chckp_root_folder, exp_name)
        #         data_dir = data_dirs[0]
        #         chckp_dir = chckp_folders[0]
        #         chckp = all_adapter_checkpoints[0][rep_idx]
        #         checkpoint_path = os.path.join(exp_folder, chckp_dir, chckp)
        #         print(chckp)
        #         print(domain_s, domain_t)
        #         mean_val_s = conf.means[domain_s]
        #         stddev_val_s = conf.stddevs[domain_s]
        #         mean_val_t = conf.means[domain_t]
        #         stddev_val_t = conf.stddevs[domain_t]
        #         load_adapter_and_test(checkpoint_path, domain_s, domain_t, 
        #                               mean_val_s=mean_val_s, mean_val_t=mean_val_t,
        #                               stddev_val_s=stddev_val_s, stddev_val_t=stddev_val_t,
        #                               color_mode_s=color_mode_s, color_mode_t=color_mode_t,
        #                               two_fes=True, folder='test', chckp_dir=chckp_dir, data_dir=data_dir,
        #                               chckp_basename=chckp_basename, exp_name=exp_name, chckp=chckp)
        # else:
        for i, (data_dir, chckp_dir) in enumerate(zip(data_dirs, chckp_folders)):
            adapter_checkpoints = all_adapter_checkpoints[i]
            for chckp in adapter_checkpoints:
                # checkpoint_path = os.path.join("checkpoints", chckp_dir, chckp)
                # checkpoint_path = os.path.join(chckp_root_folder, chckp_dir, chckp)
                checkpoint_path = os.path.join(chckp_root_folder, chckp)
                print(chckp)
                print(domain_s, domain_t)
                mean_val_s = conf.means[domain_s]
                stddev_val_s = conf.stddevs[domain_s]
                mean_val_t = conf.means[domain_t]
                stddev_val_t = conf.stddevs[domain_t]
                load_adapter_and_test(checkpoint_path, domain_s, domain_t, 
                                      mean_val_s=mean_val_s, mean_val_t=mean_val_t,
                                      stddev_val_s=stddev_val_s, stddev_val_t=stddev_val_t,
                                      color_mode_s=color_mode_s, color_mode_t=color_mode_t,
                                      two_fes=True, folder='test', chckp_dir=chckp_dir, data_dir=data_dir,
                                      chckp_basename=chckp_basename, exp_name=p.exp_name, chckp=chckp)
    



        
## Test grayscale classifier with AE on one set
#checkpoint_path = 'checkpoints/wdgrl-baseline-02-cl-gray-2019-06-11-4636'
#inputpath = inputpath_s
#load_and_test_one_set(checkpoint_path, inputpath, mean_val=means[staining_s], 
#                      stddev_val=stddevs[staining_s], color_mode='greyscale',
#                      cl=True, ae_preprocess=True, enc_scope_name='encoder_2', dec_scope_name='decoder_2')
