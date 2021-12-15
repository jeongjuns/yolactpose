import argparse
import torch

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
        
parser = argparse.ArgumentParser(description='PyTorch AlphaPose Training')

"----------------------------- General options -----------------------------"
parser.add_argument('--expID', default='default', type=str,
                    help='Experiment ID')
parser.add_argument('--datasetA', default='coco', type=str,
                    help='Dataset choice: mpii | coco')
parser.add_argument('--nThreads', default=30, type=int,
                    help='Number of data loading threads')
parser.add_argument('--debug', default=False, type=bool,
                    help='Print the debug information')
parser.add_argument('--snapshot', default=1, type=int,
                    help='How often to take a snapshot of the model (0 = never)')

"----------------------------- AlphaPose options -----------------------------"
parser.add_argument('--addDPG', default=False, type=bool,
                    help='Train with data augmentation')
parser.add_argument('--sp', default=True, action='store_true',
                    help='Use single process for pytorch')
parser.add_argument('--profile', default=False, action='store_true',
                    help='add speed profiling at screen output')

"----------------------------- Model options -----------------------------"
parser.add_argument('--netType', default='hgPRM', type=str,
                    help='Options: hgPRM | resnext')
parser.add_argument('--loadModel', default=None, type=str,
                    help='Provide full path to a previously trained model')
parser.add_argument('--Continue', default=False, type=bool,
                    help='Pick up where an experiment left off')
parser.add_argument('--nFeats', default=256, type=int,
                    help='Number of features in the hourglass')
parser.add_argument('--nClasses', default=33, type=int,
                    help='Number of output channel')
parser.add_argument('--nStack', default=4, type=int,
                    help='Number of hourglasses to stack')

"----------------------------- Hyperparameter options -----------------------------"
parser.add_argument('--fast_inference', default=True, type=bool,
                    help='Fast inference')
parser.add_argument('--use_pyranet', default=True, type=bool,
                    help='use pyranet')

"----------------------------- Hyperparameter options -----------------------------"
parser.add_argument('--LR', default=2.5e-4, type=float,
                    help='Learning rate')
parser.add_argument('--momentum', default=0, type=float,
                    help='Momentum')
parser.add_argument('--weightDecay', default=0, type=float,
                    help='Weight decay')
parser.add_argument('--crit', default='MSE', type=str,
                    help='Criterion type')
parser.add_argument('--optMethod', default='rmsprop', type=str,
                    help='Optimization method: rmsprop | sgd | nag | adadelta')


"----------------------------- Training options -----------------------------"
parser.add_argument('--nEpochs', default=50, type=int,
                    help='Number of hourglasses to stack')
parser.add_argument('--epoch', default=0, type=int,
                    help='Current epoch')
parser.add_argument('--trainBatch', default=40, type=int,
                    help='Train-batch size')
parser.add_argument('--validBatch', default=20, type=int,
                    help='Valid-batch size')
parser.add_argument('--trainIters', default=0, type=int,
                    help='Total train iters')
parser.add_argument('--valIters', default=0, type=int,
                    help='Total valid iters')
parser.add_argument('--init', default=None, type=str,
                    help='Initialization')

"----------------------------- Data options -----------------------------"
parser.add_argument('--inputResH', default=320, type=int,
                    help='Input image height')
parser.add_argument('--inputResW', default=256, type=int,
                    help='Input image width')
parser.add_argument('--outputResH', default=80, type=int,
                    help='Output heatmap height')
parser.add_argument('--outputResW', default=64, type=int,
                    help='Output heatmap width')
parser.add_argument('--scale', default=0.25, type=float,
                    help='Degree of scale augmentation')
parser.add_argument('--rotate', default=30, type=float,
                    help='Degree of rotation augmentation')
parser.add_argument('--hmGauss', default=1, type=int,
                    help='Heatmap gaussian size')

"----------------------------- PyraNet options -----------------------------"
parser.add_argument('--baseWidth', default=9, type=int,
                    help='Heatmap gaussian size')
parser.add_argument('--cardinality', default=5, type=int,
                    help='Heatmap gaussian size')
parser.add_argument('--nResidual', default=1, type=int,
                    help='Number of residual modules at each location in the pyranet')

"----------------------------- Distribution options -----------------------------"
parser.add_argument('--dist', dest='dist', type=int, default=1,
                    help='distributed training or not')
parser.add_argument('--backend', dest='backend', type=str, default='gloo',
                    help='backend for distributed training')
parser.add_argument('--port', dest='port',
                    help='port of server')

"----------------------------- Detection options -----------------------------"
parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                    default='res152')
#parser.add_argument('--indir', dest='inputpath',
#                    help='image-directory', default="")
parser.add_argument('--list', dest='inputlist',
                    help='image-list', default="")
parser.add_argument('--mode', dest='mode',
                    help='detection mode, fast/normal/accurate', default="normal")
parser.add_argument('--outdir', dest='outputpath',
                    help='output-directory', default="examples/res/")
parser.add_argument('--inp_dim', dest='inp_dim', type=str, default='608',
                    help='inpdim')
parser.add_argument('--conf', dest='confidence', type=float, default=0.05,
                    help='bounding box confidence threshold')
parser.add_argument('--nms', dest='nms_thesh', type=float, default=0.6,
                    help='bounding box nms threshold')
parser.add_argument('--save_img', default=False, action='store_true',
                    help='save result as image')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--matching', default=False, action='store_true',
                    help='use best matching')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--detbatch', type=int, default=1,
                    help='detection batch size')
parser.add_argument('--posebatch', type=int, default=80,
                    help='pose estimation maximum batch size')

"----------------------------- Video options -----------------------------"
parser.add_argument('--videoA', dest='video',
                    help='video-name', default="")
parser.add_argument('--webcam', dest='webcam', type=str,
                    help='webcam number', default='0')
parser.add_argument('--save_video', dest='save_video',
                    help='whether to save rendered video', default=False, action='store_true')
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)
                    
                    
                    
                    
                    
parser.add_argument('--trained_model',
                        default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
parser.add_argument('--color',  type=str,
                        help='color you want detect')
parser.add_argument('--top_k', default=5, type=int,
                        help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                        help='Whether compute NMS cross-class or per-class.')
parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')
parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
parser.add_argument('--display', dest='display', action='store_true',
                       help='Display qualitative results instead of quantitative ones.')
parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                        help='In quantitative mode, the file to save detections before calculating mAP.')
parser.add_argument('--resume', dest='resume', action='store_true',
                        help='If display not set, this resumes mAP calculations from the ap_data_file.')
parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')
parser.add_argument('--config', default=None,
                        help='The config object to use.')
parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
parser.add_argument('--web_det_path', default='web/dets/', type=str,
                        help='If output_web_json is set, this is the path to dump detections into.')
parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                        help='Equivalent to running display mode but without displaying an image.')
parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                        help='Do not sort images by hashed image ID.')
parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks with the predicted bounding box.')
parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.')
parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
parser.add_argument('--video_multiframe', default=1, type=int,
                        help='The number of frames to evaluate in parallel to make videos play at higher fps.')
parser.add_argument('--score_threshold', default=0, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
parser.add_argument('--display_fps', default=False, dest='display_fps', action='store_true',
                        help='When displaying / saving video, draw the FPS on the frame')
parser.add_argument('--emulate_playback', default=False, dest='emulate_playback', action='store_true',
                        help='When saving a video, emulate the framerate that you\'d get running in real-time mode.')
parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False, shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False, display_fps=False,
                        emulate_playback=False)

opt = parser.parse_args()

opt.num_classes = 80
