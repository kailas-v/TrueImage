import argparse
import gradio as gr
import matplotlib.pyplot as plt
import yaml

from utils import *
from segmentation import load_skin_segmentation, load_lesion_segmentation
from detect_quality import load_quality_detectors

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--path_to_images', type=str, default=None)
    parser.add_argument('--use_gradio', action='store_true', default=False)
    parser.add_argument('--web', action='store_true', default=False)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)
    config['use_gradio'] = args.use_gradio
    config['share_gradio'] = args.webs
    if args.path_to_images is not None:
        config['path_to_images'] = args.path_to_images
    args = Bunch(config)
    return args



def main(args):
    args.skin_segmentation = load_skin_segmentation(args)
    args.lesion_segmentation = load_lesion_segmentation(args)
    args.quality_detectors = load_quality_detectors(args)

    logistic_decide_quality = LogisticDecision("models/logistic")

    def process_im(im, plot_mask=True, plot_quality=True):
        """Processes and analyzes a single image.""" 
        set_seeds(args.seed)
        skin_mask = args.skin_segmentation(im)
        lesion_mask = args.lesion_segmentation(im, skin_mask)
        quality_scores = {k:[qii for qi in q for qii in make_list(qi(im,skin_mask,lesion_mask))] 
                            for k,q in args.quality_detectors.items()}



        im = display_tensor_im(im)
        if plot_mask:
            skin_mask = im*display_tensor_im(skin_mask)
            lesion_mask = im*display_tensor_im(lesion_mask)

        # -- Display quality scores --
        if plot_quality:
            quality_decisions = logistic_decide_quality.decide(quality_scores)
            quality_scores = "{0}".format([k for k,q in quality_decisions.items() if q])

        # -- Return values for display --
        return quality_scores, to_numpy(im), to_numpy(skin_mask), to_numpy(lesion_mask)

    
    # -- Use gradio to test interactively --
    if args.use_gradio:
        def gradio_process_im(im, skin_segmentation_threshold):
            args.skin_segmentation.threshold = skin_segmentation_threshold
            quality_scores, _, skin_mask, lesion_mask = process_im(im)
            return quality_scores, skin_mask, lesion_mask

        slider1 = args.skin_segmentation.threshold
        gr.Interface(fn=gradio_process_im,
             inputs=[gr.inputs.Image(shape=None), 
                     gr.inputs.Slider(minimum=2, maximum=100, default=slider1, label="skin threshold"),
                    ],
             outputs=[gr.outputs.Textbox(label="Quality labels"),
                      gr.outputs.Image(label="Skin Segmentation"),
                      gr.outputs.Image(label="Lesion Segmentation"),
                    ],
        ).launch(share=args.share_gradio)
    # -- Load image path or directory of images --
    else:
        for im, im_path in load_images(args):
            quality_scores, im, skin_mask, lesion_mask = process_im(im, plot_quality=False)
            print(f"{im_path}:\n"+"-"*80)
            for k,q in quality_scores.items():
                print(f"{k} -- {q}")
            display_image_and_mask_and_saliency(im, skin_mask, lesion_mask)
            print("-"*80+"-"*80+"\n\n")



if __name__ == '__main__':
    args = get_args()
    main(args)