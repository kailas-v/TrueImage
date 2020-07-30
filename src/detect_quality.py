from utils import *
from segmentation import *

def load_quality_detectors(args):
    quality_detectors = {}
    try:
        a_ = args.blur_detection
        quality_detectors['blur'] = [get_blur(alg=alg, alg_args=alg_args, i=i, all_args=args) 
            for i,(alg,alg_args) in enumerate(zip(a_['alg'], a_['alg_args']))]
    except:
        pass
    try:
        a_ = args.lighting_detection
        quality_detectors['lighting'] = [get_lighting(alg=alg,alg_args=alg_args,i=i,all_args=args)
            for i,(alg,alg_args) in enumerate(zip(a_['alg'], a_['alg_args']))]
    except:
        pass
    try:
        a_ = args.zoom_detection
        quality_detectors['zoom'] = [get_zoom(alg=alg, alg_args=alg_args, i=i, all_args=args) 
            for i,(alg,alg_args) in enumerate(zip(a_['alg'], a_['alg_args']))]
    except:
        pass
    return quality_detectors



def get_blur(alg, alg_args, i, all_args):
    if alg == 'fourier':
        return FourierBlur(i=i, all_args=all_args, **alg_args)
    elif alg == 'laplacian':
        return LaplacianBlur(i=i, all_args=all_args, **alg_args)
    else:
        raise
def get_lighting(alg, alg_args, i, all_args):
    if alg == 'default':
        return DefaultLighting(i=i, all_args=all_args, **alg_args)
    elif alg == 'skin_dist':
        return SkinDistLighting(i=i, all_args=all_args, **alg_args)
    else:
        raise
def get_zoom(alg, alg_args, i, all_args):
    if alg == 'default':
        return DefaultZoom(i=i, all_args=all_args, **alg_args)
    else:
        raise

class FourierBlur:
    def __init__(self, threshold, use_patches, use_center_mask, use_skin_mask, i, all_args, **unused):
        self.threshold = threshold
        self.use_patches = use_patches
        self.use_center_mask = use_center_mask
        self.use_skin_mask = use_skin_mask
        self.all_args = all_args
        self._i = i
        self.ratio = 0.5
    def __call__(self, image, skin_mask, lesion_mask):
        def evaluate_blur(img_col):
            """From https://github.com/whdcumt/BlurDetection/blob/master/main.py"""
            img_gry = cv2.cvtColor(img_col, cv2.COLOR_RGB2GRAY)
            rows, cols = img_gry.shape
            crow, ccol = rows//2, cols//2
            f = np.fft.fft2(img_gry)
            fshift = np.fft.fftshift(f)
            fshift[crow-rows//8:crow+rows//8, ccol-cols//8:ccol+cols//8] = 0
            f_ishift = np.fft.ifftshift(fshift)
            img_fft = np.fft.ifft2(f_ishift)
            img_fft = 20*np.log(np.abs(img_fft))
            result = np.mean(img_fft)
            return result

        mask = skin_mask
        if self.use_center_mask:
            image = apply_center_crop(image, ratio=self.ratio)
            skin_mask = apply_center_crop(skin_mask, ratio=self.ratio)
            mask = np.ones(image.shape[:2]).astype(bool)
        if self.use_skin_mask:
            mask = mask & skin_mask
        if self.use_patches:
            patches = gen_patches_from_im(torch.Tensor(image), torch.Tensor(mask[:,:,None]), patch_size=32, max_patches=100)
            if patches is None:
                blur_scores = None
            else:
                blur_scores = [evaluate_blur(p) for p in patches]
                #blur_score = [np.quantile(blur_scores, q_) for q_ in np.arange(0.1,1.,0.2)]
                blur_score = basic_stats([b for b in blur_scores])
        if not self.use_patches or blur_scores is None:
            blur_score = basic_stats(evaluate_blur(image))
        
        return blur_score
    def decide(self, scores):
        return scores['blur'] < self.threshold

class LaplacianBlur:
    def __init__(self, threshold, use_patches, use_center_mask, use_skin_mask, i, all_args, **unused):
        self.threshold = threshold
        self.use_patches = use_patches
        self.use_center_mask = use_center_mask
        self.use_skin_mask = use_skin_mask
        self.all_args = all_args
        self._i = i
        self.ratio = 0.5
    def __call__(self, image, skin_mask, lesion_mask):
        def evaluate_blur(img_col):
            """From https://github.com/whdcumt/BlurDetection/blob/master/main.py"""
            img_gry = cv2.cvtColor(img_col, cv2.COLOR_RGB2GRAY)
            return cv2.Laplacian(img_gry, cv2.CV_64F).var()

        mask = skin_mask
        if self.use_center_mask:
            image = apply_center_crop(image, ratio=self.ratio)
            skin_mask = apply_center_crop(skin_mask, ratio=self.ratio)
            mask = np.ones(image.shape[:2]).astype(bool)
        if self.use_skin_mask:
            mask = mask & skin_mask
        if self.use_patches:
            patches = gen_patches_from_im(torch.Tensor(image), torch.Tensor(mask[:,:,None]), patch_size=32, max_patches=100)
            if patches is None:
                blur_scores = None
            else:
                blur_scores = [evaluate_blur(p) for p in patches]
                #blur_score = [np.quantile(blur_scores, q_) for q_ in np.arange(0.1,1.,0.2)]
                blur_score = basic_stats([b for b in blur_scores])
        if not self.use_patches or blur_scores is None:
            blur_score = basic_stats(evaluate_blur(image))
        
        return blur_score
    def decide(self, scores):
        return scores['blur'] < self.threshold

class DefaultLighting:
    def __init__(self, threshold, use_patches, use_center_mask, use_skin_mask, i, all_args, **unused):
        self.threshold = threshold
        self.use_patches = use_patches
        self.use_center_mask = use_center_mask
        self.use_skin_mask = use_skin_mask
        self.all_args = all_args
        self._i = i
        self.ratio = 0.5
    def __call__(self, image, skin_mask, lesion_mask):
        def bad_lighting(patch):
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            under_exposed = gray[(gray < 50)]
            over_exposed = gray[(gray > 205)]
            if len(under_exposed) == 0:
                under_exposed = basic_stats2([24.5]) 
            else:
                under_exposed = basic_stats2(under_exposed)
            if len(over_exposed) == 0:
                over_exposed = basic_stats2([229.5]) 
            else:
                over_exposed = basic_stats2(over_exposed)
            return np.concatenate((under_exposed, over_exposed))

        mask = skin_mask
        if self.use_center_mask:
            image = apply_center_crop(image, ratio=self.ratio)
            skin_mask = apply_center_crop(skin_mask, ratio=self.ratio)
            mask = np.ones(image.shape[:2]).astype(bool)
        if self.use_skin_mask:
            mask = mask & skin_mask
        if self.use_patches:
            patches = gen_patches_from_im(torch.Tensor(image), torch.Tensor(mask[:,:,None]), patch_size=32, max_patches=100)
            if patches is None:
                lighting_scores = None
            else:
                lighting_scores = [bad_lighting(p) for p in patches]
                lighting_score = [lii for i in range(len(lighting_scores[0])) for lii in basic_stats([l[i] for l in lighting_scores])]
        if not self.use_patches or lighting_scores is None:
            lighting_score = [lii for li in bad_lighting(image) for lii in basic_stats(li)]

        return lighting_score

    def decide(self, scores):
        return scores['lighting'] < self.threshold

class SkinDistLighting:
    def __init__(self, threshold, use_patches, use_center_mask, use_skin_mask, i, all_args, **unused):
        self.threshold = threshold
        self.use_patches = use_patches
        self.use_center_mask = use_center_mask
        self.use_skin_mask = use_skin_mask
        self.all_args = all_args
        self._i = i
        self.ratio = 0.5
    def __call__(self, image, skin_mask, lesion_mask):
        def bad_lighting(patch):
            return basic_stats2(patch)

        # -- Generate an extra-lenient mask --
        mask_ = SkinDistribution(model_path="models/skin_dist.pkl")
        scored_mask = mask_(image)
        if self.use_center_mask:
            image = apply_center_crop(image, ratio=self.ratio)
            skin_mask = apply_center_crop(skin_mask, ratio=self.ratio)
            scored_mask = apply_center_crop(scored_mask, ratio=self.ratio)
            mask = np.ones(image.shape[:2]).astype(bool)
        if self.use_skin_mask:
            mask = mask & skin_mask
        if self.use_patches:
            # patches = gen_patches_from_im(torch.Tensor(np.concatenate((image, scored_mask[:,:,None]), axis=-1)), torch.Tensor(mask[:,:,None]), patch_size=32, max_patches=100)
            patches = gen_patches_from_im(torch.Tensor(scored_mask[:,:,None]), torch.Tensor(mask[:,:,None]), patch_size=32, max_patches=100)
            if patches is None:
                lighting_scores = None
            else:
                lighting_scores = [bad_lighting(p) for p in patches]
                lighting_score = [lii for i in range(len(lighting_scores[0])) for lii in basic_stats([l[i] for l in lighting_scores])]
        if not self.use_patches or lighting_scores is None:
            lighting_scores = bad_lighting(image)
            lighting_score = [lii for li in bad_lighting(image) for lii in basic_stats(li)]

        return lighting_score

    def decide(self, scores):
        return np.sum(scores['lighting'])-1 < self.threshold

class DefaultZoom:
    def __init__(self, threshold, use_center_mask, i, all_args, **unused):
        self.threshold = threshold
        self.use_center_mask = use_center_mask
        self.all_args = all_args
        self._i = i
        self.ratio = 0.5
    def __call__(self, image, skin_mask, lesion_mask):
        if self.use_center_mask:
            skin_mask = apply_center_crop(skin_mask, ratio=self.ratio)
        skin_area = np.sum(skin_mask)/np.size(skin_mask)
        return skin_area
    def decide(self, scores):
        return scores['zoom'] < self.threshold