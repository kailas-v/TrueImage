from utils import *
from segmentation import *

def load_quality_detectors(args):
    quality_detectors = {}
    try:
        quality_detectors['blur'] = get_blur(all_args=args, **args.blur_detection)
    except:
        pass
    try:
        quality_detectors['lighting'] = get_lighting(all_args=args, **args.lighting_detection)
    except:
        pass
    try:
        quality_detectors['zoom'] = get_zoom(all_args=args, **args.zoom_detection)
    except:
        pass
    return quality_detectors



def get_blur(alg, alg_args, all_args):
    if alg == 'fourier':
        return FourierBlur(all_args=all_args, **alg_args)
    elif alg == 'laplacian':
        return FourierBlur(all_args=all_args, **alg_args)
    else:
        raise
def get_lighting(alg, alg_args, all_args):
    if alg == 'default':
        return DefaultLighting(all_args=all_args, **alg_args)
    elif alg == 'skin_dist':
        return SkinDistLighting(all_args=all_args, **alg_args)
    else:
        raise
def get_zoom(alg, alg_args, all_args):
    if alg == 'default':
        return DefaultZoom(all_args=all_args, **alg_args)
    else:
        raise

class FourierBlur:
    def __init__(self, threshold, use_patches, use_center_mask, use_skin_mask, all_args, **unused):
        self.threshold = threshold
        self.use_patches = use_patches
        self.use_center_mask = use_center_mask
        self.use_skin_mask = use_skin_mask
        self.all_args = all_args
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
            blur_scores = [evaluate_blur2(p) for p in patches]
            blur_score = np.quantile(blur_scores, 0.25)
        else:
            blur_score = evaluate_blur(image)
        
        return blur_score
    def decide(self, scores):
        return scores['blur'] < self.threshold

class LaplacianBlur:
    def __init__(self, threshold, use_patches, use_center_mask, use_skin_mask, all_args, **unused):
        self.threshold = threshold
        self.use_patches = use_patches
        self.use_center_mask = use_center_mask
        self.use_skin_mask = use_skin_mask
        self.all_args = all_args
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
            blur_scores = [evaluate_blur2(p) for p in patches]
            blur_score = np.quantile(blur_scores, 0.25)
        else:
            blur_score = evaluate_blur(image)
        
        return blur_score
    def decide(self, scores):
        return scores['blur'] < self.threshold

class DefaultLighting:
    def __init__(self, threshold, use_patches, use_center_mask, use_skin_mask, all_args, **unused):
        self.threshold = threshold
        self.use_patches = use_patches
        self.use_center_mask = use_center_mask
        self.use_skin_mask = use_skin_mask
        self.all_args = all_args
        self.ratio = 0.5
    def __call__(self, image, skin_mask, lesion_mask):
        def bad_lighting(patch):
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            dark_part = cv2.inRange(gray, 0, 30)
            bright_part = cv2.inRange(gray, 220, 255)
            total_pixel = np.size(gray)
            dark_pixel = np.sum(dark_part > 0)
            bright_pixel = np.sum(bright_part > 0)

            # -- Underexposed / Overexposed --
            score = 1 - (dark_pixel+bright_pixel)/total_pixel
            return score

        mask = skin_mask
        if self.use_center_mask:
            image = apply_center_crop(image, ratio=self.ratio)
            skin_mask = apply_center_crop(skin_mask, ratio=self.ratio)
            mask = np.ones(image.shape[:2]).astype(bool)
        if self.use_skin_mask:
            mask = mask & skin_mask
        if self.use_patches:
            patches = gen_patches_from_im(torch.Tensor(image), torch.Tensor(mask[:,:,None]), patch_size=32, max_patches=100)
            lighting_scores = [bad_lighting(p)[0] for p in self.all_args.patches]
            lighting_score = np.quantile(lighting_scores, 0.25)
        else:
            lighting_score = bad_lighting(image)

        return lighting_score

    def decide(self, scores):
        return scores['lighting'] < self.threshold

class SkinDistLighting:
    def __init__(self, threshold, use_patches, use_center_mask, use_skin_mask, all_args, **unused):
        self.threshold = threshold
        self.use_patches = use_patches
        self.use_center_mask = use_center_mask
        self.use_skin_mask = use_skin_mask
        self.all_args = all_args
        self.ratio = 0.5
    def __call__(self, image, skin_mask, lesion_mask):
        def bad_lighting(patch):
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            dark_part = cv2.inRange(gray, 0, 30)
            bright_part = cv2.inRange(gray, 220, 255)
            total_pixel = np.size(gray)
            dark_pixel = np.sum(dark_part > 0)
            bright_pixel = np.sum(bright_part > 0)

            # -- Underexposed / Overexposed --
            # score = 1 - (dark_pixel+bright_pixel)/total_pixel
            # return score
            return 1 - (dark_pixel)/total_pixel, 1 - (bright_pixel)/total_pixel

        # -- Generate an extra-lenient mask --
        mask_ = SkinSegmentationThreshold2(threshold=50, model_path="models/skin_dist.pkl")
        mask = mask_(image)
        if self.use_center_mask:
            image = apply_center_crop(image, ratio=self.ratio)
            skin_mask = apply_center_crop(skin_mask, ratio=self.ratio)
            mask = np.ones(image.shape[:2]).astype(bool)
        if self.use_skin_mask:
            mask = mask & skin_mask
        if self.use_patches:
            patches = gen_patches_from_im(torch.Tensor(image), torch.Tensor(mask[:,:,None]), patch_size=32, max_patches=100)
            lighting_scores = [bad_lighting(p)[0] for p in self.all_args.patches]
            lighting_score = np.quantile(lighting_scores, 0.25)
            lighting_scores = [bad_lighting(p)[1] for p in self.all_args.patches]
            lighting_score = lighting_score, np.quantile(lighting_scores, 0.25)
        else:
            lighting_score = bad_lighting(image)

        return lighting_score

    def decide(self, scores):
        return np.sum(scores['lighting'])-1 < self.threshold

class DefaultZoom:
    def __init__(self, threshold, use_center_mask, all_args, **unused):
        self.threshold = threshold
        self.use_center_mask = use_center_mask
        self.all_args = all_args
        self.ratio = 0.5
    def __call__(self, image, skin_mask, lesion_mask):
        if self.use_center_mask:
            skin_mask = apply_center_crop(skin_mask, ratio=self.ratio)
        skin_area = np.sum(skin_mask)/np.size(skin_mask)
        return skin_area
    def decide(self, scores):
        return scores['zoom'] < self.threshold