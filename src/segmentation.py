from utils import *
add_outside_module("outside_repos/SemanticSegmentation/")
import semantic_segmentation 

def load_skin_segmentation(args):
    alg_args = args.skin_segmentation
    if alg_args['alg'][0] == 'center_crop':
        return CenterCrop(**alg_args['alg_args'][0])
    elif alg_args['alg'][0] == 'threshold_1':
        return SkinSegmentationThreshold1(**alg_args['alg_args'][0])
    elif alg_args['alg'][0] == 'threshold_2':
        return SkinSegmentationThreshold2(**alg_args['alg_args'][0])
    elif alg_args['alg'][0] == 'cnn':
        return SkinSegmentationCNN(**alg_args['alg_args'][0])
    else:
        raise
def load_lesion_segmentation(args):
    alg_args = args.lesion_segmentaion
    if alg_args['alg'][0] == 'default':
        return DefaultLesion(**alg_args['alg_args'][0])
    elif alg_args['alg'][0] == 'cnn':
        return LesionSegmentationCNN(**alg_args['alg_args'][0])
    else:
        raise

# -- Skin Segmentation --
class CenterCrop:
    def __init__(self, **unused):
        pass
    def __call__(self, image):
        mask = np.zeros(image.shape[:2])
        im_w, im_h = mask.shape[:2]
        mask[im_w//4:3*im_w//4, im_h//4:3*im_h//4] = 1
        return mask.astype(bool)
class SkinSegmentationThreshold1:
    def __init__(self, threshold, center_crop=False, **unused):
        self.threshold = threshold
        self.center_crop = center_crop
    def __call__(self, image):
        mask = pil_to_skin_mask(image, self.threshold).astype(bool)
        if self.center_crop:
            mask = mask & gen_center_mask(image, ratio=0.5)
        return mask
class SkinSegmentationThreshold2:
    def __init__(self, threshold, model_path, center_crop=False, **unused):
        self.threshold = threshold
        self.center_crop = center_crop
        self.model = SkinDistribution(model_path)
    def __call__(self, image):
        mask = np.zeros(image.shape[:2])
        mask.reshape((-1,))[np.where(self.model(image).reshape((-1,)) < self.threshold)] = 1
        mask = mask.astype(np.uint8)
        mask = cv2.medianBlur(mask,3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))
        mask = mask.astype(bool)
        if self.center_crop:
            mask = mask & gen_center_mask(image, ratio=0.5)
        return mask
class SkinSegmentationCNN:
    def __init__(self, threshold, model_path, center_crop=False, **unused):
        self.threshold = threshold
        self.center_crop = center_crop

        if torch.cuda.is_available():
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        self.model = semantic_segmentation.load_model(semantic_segmentation.models['FCNResNet101'], state_dict)

        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()
        
        self.fn_image_transform2 = transforms.ToTensor()
        self.fn_image_transform3 = transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                                        std=(0.229, 0.224, 0.225))
    
    def __call__(self, image):
        if type(image) is not torch.Tensor:
            image = self.fn_image_transform2(image)
        image = self.fn_image_transform3(image)
            
        with torch.no_grad():
            if torch.cuda.is_available():
                image = image.cuda()
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            results = self.model(image)['out']
            results = torch.sigmoid(results) > self.threshold
            
        return results.detach().cpu().numpy().astype(bool)[0,0]

class LesionSegmentationCNN:
    def __init__(self, threshold, model_path, center_crop=False, **unused):
        self.threshold = threshold
        self.center_crop = center_crop

        
        if torch.cuda.is_available():
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        num_classes = state_dict['classifier.6.bias'].shape[0] # specific for VGG16
        self.model = models.vgg16(pretrained=True)
        self.model.classifier[-1] = nn.Linear(4096, num_classes)
        self.model.load_state_dict(state_dict)

        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()
        
        self.fn_image_transform2 = transforms.ToTensor()
        self.fn_image_transform3 = transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                                        std=(0.229, 0.224, 0.225))
    
    def __call__(self, image, skin_mask):
        if type(image) is not torch.Tensor:
            image = self.fn_image_transform2(image)
        image = self.fn_image_transform3(image)
        
        if torch.cuda.is_available():
            image = image.cuda()
        image = image.unsqueeze(0)
        image.requires_grad_()
        results_ = self.model(image)
        
        results = torch.nn.functional.softmax(results_, dim=1)
        classification = torch.argmax(results)
        
        score_max = results_[0,results_.argmax()]
        score_max.backward()
        saliency, _ = torch.max(image.grad.data.abs(),dim=1)
        
        return saliency

class DefaultLesion:
    def __init__(self, **unused):
        pass
    def __call__(self, image, skin_mask):
        return skin_mask
