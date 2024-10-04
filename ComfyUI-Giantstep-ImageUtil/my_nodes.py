import cv2
import numpy as np
from PIL import Image
import torch

class MeanBrightnessNode_Scaled:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),  # 입력 이미지
            # "min_value": ("FLOAT", {"default": 0.0, "description": "min brightness", 
            #         "min": 0.0,
            #         "max": 100.0,
            #         "step": 0.1,}),
            "max_value": ("FLOAT", {"default": 20, "description": "max brightness", 
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,}),
            }
        }

    RETURN_TYPES = ("FLOAT",)  # 평균 밝기 값을 반환
    RETURN_NAMES = ("mean_brightness",)
    FUNCTION = "execute"
    CATEGORY = "Giantstep"
    DESCRIPTION = "이미지의 평균 밝기를 계산하는 노드 (1 - mean_brightness) * max_value"
    
    def execute(self, input_image, max_value):
        print("첫 이미지 형태:", input_image.shape)
        if input_image.dim() == 4:  # batch 차원이 있는 경우
            input_image = input_image[0]  # 첫 번째 이미지만 사용

        # Tensor를 Numpy 배열로 변환
        input_image = input_image.cpu().numpy()
        print("Numpy array 형태:", input_image.shape)

        # 이미지 변환 (배열을 HWC 형식으로 변환 후, RGB->BGR로 변경)
        if input_image.shape[0] == 3:  # 채널 차원이 첫 번째인 경우 (C, H, W)
            input_image = np.transpose(input_image, (1, 2, 0))  # (H, W, C)로 변환
        
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

        # 그레이스케일로 변환
        gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        # 평균 밝기 계산
        mean_brightness = np.mean(gray_image)
        
         # 평균 밝기를 0~20 사이로 스케일링
        scaled_brightness = (1 - mean_brightness) * max_value
        # scaled_brightness = mean_brightness * max_value

        # 평균 밝기 로그 출력
        print(f"Mean brightness: {mean_brightness} , Scaled_brightness : {scaled_brightness}")

        return (scaled_brightness,)
    
class MeanBrightnessNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("FLOAT",)  # 평균 밝기 값을 반환
    RETURN_NAMES = ("mean_brightness",)
    FUNCTION = "execute"
    CATEGORY = "Giantstep"
    DESCRIPTION = "이미지의 평균 밝기를 계산하는 노드"
    
    def execute(self, input_image):
        print("첫 이미지 형태:", input_image.shape)
        if input_image.dim() == 4:  # batch 차원이 있는 경우
            input_image = input_image[0]  # 첫 번째 이미지만 사용

        # Tensor를 Numpy 배열로 변환
        input_image = input_image.cpu().numpy()
        print("Numpy array 형태:", input_image.shape)

        # 이미지 변환 (배열을 HWC 형식으로 변환 후, RGB->BGR로 변경)
        if input_image.shape[0] == 3:  # 채널 차원이 첫 번째인 경우 (C, H, W)
            input_image = np.transpose(input_image, (1, 2, 0))  # (H, W, C)로 변환
        
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

        # 그레이스케일로 변환
        gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        # 평균 밝기 계산
        mean_brightness = np.mean(gray_image)
        
        # 평균 밝기 로그 출력
        print(f"Mean brightness: {mean_brightness}")
        
        if mean_brightness >= 0.45:
            mean_brightness = mean_brightness * 1.3
            if mean_brightness > 1.0:
                mean_brightness = 1.0
            print(f"changed Mean brightness: {mean_brightness}")

        return (mean_brightness,)

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class ImageAutoCrop_RGBA:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute_rgba"
    CATEGORY = "Giantstep"

    def execute2(self, image):
        # image = pil2tensor(remove(tensor2pil(image)))
        pil_image = tensor2pil(image)

        # 1. Pillow 이미지를 Numpy 배열로 변환 (RGB로 유지)
        pil_image_np = np.array(pil_image)

        # 2. RGB -> BGR 변환 (OpenCV와 호환되게)
        pil_image_np_bgr = cv2.cvtColor(pil_image_np, cv2.COLOR_RGB2BGR)

        # 3. Numpy 배열을 사용하여 그레이스케일 변환
        gray = cv2.cvtColor(pil_image_np_bgr, cv2.COLOR_BGR2GRAY)

        # 4. 이진화 처리 및 컨투어 찾기
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)

        # 5. 경계 상자 계산 및 이미지 크롭
        x, y, w, h = cv2.boundingRect(contour)
        cropped_img = pil_image_np_bgr[y:y+h, x:x+w]

        cv2.imwrite("cropped_img.png", cropped_img)

        # 6. 크롭된 이미지를 Pillow로 변환 (BGR -> RGB 변환)
        cropped_img_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

        # 7. PIL 이미지를 PyTorch 텐서로 변환
        tensor_image = pil2tensor(cropped_img_pil)
        
        return (tensor_image,)
    
    def execute(self, image):
        # image = pil2tensor(remove(tensor2pil(image)))
        pil_image = tensor2pil(image)

        # 1. Pillow 이미지를 Numpy 배열로 변환 (RGBA로 유지)
        pil_image_np = np.array(pil_image)

        # 2. 투명 배경(알파 채널)을 고려하여 처리 (알파 채널 확인)
        if pil_image_np.shape[-1] == 4:  # RGBA 이미지인 경우
            # 알파 채널을 분리하여 배경을 무시하고 경계를 잡기 위한 마스크 생성
            alpha_channel = pil_image_np[:, :, 3]

            # 알파 채널을 기준으로 그레이스케일 변환 (투명한 부분을 제외한 마스크 생성)
            gray = alpha_channel  # 알파 채널 자체가 그레이스케일 마스크 역할을 함
        else:
            # 알파 채널이 없는 경우 (기존 처리 방식)
            pil_image_np_bgr = cv2.cvtColor(pil_image_np, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(pil_image_np_bgr, cv2.COLOR_BGR2GRAY)

        # 3. 이진화 처리 및 컨투어 찾기 (알파 채널에서 컨투어 찾기)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None  # 컨투어가 없을 경우 처리 중단

        # 4. 가장 큰 컨투어 찾기
        contour = max(contours, key=cv2.contourArea)

        # 5. 경계 상자 계산 및 이미지 크롭 (RGB와 알파 채널을 함께 크롭)
        x, y, w, h = cv2.boundingRect(contour)

        if pil_image_np.shape[-1] == 4:  # RGBA 이미지인 경우
            # RGB와 알파 채널을 모두 크롭
            cropped_rgb = pil_image_np[y:y+h, x:x+w, :3]
            cropped_alpha = alpha_channel[y:y+h, x:x+w]
            
            # RGB와 알파 채널을 다시 합쳐 RGBA로 변환
            cropped_img = np.dstack((cropped_rgb, cropped_alpha))
        else:
            # 알파 채널이 없는 경우 (기존 처리)
            cropped_img = pil_image_np_bgr[y:y+h, x:x+w]

        # 6. 크롭된 이미지를 Pillow로 변환 (BGR -> RGB 변환, RGBA 처리)
        if pil_image_np.shape[-1] == 4:
            # RGBA 이미지로 변환
            cropped_img_pil = Image.fromarray(cropped_img, 'RGBA')
        else:
            # BGR -> RGB로 변환 후 Pillow 이미지로 변환
            cropped_img_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

        # 7. PIL 이미지를 PyTorch 텐서로 변환
        tensor_image = pil2tensor(cropped_img_pil)

        return (tensor_image,)
    
    def execute_rgba(self, image):
        # image = pil2tensor(remove(tensor2pil(image)))
        pil_image = tensor2pil(image)

        # 1. Pillow 이미지를 Numpy 배열로 변환 (RGBA로 유지)
        pil_image_np = np.array(pil_image)

        # 2. 알파 채널을 분리하여 배경을 무시하고 경계를 잡기 위한 마스크 생성
        alpha_channel = pil_image_np[:, :, 3]

        # 3. 알파 채널을 기준으로 그레이스케일 변환 (투명한 부분을 제외한 마스크 생성)
        gray = alpha_channel  # 알파 채널 자체가 그레이스케일 마스크 역할을 함

        # 4. 이진화 처리 및 컨투어 찾기 (알파 채널에서 컨투어 찾기)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None  # 컨투어가 없을 경우 처리 중단

        # 5. 가장 큰 컨투어 찾기
        contour = max(contours, key=cv2.contourArea)

        # 6. 경계 상자 계산 및 이미지 크롭 (RGB와 알파 채널을 함께 크롭)
        x, y, w, h = cv2.boundingRect(contour)
        cropped_rgb = pil_image_np[y:y+h, x:x+w, :3]
        cropped_alpha = alpha_channel[y:y+h, x:x+w]

        # 7. RGB와 알파 채널을 다시 합쳐 RGBA로 변환
        cropped_img = np.dstack((cropped_rgb, cropped_alpha))

        # 8. 크롭된 이미지를 Pillow로 변환 (RGBA 유지)
        cropped_img_pil = Image.fromarray(cropped_img, 'RGBA')

        # 9. PIL 이미지를 PyTorch 텐서로 변환
        tensor_image = pil2tensor(cropped_img_pil)

        return (tensor_image,)

class ImageAutoCrop_RGB:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute_rgb"
    CATEGORY = "Giantstep"

    def execute_rgb(self, image):
        # image = pil2tensor(remove(tensor2pil(image)))
        pil_image = tensor2pil(image)

        # 1. Pillow 이미지를 Numpy 배열로 변환 (RGB로 유지)
        pil_image_np = np.array(pil_image)

        # 2. RGB -> BGR 변환 (OpenCV와 호환되게)
        pil_image_np_bgr = cv2.cvtColor(pil_image_np, cv2.COLOR_RGB2BGR)

        # 3. Numpy 배열을 사용하여 그레이스케일 변환
        gray = cv2.cvtColor(pil_image_np_bgr, cv2.COLOR_BGR2GRAY)

        # 4. 이진화 처리 및 컨투어 찾기
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None  # 컨투어가 없을 경우 처리 중단

        # 5. 가장 큰 컨투어 찾기
        contour = max(contours, key=cv2.contourArea)

        # 6. 경계 상자 계산 및 이미지 크롭
        x, y, w, h = cv2.boundingRect(contour)
        cropped_img = pil_image_np_bgr[y:y+h, x:x+w]

        # 7. 크롭된 이미지를 Pillow로 변환 (BGR -> RGB 변환)
        cropped_img_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

        # 8. PIL 이미지를 PyTorch 텐서로 변환
        tensor_image = pil2tensor(cropped_img_pil)

        return (tensor_image,)
    
class ImageAutoCrop_RGB_UseMask:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Giantstep"

    def execute(self, image, mask):
        
        pil_image = tensor2pil(image)
        pil_image_np = np.array(pil_image)
        pil_image_np_bgr = cv2.cvtColor(pil_image_np, cv2.COLOR_RGB2BGR)
        
        
        pil_mask = tensor2pil(mask)
        pil_mask_np = np.array(pil_mask)
        pil_mask_np_bgr = cv2.cvtColor(pil_mask_np, cv2.COLOR_RGB2BGR)
        gray_mask = cv2.cvtColor(pil_mask_np_bgr, cv2.COLOR_BGR2GRAY)
        inverted_mask = cv2.bitwise_not(gray_mask)
        _, thresh = cv2.threshold(inverted_mask, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None  # 컨투어가 없을 경우 처리 중단
        
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        cropped_img = pil_image_np_bgr[y:y+h, x:x+w]
        cropped_img_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        tensor_image = pil2tensor(cropped_img_pil)

        return (tensor_image,)

# NODE_CLASS_MAPPINGS에 새 노드 추가
NODE_CLASS_MAPPINGS = {
    "MeanBrightnessNode_Scaled": MeanBrightnessNode_Scaled,
    "MeanBrightnessNode": MeanBrightnessNode,
    "ImageAutoCrop_RGBA": ImageAutoCrop_RGBA,
    "ImageAutoCrop_RGB": ImageAutoCrop_RGB,
    "ImageAutoCrop_RGB_UseMask": ImageAutoCrop_RGB_UseMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeanBrightnessNode_Scaled": "Mean Brightness Calculator - Scaled",
    "MeanBrightnessNode": "Mean Brightness Calculator",
    "ImageAutoCrop_RGBA": "ImageAutoCrop_RGBA",
    "ImageAutoCrop_RGB": "ImageAutoCrop_RGB",
    "ImageAutoCrop_RGB_UseMask": "ImageAutoCrop_RGB_UseMask"
}
