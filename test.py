import torch
from torchvision.transforms.functional import rotate
import numpy as np
import matplotlib.pyplot as plt
import cv2
from metrics.old_metrics import Metrics
from torchvision.transforms.functional import resize

def resize_ndarray(image, scale=4):
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)
    size = int(image.shape[2] * scale)
    image = resize(image, size, antialias=False)
    image = image.squeeze(0)
    image = image.numpy()
    return image

def main():
    data = True
    treed_reconstruction = np.load("pipeline_cache/reconstruction.npy")
    original_object = np.load("pipeline_cache/original_object.npy")
    print(treed_reconstruction.max(), treed_reconstruction.min())
    print(original_object.max(), original_object.min())
    print(treed_reconstruction.shape, original_object.shape)
    metric_data = {"mse": [], "psnr": [], "ssim": []}
    for i in range(treed_reconstruction.shape[0]):
        # images
        reconstructed_image = treed_reconstruction[i]
        original_image = original_object[i]
        
        # metrics 
        metric = Metrics()
        mse = metric.mse(reconstructed_image, original_image)
        psnr = metric.psnr(reconstructed_image, original_image)
        ssim = metric.ssim(reconstructed_image, original_image)
        metric_data["mse"].append(mse)
        metric_data["psnr"].append(psnr)
        metric_data["ssim"].append(ssim)
        text = f"MSE: {mse:.4f}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}"
        
        # resize
        if data:
            scale = 1
        else:
            scale = 4
        reconstructed_image = resize_ndarray(reconstructed_image, scale)
        original_image = resize_ndarray(original_image, scale)
        
        # concatenate
        image = np.concatenate((reconstructed_image, original_image), axis=1)
        
        # put text
        cv2.putText(image, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # show
        cv2.imshow("image", image)
        pressed_key = cv2.waitKey(100)
        if pressed_key == ord('q'): break
    
    # plot metrics
    plt.figure(0, figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(metric_data["mse"], label=f"MSE ({np.mean(metric_data['mse']):.4f})")
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(metric_data["psnr"], label=f"PSNR ({np.mean(metric_data['psnr']):.4f})")
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(metric_data["ssim"], label=f"SSIM ({np.mean(metric_data['ssim']):.4f})")
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()
