from skimage.metrics import structural_similarity


def ssim(img1, img2):
    img1_max, img1_min = img1.max(), img1.min()
    img2_max, img2_min = img2.max(), img2.min()

    img1 = (img1 - img1_min) / (img1_max - img1_min)
    img2 = (img2 - img2_min) / (img2_max - img2_min)

    return structural_similarity(img1, img2, data_range=1)
