
from einops import rearrange


def warp_time(t, dt=None, s=.25):
    # https://drscotthawley.github.io/blog/posts/FlowModels.html#more-points-where-needed-via-time-warping
    # samples more often t in the middle
    tw = 4*(1-s)*t**3 + 6*(s-1)*t**2 + (3-2*s)*t 
    if dt:
        return tw,  dt * 12*(1-s)*t**2 + 12*(s-1)*t + (3-2*s) 
    return tw

def image_to_patches(x, patch_size):
    x_patches = rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size)
    return x_patches

def patches_to_image(x_hat_patches, patch_size, input_h):
    x_hat = rearrange(x_hat_patches, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1=patch_size, p2=patch_size, h=input_h//patch_size)
    return x_hat


def image_to_patches_mnist(x, patch_size):
    x_patches = rearrange(x, "b (h p1) (w p2) -> b (h w) (p1 p2)", p1=patch_size, p2=patch_size)
    return x_patches

def patches_to_image_mnist(x_hat_patches, patch_size, input_h):
    x_hat = rearrange(x_hat_patches, "b (h w) (p1 p2) -> b (h p1) (w p2)", p1=patch_size, p2=patch_size, h=input_h//patch_size)
    return x_hat
