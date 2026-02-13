
from .transforms import (
    impulse_noise, impulse_noise2, shot_noise, gaussian_noise,
    speckle_noise, defocus_blur, zoom_blur, zoom_in,
    freeze, low_light, sampling, fog, rain, occlusion, motion_blur, snow
)

ALL_TRANSFORMS = {
    "impulse_noise": impulse_noise,
    "impulse_noise2": impulse_noise2,
    "shot_noise": shot_noise,
    "gaussian_noise": gaussian_noise,
    "speckle_noise": speckle_noise,
    "defocus_blur": defocus_blur,
    "zoom_blur": zoom_blur,
    "zoom_in": zoom_in,
    "freeze": freeze,
    "low_light": low_light,
    "sampling": sampling,
    "fog": fog,
    "rain": rain,
    "occlusion": occlusion,
    "motion_blur": motion_blur,
    "snow": snow,
}
