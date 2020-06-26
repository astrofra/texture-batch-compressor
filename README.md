# texture-batch-compressor

## Automatic texture compressor, aiming to :
* find the best compressor between PNG/JPEG
* find the best amount of colors when using the PNG format, while keeping an acceptable quality VS the original
* stretch the textures when possible

## Based on
* ImageMagick
* Mozjpeg
* Pngquant

## Requirements
* PIL (pillow), to manipulate the texture files (in/out)
* Scikit, to process the images
* Statistics, to evaluate the quality of the compressed texture
* Numpy