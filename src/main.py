import os
import shutil
import statistics
import numpy as np
from PIL import Image
from PIL import ImageChops
from PIL import ImageFilter
from skimage.measure import compare_ssim as ssim

input_folder = '..\\in'
output_folder = '..\\out'
tmp_folder = '..\\tmp'
input_file_ext = '.png'


def make_tmp_folder():
	if os.path.exists(output_folder):
		shutil.rmtree(output_folder)
	os.makedirs(output_folder, exist_ok = True)

	if os.path.exists(tmp_folder):
		shutil.rmtree(tmp_folder)
	os.makedirs(tmp_folder, exist_ok = True)


def optimize_as_indexed_colors(file, input_path, output_path):
	variants = []
	# image magick
	for quant in ["SRGB", "YUV"]:
		for colors in [8, 16, 32, 64, 128, 256]:
			for dither in ["FloydSteinberg", "Riemersma"]: # , "None"
				_opt = [{'key': 'quantize', 'value': quant}, {'key':'dither', 'value': dither}, {'key': 'colors', 'value': colors}]
				variants.append(optimize_image_as_im_png(file, input_path, output_path, options=_opt))

	# pngquant
	for colors in [8, 16, 32, 64, 128, 256]:
		for dither in [0.5, 0.75, 1.0]:
			_opt = [{'key': 'speed', 'value': 1}, {'key': 'strip', 'value': None}, {'key': 'floyd=', 'value': dither}, {'key': '.colors', 'value': colors}]
			variants.append(optimize_image_as_quant_png(file, input_path, output_path, options=_opt))

	return variants


def optimize_image_as_im_png(file, input_path, output_path, options = []):
	options_str = '' # '-quantize YUV -dither FloydSteinberg -colors 32'
	for opt in options:
		if opt['value'] is not None:
			options_str += '-' + str(opt['key']) + ' ' + str(opt['value']) + ' '

	options_str = options_str.strip(' ')
	print(options_str)

	srcfile = os.path.join(input_path, file)
	options_as_filename = options_str.replace('-','_').replace(' ','_').replace('__','_').lower()
	destfile = os.path.join(output_path, file.split('.')[0] + options_as_filename + '.' + file.split('.')[1])
	result = os.popen('..\\bin\\imagemagick\\convert.exe' + ' "' + srcfile + '" ' + options_str + ' "' + destfile + '"').read()

	return destfile


def optimize_image_as_quant_png(file, input_path, output_path, options = []):
	options_str = ''
	for opt in options:
		if str(opt['key'])[0] == '.':
			options_str += ' ' + str(opt['value']) + ' '
		else:
			if opt['value'] is not None:
				options_str += '--' + str(opt['key']) + ' ' + str(opt['value']) + ' '
			else:
				options_str += '--' + str(opt['key']) + ' '

	options_str = options_str.replace('= ', '=')
	options_str = options_str.strip(' ')
	print(options_str)

	srcfile = os.path.join(input_path, file)
	options_as_filename = options_str.replace('-','_').replace(' ','_').replace('__','_').replace('__','_').lower()
	destfile = os.path.join(output_path, file.split('.')[0] + options_as_filename + '.' + file.split('.')[1])
	result = os.popen('..\\bin\\pngquant\\pngquant.exe' + ' ' + options_str + ' --output "' + destfile + '" "' + srcfile + '"').read()

	return destfile


def rgb_to_luminance(R,G,B):
	return 0.2126*R + 0.7152*G + 0.0722*B


def naive_diff_compare(filename_image_a, filename_image_b, smooth_before_comparison=True):
	# compare original and optimized image
	image_a = Image.open(filename_image_a)
	image_a = image_a.convert("RGB")

	image_b = Image.open(filename_image_b)
	image_b = image_b.convert("RGB")

	if smooth_before_comparison:
		image_a = image_a.filter(ImageFilter.SMOOTH)
		image_b = image_b.filter(ImageFilter.SMOOTH)
		image_b.save(os.path.splitext(filename_image_b)[0] + "_smooth" + os.path.splitext(filename_image_b)[1], "PNG")

	# compute the difference
	diff = ImageChops.difference(image_a, image_b)

	# accumulate the differences
	diff_sum = 0
	max_sum = diff.height * diff.width
	if max_sum > 0:
		for px in diff.getdata():
			diff_sum += (rgb_to_luminance(px[0], px[1], px[2]) / 255)

		diff_sum = 100.0 * diff_sum / max_sum
		return diff_sum

	return -1


def mse_compare(filename_image_a, filename_image_b, smooth_before_comparison=True):
	image_a = Image.open(filename_image_a).convert("RGB")
	image_b = Image.open(filename_image_b).convert("RGB")

	if smooth_before_comparison:
		image_a = image_a.filter(ImageFilter.SMOOTH)
		image_b = image_b.filter(ImageFilter.SMOOTH)
		image_b.save(os.path.splitext(filename_image_b)[0] + "_smooth" + os.path.splitext(filename_image_b)[1], "PNG")

	npa = []
	npb = []
	for px in image_a.getdata():
		npa.append(rgb_to_luminance(px[0], px[1], px[2]) / 255)
	for px in image_b.getdata():
		npb.append(rgb_to_luminance(px[0], px[1], px[2]) / 255)

	npa = np.asarray(npa)
	npb = np.asarray(npb)

	err = np.sum((npa.astype("float") - npb.astype("float")) ** 2)
	err /= float(image_a.height * image_a.width)

	return err


def ssim_compare(filename_image_a, filename_image_b, smooth_before_comparison=True):
	image_a = Image.open(filename_image_a).convert("RGB")
	image_b = Image.open(filename_image_b).convert("RGB")

	if smooth_before_comparison:
		image_a = image_a.filter(ImageFilter.SMOOTH)
		image_b = image_b.filter(ImageFilter.SMOOTH)
		image_b.save(os.path.splitext(filename_image_b)[0] + "_smooth" + os.path.splitext(filename_image_b)[1], "PNG")

	npa = []
	npb = []
	for px in image_a.getdata():
		npa.append(rgb_to_luminance(px[0], px[1], px[2]) / 255)
	for px in image_b.getdata():
		npb.append(rgb_to_luminance(px[0], px[1], px[2]) / 255)

	npa = np.asarray(npa)
	npb = np.asarray(npb)

	err = (1.0 - ssim(npa, npb)) * 100
	return err


def main():
	make_tmp_folder()

	for file_entry in os.listdir(input_folder):
		if file_entry.endswith(input_file_ext):
			print(file_entry)
			optimized_files = optimize_as_indexed_colors(file_entry, input_folder, tmp_folder)

			difference_scores = []
			candidates = []

			for optimized_file in optimized_files:
				# diff_sum = mse_compare(os.path.join(input_folder, file_entry), optimized_file)
				diff_sum = ssim_compare(os.path.join(input_folder, file_entry), optimized_file)
				# diff_sum = naive_diff_compare(os.path.join(input_folder, file_entry), optimized_file)
				difference_scores.append(diff_sum)
				size_ratio = os.path.getsize(os.path.join(input_folder, file_entry)) / os.path.getsize(optimized_file)
				print(optimized_file + ", diff = " + str(diff_sum)[0:6] + ", size ratio = " + str(size_ratio)[0:6])
				candidates.append({"filename": optimized_file, "difference": diff_sum, "compression": size_ratio})

			median_score = statistics.median(difference_scores) # * 0.5
			print("Median score = " + str(median_score))

			best_candidate = None
			for candidate in candidates:
				if candidate["difference"] <= median_score:
					if best_candidate is None or candidate["compression"] > best_candidate["compression"]:
						best_candidate = candidate

			print("Best difference/compression ratio found:")
			print(best_candidate)

			shutil.copy(best_candidate["filename"], os.path.join(output_folder, file_entry))


main()