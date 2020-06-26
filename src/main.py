import os
import shutil
import statistics
import numpy as np
from PIL import Image
from PIL import ImageChops
from PIL import ImageFilter
from skimage.metrics import structural_similarity as ssim
import threading
import time

# Compression aggressivity : 0.0 less aggressive, 1.0 more aggressive
aggressivity = 0.5

input_folder = '..\\in'
output_folder = '..\\out'
tmp_folder = '..\\tmp'
input_file_ext = '.png'


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


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
	thrds = []
	for quant in ["YUV"]: #["SRGB", "YUV"]:
		for colors in [128]:
			for dither in ["FloydSteinberg"]: #["FloydSteinberg", "Riemersma"]: # , "None"
				_opt = [{'key': 'quantize', 'value': quant}, {'key':'dither', 'value': dither}, {'key': 'colors', 'value': colors}]
				t = threading.Thread(target=optimize_image_as_im_png, args=(file, input_path, output_path, _opt, variants))
				t.start()
				thrds.append(t)
				# variants.append(optimize_image_as_im_png(file, input_path, output_path, options=_opt))

	print("waiting for optimize_image_as_im_png() threads")
	for i, t in enumerate(thrds):
		t.join()
		# variants.append(t.join())
	print("Done!")
	print(variants)

	# mozjpeg
	thrds = []
	for quality in [40, 50, 60, 70, 80, 90]:
		_opt = [{'key': 'quality', 'value': quality}]
		# variants.append(optimize_image_as_mozilla_jpeg(file, input_path, output_path, options=_opt))
		t = threading.Thread(target=optimize_image_as_mozilla_jpeg, args=(file, input_path, output_path, _opt, variants))
		t.start()
		thrds.append(t)

	print("waiting for optimize_image_as_mozilla_jpeg() threads")
	for i, t in enumerate(thrds):
		t.join()
		# variants.append(t.join())
	print("Done!")
	print(variants)

	# pngquant
	thrds = []
	for colors in [16, 64, 128, 256]:
		for dither in [0.8]: # [0.5, 0.9]:
			_opt = [{'key': 'speed', 'value': 1}, {'key': 'strip', 'value': None}, {'key': 'floyd=', 'value': dither}, {'key': '.colors', 'value': colors}]
			# variants.append(optimize_image_as_quant_png(file, input_path, output_path, options=_opt))
			t = threading.Thread(target=optimize_image_as_quant_png, args=(file, input_path, output_path, _opt, variants))
			t.start()
			thrds.append(t)

	print("waiting for optimize_image_as_quant_png() threads")
	for i, t in enumerate(thrds):
		t.join()
		# variants.append(t.join())
	print("Done!")
	print(variants)

	return variants

def optimize_image_as_mozilla_jpeg(file, input_path, output_path, options = [], v = []):
	options_str = '' # '-quantize YUV -dither FloydSteinberg -colors 32'
	for opt in options:
		if opt['value'] is not None:
			options_str += '-' + str(opt['key']) + ' ' + str(opt['value']) + ' '

	options_str = options_str.strip(' ')
	print('mozjpeg.exe ' + options_str)

	srcfile = os.path.join(input_path, file)
	options_as_filename = options_str.replace('-', '_').replace(' ', '_').replace('__', '_').lower()
	tempfile = os.path.join(output_path, "tmp_" + file.split('.')[0] + options_as_filename + '.' + 'ppm')
	destfile = os.path.join(output_path, file.split('.')[0] + options_as_filename + '.' + 'jpg')
	result = os.popen('..\\bin\\imagemagick\\convert.exe' + ' "' + srcfile + '" ' + '"' + tempfile + '"').read()
	result = os.popen('..\\bin\\mozjpeg\\cjpeg-static.exe' + ' ' + options_str + ' -outfile "' + destfile + '" "' + tempfile + '"').read()

	v.append(destfile)
	return destfile


def optimize_image_as_im_png(file, input_path, output_path, options = [], v = []):
	options_str = '' # '-quantize YUV -dither FloydSteinberg -colors 32'
	for opt in options:
		if opt['value'] is not None:
			options_str += '-' + str(opt['key']) + ' ' + str(opt['value']) + ' '

	options_str = options_str.strip(' ')
	print('imagemagick.exe ' + options_str)

	srcfile = os.path.join(input_path, file)
	options_as_filename = options_str.replace('-','_').replace(' ','_').replace('__','_').lower()
	destfile = os.path.join(output_path, file.split('.')[0] + options_as_filename + '.' + file.split('.')[1])
	result = os.popen('..\\bin\\imagemagick\\convert.exe' + ' "' + srcfile + '" ' + options_str + ' "' + destfile + '"').read()

	v.append(destfile)
	return destfile


def optimize_image_as_quant_png(file, input_path, output_path, options = [], v = []):
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
	print('quantpng.exe ' + options_str)

	srcfile = os.path.join(input_path, file)
	options_as_filename = options_str.replace('-','_').replace(' ','_').replace('__','_').replace('__','_').lower()
	destfile = os.path.join(output_path, file.split('.')[0] + options_as_filename + '.' + file.split('.')[1])
	result = os.popen('..\\bin\\pngquant\\pngquant.exe' + ' ' + options_str + ' --output "' + destfile + '" "' + srcfile + '"').read()

	v.append(destfile)
	return destfile


def rgb_to_luminance(R,G,B):
	return 0.2126*R + 0.7152*G + 0.0722*B


def naive_diff_compare(filename_image_a, filename_image_b, smooth_before_comparison=False, d={}):
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
			diff_sum += (rgb_to_luminance(px[0], px[1], px[2]) * 0.00392156862745098) # / 255)

		diff_sum = 100.0 * diff_sum / max_sum
	else:
		diff_sum = -1

	d["diff_sum_naive"] = diff_sum
	return diff_sum


def mse_compare(filename_image_a, filename_image_b, smooth_before_comparison=False, d={}):
	image_a = Image.open(filename_image_a).convert("RGB")
	image_b = Image.open(filename_image_b).convert("RGB")

	if smooth_before_comparison:
		image_a = image_a.filter(ImageFilter.SMOOTH)
		image_b = image_b.filter(ImageFilter.SMOOTH)
		image_b.save(os.path.splitext(filename_image_b)[0] + "_smooth" + os.path.splitext(filename_image_b)[1], "PNG")

	npa = []
	npb = []
	for px in image_a.getdata():
		npa.append(rgb_to_luminance(px[0], px[1], px[2]) * 0.00392156862745098) #  / 255)
	for px in image_b.getdata():
		npb.append(rgb_to_luminance(px[0], px[1], px[2]) * 0.00392156862745098) #  / 255)

	npa = np.asarray(npa)
	npb = np.asarray(npb)

	err = np.sum((npa.astype("float") - npb.astype("float")) ** 2)
	err /= float(image_a.height * image_a.width)

	d["diff_sum_mse"] = err
	# diff_sum_ssim = diff_dict["diff_sum_ssim"]
	# diff_sum_naive = diff_dict["diff_sum_naive"]
	return err


def ssim_compare(filename_image_a, filename_image_b, smooth_before_comparison=False, d={}):
	image_a = Image.open(filename_image_a).convert("RGB")
	image_b = Image.open(filename_image_b).convert("RGB")

	if smooth_before_comparison:
		image_a = image_a.filter(ImageFilter.SMOOTH)
		image_b = image_b.filter(ImageFilter.SMOOTH)
		image_b.save(os.path.splitext(filename_image_b)[0] + "_smooth" + os.path.splitext(filename_image_b)[1], "PNG")

	npa = []
	npb = []
	for px in image_a.getdata():
		npa.append(rgb_to_luminance(px[0], px[1], px[2]) * 0.00392156862745098) #  / 255)
	for px in image_b.getdata():
		npb.append(rgb_to_luminance(px[0], px[1], px[2]) * 0.00392156862745098) #  / 255)

	npa = np.asarray(npa)
	npb = np.asarray(npb)

	err = (1.0 - ssim(npa, npb)) * 100

	d["diff_sum_ssim"] = err
	return err


def main(aggressivity):
	make_tmp_folder()

	flist = os.listdir(input_folder)
	for file_idx, file_entry in enumerate(flist): # os.listdir(input_folder):
		if file_entry.endswith(input_file_ext):
			print(file_entry)
			optimized_files = optimize_as_indexed_colors(file_entry, input_folder, tmp_folder)
			# optimized_files = optimize_image_as_mozilla_jpeg(file_entry, input_folder, tmp_folder)

			difference_scores = []
			candidates = []

			for optimized_file in optimized_files:
				diff_dict = {"diff_sum_mse": 0.0, "diff_sum_ssim": 0.0, "diff_sum_naive": 0.0}
				# diff_sum_mse = mse_compare(os.path.join(input_folder, file_entry), optimized_file, False)
				# diff_sum_ssim = ssim_compare(os.path.join(input_folder, file_entry), optimized_file, False)
				# diff_sum_naive = naive_diff_compare(os.path.join(input_folder, file_entry), optimized_file, False)
				t_diff_sum_mse = threading.Thread(target=mse_compare, args=(os.path.join(input_folder, file_entry), optimized_file, False, diff_dict))
				t_diff_sum_ssim = threading.Thread(target=ssim_compare, args=(os.path.join(input_folder, file_entry), optimized_file, False, diff_dict))
				t_diff_sum_naive = threading.Thread(target=naive_diff_compare, args=(os.path.join(input_folder, file_entry), optimized_file, False, diff_dict))

				# t_diff_sum_ssim, clock = 1.432041893000001
				# t_diff_sum_mse, clock = 1.1640078779999996
				# t_diff_sum_naive, clock = 0.5053621260000014

				t = time.process_time()
				t_diff_sum_ssim.start()
				t_diff_sum_mse.start()
				t_diff_sum_naive.start()

				t_diff_sum_naive.join()
				t_diff_sum_mse.join()
				t_diff_sum_ssim.join()
				t = time.process_time() - t
				print("diff took " + str(t) + " s.")

				diff_sum_mse = diff_dict["diff_sum_mse"]
				diff_sum_ssim = diff_dict["diff_sum_ssim"]
				diff_sum_naive = diff_dict["diff_sum_naive"]

				diff_sum = (diff_sum_mse + diff_sum_ssim + diff_sum_naive) / 3.0

				# diff_sum = difference with the original (less is better)
				# ratio = difference * size (less is better)
				difference_scores.append(diff_sum)
				optimized_file_size = os.path.getsize(optimized_file)
				size_ratio = os.path.getsize(os.path.join(input_folder, file_entry)) * optimized_file_size
				print(optimized_file + ", diff = " + str(diff_sum)[0:6] + ", size ratio = " + str(size_ratio)[0:6])
				candidates.append({"filename": optimized_file, "difference": diff_sum, "compression": size_ratio, "size": optimized_file_size})
				
			median_score = statistics.median(difference_scores)
			min_score = min(difference_scores)
			print("Median score = " + str(median_score))
			print("Minimum score = " + str(min_score))
			median_score = median_score * (1.0 - aggressivity) + min_score * aggressivity
			print("Aggressive median score = " + str(median_score))

			# search for the candidates that reach the desired quality
			quality_candidates = []
			for candidate in candidates:
				if candidate["difference"] <= median_score:
					quality_candidates.append(candidate)


			print("found " + str(len(quality_candidates)) + " candidates with aggressivity = " + str(aggressivity))

			# # search for the smallest candidate
			# best_candidate = quality_candidates[0]
			# for candidate in quality_candidates:
			# 	if candidate["size"] < best_candidate["size"]:
			# 		best_candidate = candidate

			# search for the candidate with the best ratio
			best_candidate = quality_candidates[0]
			for candidate in quality_candidates:
				if candidate["compression"] < best_candidate["compression"]:
					best_candidate = candidate

			# best_candidate = None
			# # best_candidate = candidates[0]
			# for candidate in candidates:
			# 	if candidate["difference"] <= median_score:
			# 		if candidate["compression"] > best_candidate["compression"]:
			# 			best_candidate = candidate

			print("Best difference/compression ratio found:")
			print(best_candidate)

			shutil.copy(best_candidate["filename"], os.path.join(output_folder, os.path.splitext(file_entry)[0] + os.path.splitext(best_candidate["filename"])[1]))


main(aggressivity)