from __future__ import print_function
from __future__ import division

import sys
sys.path.insert(0, 'src')
import argparse
import numpy as np
import transform, vgg, pdb, os, subprocess
import tensorflow as tf
import cv2
from datetime import datetime
import logging as log
from PIL import Image
import scipy.ndimage as ndimage





#entendable model container
models=[{"ckpt":"models/ckpt_leafveins/fns.ckpt", "style":"styles/leafveins.png"}, #sosososo good. try with more style weight #################
	{"ckpt":"models/ckpt_fruitveg/fns.ckpt", "style":"styles/fruitveg.jpeg"},
	{"ckpt":"models/ckpt_solar_filament/fns.ckpt", "style":"styles/solarfilament.jpg"},
	{"ckpt":"models/ckpt_kusama_content4e1/fns.ckpt", "style":"styles/kusama.jpg"},
	#{"ckpt":"models/ckpt_0026/fns.ckpt", "style":"styles/0032.png"}, #pretty amazing
	#{"ckpt":"models/ckpt_chameleon_tail/fns.ckpt", "style":"styles/chameleon_tail.jpg"}, #this can pass as a snake one and its really good
	#{"ckpt":"models/ckpt_egon_shiele_embrace/fns.ckpt", "style":"styles/schiele_embrace.jpg"}, #this is pretty awesome. maybe increase style ##########
	{"ckpt":"models/ckpt_snakeface5_content7e1/fns.ckpt", "style":"styles/snakeface5.jpg"}, #this is definitely the best content weight so far. Even more if there's time
	{"ckpt":"models/ckpt_fractalleaf_content3e1/fns.ckpt", "style":"styles/fractalleaf.jpg"},
	{"ckpt":"models/ckpt_fractalleaf2/fns.ckpt", "style":"styles/fractalleaf2.jpg"}, #pretty good
	{"ckpt":"models/ckpt_fractallightning_content1.5e1/fns.ckpt", "style":"styles/fractallightning.jpg"}, #YES!
	#{"ckpt":"models/ckpt_fractalspirals/fns.ckpt", "style":"styles/fractalspirals.jpg"}, #this is one of the best fractal ones
	{"ckpt":"models/ckpt_snakeface4/fns.ckpt", "style":"styles/snakeface4.png"}, #this has a really good color/effect on faces, but increase style weight
	{"ckpt":"models/ckpt_fractaltexture/fns.ckpt", "style":"styles/fractaltexture.jpg"}, #interesting
	{"ckpt":"models/ckpt_lotsasnakes4/fns.ckpt", "style":"styles/lotsasnakes4.jpg"}, #VERY on theme, maybe a tad more style
	#{"ckpt":"models/ckpt_fractaltexture2_content1.5e1/fns.ckpt", "style":"styles/fractaltexture2.jpg"}, #better!
	#{"ckpt":"models/ckpt_fractaltexture4/fns.ckpt", "style":"styles/fractaltexture4.jpg"}, #very colorful
	{"ckpt":"models/ckpt_fruitveg/fns.ckpt", "style":"styles/fruitveg.jpeg"}, #this is by far the best veg one, MUST HAVE
	{"ckpt":"models/ckpt_groundsel_flower_3e2/fns.ckpt", "style":"styles/groundsel_flower.jpg"}, #this is the best weight for this particular style pic
	#{"ckpt":"models/ckpt_haeckel_actiniae3e2/fns.ckpt", "style":"styles/haeckel_actiniae.jpg"}, #trippy
	#{"ckpt":"models/ckpt_hela2/fns.ckpt", "style":"styles/hela2.jpg"}, #amazing
	{"ckpt":"models/ckpt_hieroglyphs/fns.ckpt", "style":"styles/hieroglyphs.jpg"}, #pretty good
	{"ckpt":"models/ckpt_klee/fns.ckpt", "style":"styles/klee.jpg"}, #this one has a very contrasty, heavy style would be interesting
	{"ckpt":"models/ckpt_klimt_life_death_content1.5e1/fns.ckpt", "style":"styles/klimt-death-and-life.jpg"}, #this is even better
	#{"ckpt":"models/ckpt_kokoschka/fns.ckpt", "style":"styles/kokoschka.jpg"}, #this is growing on me 
	{"ckpt":"models/ckpt_kusama_content4e1/fns.ckpt", "style":"styles/kusama.jpg"},#this is great but even more content weight
	#{"ckpt":"models/ckpt_legopattern_content5e1/fns.ckpt", "style":"styles/lego_pattern.jpg"}, #best one out of legopattern
	#{"ckpt":"models/ckpt_legos_hires/fns.ckpt", "style":"styles/legos_high_res.jpg"}, #awesome colors, awesome patterns
	{"ckpt":"models/ckpt_lotsasnakes/fns.ckpt", "style":"styles/lotsasnakes.jpg"}, #VERY on theme
	{"ckpt":"models/ckpt_lotsasnakes2/fns.ckpt", "style":"styles/lotsasnakes2.jpg"}, #too much style weight
	#{"ckpt":"models/ckpt_mandlebrot_content1.5e1/fns.ckpt", "style":"styles/mandlebro.jpg"}, #better with less content style
	#{"ckpt":"models/ckpt_muse/la_muse.ckpt", "style":"styles/muse.jpg"}, #awesome
	#{"ckpt":"models/ckpt_my_girls/fns.ckpt", "style":"styles/my_girls.jpg"}, #pretty good
	#{"ckpt":"models/ckpt_nate/fns.ckpt", "style":"styles/nate_colors.jpg"}, #homage
	{"ckpt":"models/ckpt_naturalfractal/fns.ckpt", "style":"styles/naturalfractal.jpg"}, #very good, very woodsy
	{"ckpt":"models/ckpt_nefertiri/fns.ckpt", "style":"styles/nefertiri.jpg"}, #this NEEEEED MORE STYLE
	#{"ckpt":"models/ckpt_okeeffe_5e2/fns.ckpt", "style":"styles/okeeffe.jpg"}, #still not as good as I hoped
	#{"ckpt":"models/ckpt_olivesbowl/fns.ckpt", "style":"styles/olivesbowl.jpg"}, #interesting color
	{"ckpt":"models/ckpt_orchard_cabbage/fns.ckpt", "style":"styles/orchard_cabbage.jpg"}, #contender
	#{"ckpt":"models/ckpt_picasso_abstract/fns.ckpt", "style":"styles/picasso_abstract.jpg"}, #very good
	{"ckpt":"models/ckpt_pointilism_content5e1/fns.ckpt", "style":"styles/pointilism.jpg"}, #pretty good
	{"ckpt":"models/ckpt_poussette_dart/fns.ckpt", "style":"styles/the_transcendental.jpg"}, #dark and crazy
	#{"ckpt":"models/ckpt_resized_pillars/fns.ckpt", "style":"styles/resized_pillars_of_creation.jpg"}, #pretty great
	{"ckpt":"models/ckpt_roman/fns.ckpt", "style":"styles/roman.jpg"}, #yes!
	{"ckpt":"models/ckpt_sandcracks/fns.ckpt", "style":"styles/sandcracks.jpg"}, #very good
	#{"ckpt":"models/ckpt_mandlebrot2/fns.ckpt", "style":"styles/mandlebrot2.jpg"}, #pretty good
	#{"ckpt":"models/ckpt_scream/scream.ckpt", "style":"styles/the_scream.jpg"}, #very good
	{"ckpt":"models/ckpt_seuss/fns.ckpt", "style":"styles/seuss.jpg"}, #this ones silly af
	{"ckpt":"models/ckpt_snakeface/fns.ckpt", "style":"styles/snakeface.png"}, #very scaley
	#{"ckpt":"models/ckpt_flowpaint_content1.5e1/fns.ckpt", "style":"styles/flowpaint.jpg"}, #great and different!
	#{"ckpt":"models/ckpt_snakeface2/fns.ckpt", "style":"styles/snakeface2.jpg"}, #very wild looking
	{"ckpt":"models/ckpt_snakescales/fns.ckpt", "style":"styles/snakescales.jpg"}, #not bad
	#{"ckpt":"models/ckpt_spacecat/fns.ckpt", "style":"styles/spacecat.jpg"}, #pretty ridiculous, decrease style weight?
	{"ckpt":"models/ckpt_spainfractals/fns.ckpt", "style":"styles/spainfractals.jpg"}, #try with more style
	#{"ckpt":"models/ckpt_starrynight/fns.ckpt", "style":"styles/starrynight.jpg"},#awesome, maybe a tad more style >2.5e2
	{"ckpt":"models/ckpt_hexagon1/fns.ckpt", "style":"styles/hexagon1.jpg"},
	{"ckpt":"models/ckpt_derain/fns.ckpt", "style":"styles/derain.jpg"}, #pretty good
	{"ckpt":"models/ckpt_aluminumfoil/fns.ckpt", "style":"styles/leafveins.png"},
	{"ckpt":"models/ckpt_fellstrees/fns.ckpt", "style":"styles/hexagon1.jpg"}]
	#{"ckpt":"models/ckpt_fractal_painting/fns.ckpt", "style":"styles/fractal_painting.jpg"}]

def rotate_velocity_map(velocity_map, angle_degrees):
    height, width, _ = velocity_map.shape
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1)
    rotated_velocity_map = cv2.warpAffine(velocity_map, rotation_matrix, (width, height))
    return rotated_velocity_map


def apply_zoom1(image, zoom_scale, sigma=1):
    height, width, _ = image.shape
    zoomed_image = cv2.resize(image, (int(width * zoom_scale), int(height * zoom_scale)))

    # Calculate the difference between the zoomed and original dimensions
    delta_w = zoomed_image.shape[1] - width
    delta_h = zoomed_image.shape[0] - height

    # Create a Gaussian filter with the given sigma value
    x = np.linspace(-width // 2, width // 2, width)
    y = np.linspace(-height // 2, height // 2, height)
    x, y = np.meshgrid(x, y)
    gaussian_filter = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    # Normalize the Gaussian filter
    gaussian_filter /= np.max(gaussian_filter)

    # Compute the circular velocity map
    velocity_map = np.dstack([gaussian_filter] * 3)

    # Calculate shifted indices
    y_shifts = (delta_h * velocity_map[:, :, 0]).astype(int)
    x_shifts = (delta_w * velocity_map[:, :, 1]).astype(int)
    y_indices = np.clip(np.arange(height).reshape(-1, 1) + y_shifts, 0, zoomed_image.shape[0] - 1)
    x_indices = np.clip(np.arange(width).reshape(1, -1) + x_shifts, 0, zoomed_image.shape[1] - 1)

    # Apply the velocity map to the zoomed image
    output = zoomed_image[y_indices, x_indices]

    return output

def apply_zoom2(image, zoom_scale, angle_degrees, sigma=30):
    height, w = image.shape
    zoomed_image = cv2.resize(image, (int(width * zoom_scale), int(height * zoom_scale)))

    delta_w = zoomed_image.shape[1] - width
    delta_h = zoomed_image.shape[0] - height

    x = np.linspace(-width // 2, width // 2, width)
    y = np.linspace(-height // 2, height // 2, height)
    x, y = np.meshgrid(x, y)
    gaussian_filter = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian_filter /= np.max(gaussian_filter)
    velocity_map = np.dstack([gaussian_filter] * 3)

    # Rotate the velocity map
    rotated_velocity_map = rotate_velocity_map(velocity_map, angle_degrees)

    y_shifts = (delta_h * rotated_velocity_map[:, :, 0]).astype(int)
    x_shifts = (delta_w * rotated_velocity_map[:, :, 1]).astype(int)
    y_indices = np.clip(np.arange(height).reshape(-1, 1) + y_shifts, 0, zoomed_image.shape[0] - 1)
    x_indices = np.clip(np.arange(width).reshape(1, -1) + x_shifts, 0, zoomed_image.shape[1] - 1)

    output = zoomed_image[y_indices, x_indices]

    return output

#def apply_zoom(image, max_zoom, angle, sigma=30):
def apply_zoom3(image, max_zoom, angle, sigma=50, downscale_factor=4):
    height, width, _ = image.shape

    image = cv2.resize(image, (width // downscale_factor, height // downscale_factor))
    height, width, _ = image.shape

    center_x, center_y = width // 2, height // 2
    x = np.linspace(-center_x, center_x, width)
    y = np.linspace(-center_y, center_y, height)
    x, y = np.meshgrid(x, y)
    gaussian_filter = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian_filter /= np.max(gaussian_filter)
    velocity_map = np.dstack([gaussian_filter] * 2)

    # Rotate the velocity map
    rotation_matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)
    rotated_velocity_map = cv2.warpAffine(velocity_map, rotation_matrix, (width, height))

    # Compute the corresponding grid coordinates
    y, x = np.indices((height, width)).astype(float)
    x += max_zoom * rotated_velocity_map[:, :, 0]
    y += max_zoom * rotated_velocity_map[:, :, 1]

    # Clip the coordinates and reshape them into a format suitable for cv2.remap
    #x = np.clip(x, 0, width - 1).flatten()
    #y = np.clip(y, 0, height - 1).flatten()
    x = np.clip(x, 0, width - 1).astype(np.float32).flatten()
    y = np.clip(y, 0, height - 1).astype(np.float32).flatten()
    coordinates = np.vstack((x, y)).astype(np.float32)

    # Apply the remap transformation
    #zoomed_image = cv2.remap(image, coordinates, None, cv2.INTER_LINEAR)

    zoomed_image = cv2.resize(zoomed_image, (width * downscale_factor, height * downscale_factor))
    return zoomed_image

    return zoomed_image

def apply_zoom(image, zoom_scale, rotation_angle, sigma=50, downscale_factor=2):
    height, width, _ = image.shape
    image = cv2.resize(image, (width // downscale_factor, height // downscale_factor))
    height, width, _ = image.shape
    
    # Create the meshgrid for the image coordinates
    x = np.linspace(-width // 2, width // 2, width)
    y = np.linspace(-height // 2, height // 2, height)
    xv, yv = np.meshgrid(x, y)
    
    # Calculate the Gaussian weight matrix
    weight_matrix = np.exp(-((xv**2) + (yv**2)) / (2 * sigma**2))
    
    # Calculate the radial zoom scale by adding the weighted matrix to the base zoom scale
    radial_zoom = zoom_scale + weight_matrix
    
    # Calculate the rotation matrix
    rotation_angle = np.deg2rad(rotation_angle)
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                [np.sin(rotation_angle), np.cos(rotation_angle)]])
    
    # Apply the rotation and radial zoom to the meshgrid coordinates
    coordinates = np.dot(np.stack((xv, yv), axis=-1), rotation_matrix.T)
    coordinates[..., 0] = (coordinates[..., 0] * radial_zoom) + (width // 2)
    coordinates[..., 1] = (coordinates[..., 1] * radial_zoom) + (height // 2)
    
    # Apply the remapped coordinates to the input image
    zoomed_image = cv2.remap(image, coordinates.astype(np.float32), None, cv2.INTER_LINEAR)
    
    # Upscale the zoomed_image back to the original size
    zoomed_image = cv2.resize(zoomed_image, (width * downscale_factor, height * downscale_factor))
    
    return zoomed_image



def load_checkpoint(checkpoint, sess):
        saver = tf.train.Saver()
        try:
                saver.restore(sess, checkpoint)
                style = cv2.imread(checkpoint)
                return True
        except:
                print("checkpoint %s not loaded correctly" % checkpoint)
                return False


def get_camera_shape(cam):
        """ use a different syntax to get video size in OpenCV 2 and OpenCV 3 """
        cv_version_major, _, _ = cv2.__version__.split('.')
        if cv_version_major == '3':
                return cam.get(cv2.CAP_PROP_FRAME_WIDTH), cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        else:
                return cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH), cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)


#def image_blender_ratio(frame_num):
	#assumes 25 fps
	# TODO add a way to sequence different blending functions
	#return .2*np.sin(2*np.pi/125*frame_num)+.5 #too much on nefertiri
	#return .1*np.sin(2*np.pi/125*frame_num)+.4 #best pure sin()
#	return -(np.abs(.4*np.sin(2*np.pi/125*frame_num))) +.2 #this is slightly too rigid at lower bound

def image_blender_ratio(frame_num, min_ratio=0.2, max_ratio=0.8):
	random_factor = np.random.uniform(-0.1, 0.1)
	blend_ratio = 0.5 * np.sin(2 * np.pi / 125 * frame_num) + 0.5 + random_factor
    	blend_ratio = np.clip(blend_ratio, min_ratio, max_ratio)
    	return blend_ratio

# argparser
parser = argparse.ArgumentParser()
parser.add_argument('--device_id', type=int, help='camera device id (default 0)', required=False, default=0)
parser.add_argument('--width', type=int, help='width to resize camera feed to (default 320)', required=False, default=1500)
parser.add_argument('--display_width', type=int, help='width to display output (default 640)', required=False, default=1000)





def main(device_id, width, display_width):

        idx_model = 0
	first_frame=True
	frame_number=0
        device_t='/gpu:0'
	save_number=500
        g = tf.Graph()
        #allow it to run on CPU if no GPU
        soft_config = tf.ConfigProto(allow_soft_placement=True)
        soft_config.gpu_options.allow_growth = True
        with g.as_default(), g.device(device_t), tf.Session(config=soft_config) as sess:
                cam = cv2.VideoCapture(device_id) 
		cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)   
                cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, 1)
                cam_width, cam_height = get_camera_shape(cam)
                if(width % 4 != 0):
                    width=width + 4 - (width % 4)

                height = int(width * float(cam_height/cam_width))

                if(height % 4 != 0):
                    height=height + 4 - (height % 4)

                img_shape = (height, width, 3)
                batch_shape = (1,) + img_shape
                print("batch shape", batch_shape)


                #create placeholder and feed it to tf
                img_placeholder = tf.placeholder(tf.float32, shape=batch_shape, name='img_placeholder')
                predicates = transform.net(img_placeholder)

                # load checkpoint
                load_checkpoint(models[idx_model]["ckpt"], sess)
                style = cv2.imread(models[idx_model]["style"])

               
		print("enter cam loop")

                # enter cam loop
		# only take one image
		
		frame_num=0
		start_angle=0
		# Random zoom range
		min_zoom_ratio = 0.9
		max_zoom_ratio = 0.99
		interp_factor= 0.4
		prev_output=None
		# Initialize rotation angle
    		rotation_angle = 0

		# Keep track of the last 4 frames
		frames = [None, None, None, None]

                while True:
			if(start_angle==360):
				start_angle=0
			
        		if(frame_num % 20 >= 0 and frame_num % 20 < 4):  # Capture every 20th, 21st, 22nd, 23rd frames
        			ret, frame = cam.read()
        			frame = cv2.resize(frame, (width, height))
        			frame = cv2.flip(frame, 1)

        			frames[frame_num % 4] = frame  # Store the current frame in the frames list

        		# Start blending from the 20th frame
        		if frames[0] is not None:
            			# Initial blending ratio for the 20th frame
            			alpha = 0.1

            			blended_frame = frames[0]
            			for i in range(1, 4):
               				if frames[i] is not None:
                    				alpha += 0.3  # Increase blending ratio linearly
                    				beta = 1.0 - alpha
                    				blended_frame = cv2.addWeighted(frames[i], alpha, blended_frame, beta, 0.0)

            			frame = blended_frame
        		else:
            			output = cv2.resize(output, (width, height))
            			frame = output

        		#if frame_num == save_number:
            		#	break

        		X = np.zeros(batch_shape, dtype=np.float32)
        		X[0] = frame


			output = sess.run(predicates, feed_dict={img_placeholder:X})

			#for x in range(2):
				#print("Repeat_                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   passes_key %s" %repeat_passes_key) 	
			#	output = sess.run(predicates, feed_dict={img_placeholder:X})
			#	X[0]=output
				
			#normalize and resize
                        # Process the output image obtained from the neural network
			output = output[:, :, :, [2, 1, 0]].reshape(img_shape)
			output = np.clip(output, 0, 255).astype(np.uint8)

			# Calculate output height and width
			output_height, output_width, _ = output.shape

			# Convert the original frame and the output image to PIL Image objects
			frame_image = Image.fromarray(frame)
			output_image = Image.fromarray(output)

			# Blend the original frame and the output image
			blended_output = Image.blend(frame_image, output_image, image_blender_ratio(frame_num))

			# Apply the zooming effect
			#zoom_factor = 0.05  # Adjust this value to control the zooming intensity
			#crop_dim = (
    			#	int(output_width * zoom_factor),
    			#	int(output_height * zoom_factor),
    			#	int(output_width * (1 - zoom_factor)),
    			#	int(output_height * (1 - zoom_factor))
			#)
			#zoomed_output = blended_output.crop(crop_dim)
			#output = zoomed_output.resize((output_width, output_height), Image.ANTIALIAS)

			# Zoom effect 1 
    			#output_height, output_width, _ = output.shape
    			#zoom_ratio = np.random.uniform(min_zoom_ratio, max_zoom_ratio)
    			#new_width = int(output_width * zoom_ratio)
    			#new_height = int(output_height * zoom_ratio)
    
   			 # Crop the center of the image
    			#left = (output_width - new_width) // 2
    			#top = (output_height - new_height) // 2
    			#right = left + new_width
    			#bottom = top + new_height
    			#crop_dim = (left, top, right, bottom)
    			#output_image = Image.fromarray(output)
    			#cropped_image = output_image.crop(crop_dim)

    			# Resize the cropped image back to the original size
    			#output = np.array(cropped_image.resize((output_width, output_height), Image.ANTIALIAS))

        		# Perform zoom effect 2
        		output_height, output_width, _ = output.shape
        		zoom_scale = np.random.uniform(1, 1.0000005)
        		output = apply_zoom1(output, zoom_scale)

			# Perform zoom effect with Gaussian velocity profile and rotation
			#output_height, output_width, _ = output.shape
			#zoom_scale = np.random.uniform(1.0, 1.1)
			#zoom_scale = 1.1
			#rotation_angle = frame_num % 360  # Change the rotation angle with the frame number
			#output = apply_zoom(output, zoom_scale, rotation_angle, sigma=50)

			# Apply smoother transitions between frames
        		if prev_output is not None:
            			output = cv2.addWeighted(output, 1 - interp_factor, prev_output, interp_factor, 0)

			# Apply rotation (swirling)
        		#rotation_angle += 1
        		#M = cv2.getRotationMatrix2D((output_width // 2, output_height // 2), rotation_angle, 1)
        		#output = cv2.warpAffine(output, M, (output_width, output_height))

        		# Update previous output
        		prev_output = output.copy()

       			# Display the result
        		cv2.imshow('frame', output)

			# Convert the zoomed output back to a NumPy array and display it
			#output = np.array(zoomed_output)
			#cv2.imshow('frame', output)
			#filename='./zoom/frame-'+str(frame_num).zfill(4)+'.jpg'
			#cv2.imwrite(filename,output)
			print("Frame: ",frame_num)
			frame_num=frame_num+1
                        cv2.imshow('frame', output)
			image=Image.fromarray(output)
			#image=image.rotate(start_angle)

			
			
			crop_dim=(output_width//100,output_height//100,99*output_width//100,99*output_height//100)
			image=image.crop(crop_dim)
			output=np.array(image)
			#if(frame_number==3):

				#idx_model = (idx_model + len(models) - 1) % len(models)
				#print("load %d / %d : %s " % (idx_model, len(models), models[idx_model]))
				#load_checkpoint(models[idx_model]["ckpt"], sess)
				#style = cv2.imread(models[idx_model]["style"])
			#	frame_number=0

			frame_number=frame_number+1

			#start_angle=start_angle+1

                        key_ = cv2.waitKey(1)
                        if key_ == 27:
                                break
			elif key_ == ord('a'):
		
				idx_model = (idx_model + len(models) - 1) % len(models)
				print("load %d / %d : %s " % (idx_model, len(models), models[idx_model]))
				load_checkpoint(models[idx_model]["ckpt"], sess)
				style = cv2.imread(models[idx_model]["style"])
				#start with new seed
				first_frame=True
				start_angle=0
				frame_num=0

                # done
                cam.release()
                cv2.destroyAllWindows()


if __name__ == '__main__':
        opts = parser.parse_args()
        main(opts.device_id, opts.width, opts.display_width),
