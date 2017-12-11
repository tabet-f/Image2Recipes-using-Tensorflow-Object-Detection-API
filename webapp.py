#PATHS MUST BE CONFIGURED BASED ON YOUR IMPLEMENTATION
import sys
import os
from flask import Flask, session render_template, send_from_directory, request, Response, redirect,url_for,jsonify
from werkzeug import secure_filename
from functools import wraps
import base64
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import numpy as np
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import zipfile
import json
from collections import defaultdict
from PIL import Image



#Append 1 dir to 'research dir'
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util


#where the model will look for the image
UPLOAD_FOLDER='test_images'
 
#Used to EDIT json file (Admin Use Only)
#must have @login.request removed here
@app.route('/editjson', methods=['GET'])
@login_required
def editjson(): 
     SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
     json_url = os.path.join(SITE_ROOT, "static/data", "recipes.json")
     data1 = json.load(open(json_url))
     data = json.dumps(data1, sort_keys=True, indent=4, separators=(',', ': '))
     return render_template('editjson.html', data=data)

#Used to SAVE json file (Admin Use Only) 
#must have @login.request removed here
@app.route('/savejson', methods=['GET','POST'])
def savejson(): 
    jsonstring = request.form['editedjson']
    dataFile = open("static/data/recipes.json", "w")
    dataFile.write(jsonstring)
    dataFile.close()

    SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
    json_url = os.path.join(SITE_ROOT, "static/data", "recipes.json")
    data1 = json.load(open(json_url))
    data = json.dumps(data1, sort_keys=True, indent=4, separators=(',', ': '))
    return render_template('editjson.html', data=data, message="saved")
    


#Used to show json file
@app.route('/getjson', methods=['GET','POST'])
def getjson(): 
     SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
     json_url = os.path.join(SITE_ROOT, "static/data", "recipes.json")
     data1 = json.load(open(json_url))
     data = json.dumps(data1, sort_keys=True, indent=4, separators=(',', ': '))
     return render_template('embedjson.html', data=data)

#Used to show test data
@app.route('/gettestdata', methods=['GET'])
def gettestdata(): 
     import fnmatch
     #load img urls from dir
     images_url = os.listdir(os.path.join(app.static_folder, "dataset/test"))
     img_jpg_url = fnmatch.filter(images_url, '*.jpg')#only jpg will be sent
     return render_template('gallery-test.html', images=img_jpg_url)

#Used to show train data
@app.route('/gettraindata', methods=['GET'])
def gettraindata(): 
     import fnmatch
     #load img urls from dir
     images_url = os.listdir(os.path.join(app.static_folder, "dataset/train"))
     img_jpg_url = fnmatch.filter(images_url, '*.jpg')#only jpg will be sent
     return render_template('gallery-train.html', images=img_jpg_url)


#Used to show sample data 
@app.route('/getsampledata', methods=['GET'])
def getsampledata(): 
     import fnmatch
     #load img urls from dir
     images_url = os.listdir(os.path.join(app.static_folder, "dataset/sample"))
     img_jpg_url = fnmatch.filter(images_url, '*.jpg')#only jpg will be sent
     return render_template('gallery-sample.html', images=img_jpg_url)


#Used to show the final result
@app.route('/getrecipes', methods=['GET'])
def getrecipes(): 
   
   SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
   json_url = os.path.join(SITE_ROOT, "static/data", "recipes.json")
   data = json.load(open(json_url))
   availableIng = session['ing']
   objectsnscores = session['objectsnscores']
   

   file_handler = open('static/data/recipes.json', 'r')
   parsed_data = json.loads(file_handler.read())
   
   fullMatchRecipes = {}
   matchedRecipes=0 #matched recipes counter
   unmatchedRecipes=0 #unmatched recipes counter
   i=1
   
   for item in parsed_data['recipes']:      
        for key in item:            
            if key =='ingredients':
                 #All JSON recipes objects element must be presented below in the 'FULL MATCH' if statement to be shown on the web
               if item [key] == availableIng:
                 print("Full Match")
                 #Storing Matched Recipes Info
                 fullMatchRecipes[('id'+"_"+str(i))] = item['id']
                 fullMatchRecipes[('title'+"_"+str(i))] = item['title'] 
                 fullMatchRecipes[('image'+"_"+str(i))] = item['image'] 
                 fullMatchRecipes[('preparation time'+"_"+str(i))] = item['preparation time'] 
                 fullMatchRecipes[('instructions'+"_"+str(i))] = item['instructions'] 
                 fullMatchRecipes[('link'+"_"+str(i))] = item['link']
                 fullMatchRecipes[('ingredients'+"_"+str(i))] = item['ingredients'] 
                 i+=1
                 matchedRecipes+=1
               else:
                 unmatchedRecipes+=1
                 print("Sorry, NO Recipe(s) Match your Ingredients!")
               
   #Storing General Info    
   fullMatchRecipes['totalFullMatchRecipes'] = matchedRecipes
   fullMatchRecipes['totalUnMatchRecipes'] = unmatchedRecipes
   fullMatchRecipes['totalRecipes'] = (unmatchedRecipes+matchedRecipes)
   print(fullMatchRecipes)
   o = [1,2,3]
   return render_template('getrecipes.html', data=fullMatchRecipes)


    
#Remove Caching    
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response 

#Index page 
@app.route('/',methods=['POST','GET'])
def index():
    if request.method == 'POST':
        #Reciving the image from the POST request and do some validation
        upload_file = request.files['file']
        filename = secure_filename(upload_file.filename)
        if filename=='':
          print('No Image Uploaded')
          return '<h2>No Image Uploaded<h2>'
        elif not filename.endswith('.jpg'):
          print('Only jpg Extension Allowed')
          return '<h2>Only jpg Extension Allowed<h2>'
        else:
         print('>>> filename [',filename,'] uploaded successfully')
         upload_file.save(os.path.join(UPLOAD_FOLDER, 'image1.jpg'))
         image_size=128
         #TF Deployment Start******************
         num_channels=3
         images = []
         MODEL_NAME = 'tngGRAPH' #your model dir
         PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb' #your actual model
         PATH_TO_LABELS = os.path.join('training', 'myfirstobjectdetection.pbtxt') #your label map
         NUM_CLASSES = 3 #num of classess or objects you have
         detection_graph = tf.Graph() 
         with detection_graph.as_default():
           od_graph_def = tf.GraphDef()
           with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
             serialized_graph = fid.read()
             od_graph_def.ParseFromString(serialized_graph)
             tf.import_graph_def(od_graph_def, name='')
        

         label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
         categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
         category_index = label_map_util.create_category_index(categories)

	 #img to numpy
         def load_image_into_numpy_array(image):
           (im_width, im_height) = image.size
           return np.array(image.getdata()).reshape(
             (im_height, im_width, 3)).astype(np.uint8)


         #Specify the TEST_IMAGES_DIR and the range of images inside it (here we just have 1 image)
         PATH_TO_TEST_IMAGES_DIR = 'test_images'
         TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 2) ]
         IMAGE_SIZE = (12, 8)


	 #the detection against the img 
         with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
             objectsnscores = []
             objects = []
             image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
             detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
             detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
             detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
             num_detections = detection_graph.get_tensor_by_name('num_detections:0')
             for image_path in TEST_IMAGE_PATHS:
                image = Image.open(image_path)
                image_np = load_image_into_numpy_array(image)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                (boxes, scores, classes, num) = sess.run(
                  [detection_boxes, detection_scores, detection_classes, num_detections],
                  feed_dict={image_tensor: image_np_expanded})
     

                #get the unique objects with no scores
                i=0
                for y in [category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.5 ]:
                 for x in y.values():
                   if isinstance(x, str):
                      if x not in objects:
                        objects.append(x)
                        #print(x,scores[0][i])
                        i+=1

		#get all the objects with their scores
                j=0
                for a in [category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.5 ]:
                 for b in a.values():
                   if isinstance(b, str):
                      ingnscore = b+"_"+str(float("{0:.4f}".format(scores[0][j])))
                      objectsnscores.append(ingnscore)
                      j+=1 
 
             #TF Deployment ENDS******************
             # Visualization of the results of a detection. you can use matplotlib
            vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=2)
            #instead of using matlpotlib save the image using scipy
            import scipy.misc
            scipy.misc.imsave('static/tfOutput.jpg', image_np)
            _OBJECTS = list(objects)
            session['ing'] = objects
            session['objectsnscores'] = objectsnscores
            return render_template('result.html',response=objects, objwithscore=objectsnscores)

           

    return  '''
    PUT YOUR HOME INDEX HTML CODE HERE - NOT INCLUDED WRITE YOUR CUSTOM ONE
    '''


#run flask  
if __name__ == '__main__':
    app.secret_key = 'YOUR SECRET KEY'
    app.config["CACHE_TYPE"] = "NULL"
    app.run(host="0.0.0.0", port=int("5000"), debug=False, use_reloader=False)#set to True during development    
