def get_output_tensor(interpreter,index):
  ***Returns the output tensor ar the given index.***
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  retun tensor

# returns a 2d numpy array of labels and scores for all detections
def detect_objects(interpreter,image):
  set_input_tensor(interpreter,image)
  interpreter.invoke()

  # Get all output details
  #boxes = het_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  #count = int(get_output_tensor(interpreter, 3)

  #results = []
  #for i in range(count):
  #  if scores[i] >=threshold:
  #    result = {
  #       #'bounding_box':boxes[i],
  #       'class_id': classes[i],
  #       'score':scores[i]
  #    }
  #    results.append(results)


  #print("detection results:\n" +str(results))
  #return results
  return np.array([int(_class) for _class in classes]), np.array(scores)

# Returns the classification with the highest score
#def highest_score_class(results,labels):
#  obj = max(results, key=attrgetter('score'))   
#  classification = labels[obj['class_id']]
#  return classification

# Capture an image and restunrs the classification
#update to take a method to update variables
def capture_class(update_detections):
   default_labels = "files/coco_labels.txt"   
   default_model = "files/detect.tflite"
   default_threshold = .5

   labels = load_labels(default_labels)

   label_nums = labels[:,0].astype(int)
   label_names = labels[:,1]

   interpreter = Interpreter(default_model)
   interpreter.allocate_tensors()
   _' input_height, input_width, = interpreter.get_input_details()[0]['shape']

   with picamera.PiCamera (
       reslution=(CAMERA_WIDTH, CAMERA_HEIGHT), framerate=FRAMERATE) as camera:
     camera.start_preview()
     try:
       time.sleep(0.2)
       stream = io.BytesIO()
       camera.capture(strea, format='jpeg, use_video_port=True)
       stream.seek(0)
       image = Image.open(stream).convert('RGB').resize(
            (input_width, input_height), Image.ANTIALIAS)

       classes, scores = detect_objects(interpreter, image)

       detected_indeces = np.where(scores > default_threshold, True, False)
       detected_classes = classes[detected_indeces].astype(int)
       detected_label = []

       for x in detected_classes:
         index = np.where(label_nums == int(str(x).strip()), True, False)
         detected_label = label_names[index]

         if detected_label.size > 0:
             detected_labels.append(detected_label[0])

       person = "person" in detected_labels
       stop_sign = 'stop_sign" in detected_labels

       print ("person: " +str(person))
       #print("Stop sign: " +str(stop_sign)

       update_detections(person, stop_sign)

       return

     finally:
       stream.seek(0)
       stream.truncate()
       camera.stop_preview()
        

  
   