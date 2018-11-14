import io, os, sys, requests, pandas, threading, csv, time
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "mygoogleapi.json"
from PIL import Image
import numpy as np

# Import the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

# ---------------------------------
im_dir = "images/"
out_dir = "output/"
label_out_name = "labels.csv"
sleep_time = 5 #seconds
num_threads = 5
# ---------------------------------

# Instantiates a client
client = vision.ImageAnnotatorClient()

# Keep the first 5 results
num_of_results = 5

def process_request(index, file_name, label_results):

    label_row = [index, file_name, " ", " ", " ", " ", " ", " ", " ", " ", " ", " "]

    # Read image content
    with io.open(im_dir + file_name, 'rb') as image_file:
        content = image_file.read()

    # Convert image to Google Vision image
    image = vision.types.Image(content=content)

    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations

    # Iterate through Vision API result
    labelInd = 2 # skip the first two columns (index and file_name)
    for ind, label in enumerate(labels):
        if(ind<num_of_results): # Get first 5 results
            label_row[labelInd] = label.description
            labelInd += 1
            label_row[labelInd] = label.score
            labelInd += 1
        else:
        	break

    # Add label data for the image
    label_results[index] = label_row

    return label_results

# Create columns for the label output
label_columns = ['index', 'filename', 'label1', 'score1', 'label2', 'score2', 'label3', 'score3', 'label4', 'score4', 'label5', 'score5']

files = os.listdir(im_dir)

num_files = len(files)

print(num_files, "files found")

# Create placeholder for the data
label_results = [None] * num_files

for i in range(0, num_files-1, num_threads):
    threads = []
    for j in range(i, i + num_threads):
        if j>(num_files-1):
            break
        # Create thread for processing the request
        t = threading.Thread(target=process_request, args=(j, files[j], label_results))
        threads.append(t)
    [t.start() for t in threads]
    [t.join() for t in threads]

    print(i + len(threads), "of", num_files, "processed")

    # Write the label result
    with open(out_dir + label_out_name, "a") as output:
        writer = csv.writer(output, delimiter=',')
        if i == 0:
        	writer.writerow(kk for kk in label_columns)
        for res in label_results[i:i+num_threads]:
            writer.writerow(res)

    print("Sleeping")
    time.sleep(sleep_time)
